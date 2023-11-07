## built-in
import math,logging,json,random,functools,os,csv
import types
from functools import partial
os.environ["WANDB_IGNORE_GLOBS"]='*.bin' ## not upload ckpt to wandb cloud
import pdb
## third-party
from accelerate import Accelerator
from accelerate.logging import get_logger
import transformers
transformers.logging.set_verbosity_error()
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

## own
from utils import (
    get_yaml_file,
    set_seed,
    MAE,
    MSE,
)
from model import (
    PatchTSTForTimeSeriesPrediction,PatchTSTConfig,
)
torch.autograd.set_detect_anomaly(True)
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    ## adding args here for more control from CLI is possible
    parser.add_argument("--config_file",default='/data/shuqili/PretrainTST-main/config/run_pretrain.yaml')

    parser.add_argument("--per_device_train_batch_size",type=int)
    parser.add_argument("--per_device_eval_batch_size",type=int)
    parser.add_argument("--gradient_accumulation_steps",type=int)
    parser.add_argument("--lr",type=float)
    parser.add_argument("--weight_decay",type=float)
    parser.add_argument("--max_grad_norm",type=float)

    parser.add_argument("--seq_len",type=int)
    parser.add_argument("--label_len",type=int)
    parser.add_argument("--stride",type=int)
    parser.add_argument("--patch_len",type=int)
    parser.add_argument("--num_patience",type=int,default=20)
    parser.add_argument("--max_train_epochs",type=int)
    parser.add_argument("--mask_ratio",type=float,default=0.4)

    parser.add_argument("--exp_name",default='pretrain_patchtst')

    args = parser.parse_args()

    yaml_config = get_yaml_file(args.config_file)
    args_dict = {k:v for k,v in vars(args).items() if v is not None}
    yaml_config.update(args_dict)
    args = types.SimpleNamespace(**yaml_config)
    return args

def create_patch(xb, patch_len, stride):
    """
    xb: [bs x seq_len]
    """

    seq_len = xb.shape[1]
    num_patch = (max(seq_len, patch_len)-patch_len) // stride + 1
    tgt_len = patch_len  + stride*(num_patch-1)
    s_begin = seq_len - tgt_len
        
    xb = xb[:, s_begin:]                                                    # xb: [bs x tgt_len]
    xb = xb.unfold(dimension=1, size=patch_len, step=stride)                 # xb: [bs x num_patch x patch_len]
    return xb, num_patch

def random_masking(xb, mask_ratio):
    """
    xb: [bs x num_patch x n_vars x patch_len]
    """
    if xb.dim()==3:
        xb = xb.unsqueeze(2)
    bs, L, nvars, D = xb.shape
    x = xb.clone()
    
    len_keep = int(L * (1 - mask_ratio))
        
    noise = torch.rand(bs, L, nvars,device=xb.device)  # noise in [0, 1], bs x L x nvars
        
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)                                     # ids_restore: [bs x L x nvars]

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep, :]                                              # ids_keep: [bs x len_keep x nvars]         
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, D))     # x_kept: [bs x len_keep x nvars  x patch_len]

    # removed x
    x_removed = torch.zeros(bs, L-len_keep, nvars, D, device=xb.device)                 # x_removed: [bs x (L-len_keep) x nvars x patch_len]
    x_ = torch.cat([x_kept, x_removed], dim=1)                                          # x_: [bs x L x nvars x patch_len]

    # combine the kept part and the removed one
    x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,1,D)) # x_masked: [bs x num_patch x nvars x patch_len]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([bs, L, nvars], device=x.device)                                  # mask: [bs x num_patch x nvars]
    mask[:, :len_keep, :] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)                                  # [bs x num_patch x nvars]

    return x_masked.squeeze(2), x_kept.squeeze(2), mask.squeeze(2), ids_restore.squeeze(2)
 

def masking_collate_fn(samples, mask_ratio, patch_len, stride):
    samples = torch.tensor(np.array(samples))
    patched_batch, num_patch = create_patch(samples, patch_len, stride)  # patched_batch: [bs x num_patch x patch_len]
    x_masked, _, mask, _ = random_masking(patched_batch, mask_ratio)   # xb_mask: [bs x num_patch  x patch_len]
    mask = mask.bool()    # mask: [bs x num_patch x n_vars]
    return patched_batch, x_masked, mask       # learner.xb: masked 4D tensor
    
class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self,file_path,seq_len,label_len,stage):
        self.seq_len = seq_len
        self.label_len = label_len
        self.sample_size = 226478375
        # self.scaler = StandardScaler()

        df = np.memmap(file_path, dtype='float32', mode='r',shape=(self.seq_len, self.sample_size))
        num_samples = len(df[0])
        
        num_train = int(num_samples*0.7)
        num_test = 0
        num_dev = num_train-num_test
        
        borders = {
            "train": [0, num_train],
            "dev":   [num_train, num_train+num_dev],
            "test":  [num_train+num_dev, num_samples]
        }
        # train_data = df[borders['train'][0]:borders['train'][1]]
        # self.scaler.fit(train_data.values)
        # df = self.scaler.transform(df.values)

        self.data = df[:, borders[stage][0]:borders[stage][1]]
        
    def __len__(self):
       return self.data.shape[1]
    
    def __getitem__(self, idx):

        seq_x = self.data[:, idx]
        # seq_y = self.data[:, idx]

        return seq_x
    

    
       
 
def validate(model,dataloader,accelerator):
    model.eval()
    preds,labels,losses = [],[],[]
    for inputs in dataloader:
        with torch.no_grad():
            inputs = inputs.float()
            label, pred, mask = model(inputs)
            label=label.detach().cpu().numpy()
            pred=pred.detach().cpu().numpy()
            mask=mask.detach().cpu().numpy()
            
            loss = (pred - label) ** 2
            loss = loss.mean(dim=-1)
            loss = (loss * mask).sum() / mask.sum()
    
        preds.append(pred)
        labels.append(label)
        losses.append(loss)
    
    pdb.set_trace() 
    if accelerator.use_distributed and accelerator.num_processes>1:
        preds_from_all_gpus = [None for _ in range(accelerator.num_processes)] 
        dist.all_gather_object(preds_from_all_gpus,preds)
        preds = [x for y in preds_from_all_gpus for x in y]

        labels_from_all_gpus = [None for _ in range(accelerator.num_processes)] 
        dist.all_gather_object(labels_from_all_gpus,labels)
        labels = [x for y in labels_from_all_gpus for x in y]
    
    preds  = np.concatenate(preds,axis=0)[:len(dataloader.dataset)]
    labels = np.concatenate(labels,axis=0)[:len(dataloader.dataset)]
    
    final_loss = np.mean(losses)
            
    return final_loss

def main():
    args = parse_args()
    set_seed(args.seed)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with='wandb',
        mixed_precision='no',
    )

    accelerator.init_trackers(
        project_name=args.exp_name, 
        config=args,
    )
    if accelerator.is_local_main_process:
        wandb_tracker = accelerator.get_tracker("wandb")
        LOG_DIR = wandb_tracker.run.dir


    config = PatchTSTConfig(num_channels=args.num_channels,seq_len=args.seq_len,label_len=args.label_len,stride=args.stride,patch_len=args.patch_len)
    model  = PatchTSTForTimeSeriesPrediction(config)
    model.train()

    train_dataset = TimeSeriesDataset(args.data_file,args.seq_len,args.label_len,'train')

    # collate_fn = partial(masking_collate_fn,mask_ratio=args.mask_ratio,patch_len=args.patch_len,stride = args.stride)
    # collate_fn = None
    dev_dataset = TimeSeriesDataset(args.data_file,args.seq_len,args.label_len,'dev')
    test_dataset = TimeSeriesDataset(args.data_file,args.seq_len,args.label_len,'test')

    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=args.per_device_train_batch_size,shuffle=True,drop_last=False,num_workers=4,pin_memory=True)
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset,batch_size=args.per_device_eval_batch_size,shuffle=False,drop_last=False,num_workers=4,pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=args.per_device_eval_batch_size,shuffle=False,drop_last=False,num_workers=4,pin_memory=True)


    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters,lr=args.lr)
    
    model, optimizer, train_dataloader, dev_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, dev_dataloader,test_dataloader
    )
    
    BEST_DEV_MSE=100
    BEST_TEST_MSE=100
    BEST_TEST_MAE=100
    PATIENCE = args.num_patience
    SHOULD_BREAK=False
    NUM_UPDATES_PER_EPOCH = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    MAX_TRAIN_STEPS = NUM_UPDATES_PER_EPOCH * args.max_train_epochs
    MAX_TRAIN_EPOCHS = math.ceil(MAX_TRAIN_STEPS / NUM_UPDATES_PER_EPOCH)
    TOTAL_TRAIN_BATCH_SIZE = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    EVAL_STEPS = args.val_check_interval if isinstance(args.val_check_interval,int) else int(args.val_check_interval * NUM_UPDATES_PER_EPOCH)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer = optimizer,
                                            steps_per_epoch = NUM_UPDATES_PER_EPOCH,
                                            pct_start = args.pct_start,
                                            epochs = MAX_TRAIN_EPOCHS,
                                            max_lr = args.lr)
    progress_bar_postfix_dict = {}
    
    logger.info("***** Running training *****")
    logger.info(f"  Num train examples = {len(train_dataset)}")
    logger.info(f"  Num dev examples = {len(dev_dataset)}")
    logger.info(f"  Num test examples = {len(test_dataset)}")
    logger.info(f"  Num Epochs = {MAX_TRAIN_EPOCHS}")
    logger.info(f"  Per device train batch size = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {TOTAL_TRAIN_BATCH_SIZE}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {MAX_TRAIN_STEPS}")
    logger.info(f"  Per device eval batch size = {args.per_device_eval_batch_size}")
    logger.info(f"  Model Size = {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.6f} M")
    completed_steps = 0
    trained_samples = 0
    progress_bar = tqdm(range(MAX_TRAIN_STEPS), disable=not accelerator.is_local_main_process,ncols=150)

    for epoch in range(MAX_TRAIN_EPOCHS):
        progress_bar.set_description(f"epoch: {epoch+1}/{MAX_TRAIN_EPOCHS}")
        for step, inputs in enumerate(train_dataloader):#samples, x_masked, mask [bs, num_patch, patch_len] num_patch=seq_len, patch_len=dim
            trained_samples += inputs.shape[0]
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    inputs = inputs.float()
                    labels, preds, mask = model(inputs)
                    loss = (preds - labels) ** 2
                    loss = loss.mean(dim=-1)
                    loss = (loss * mask).sum() / mask.sum()
                    ## TODO: calcuate masked loss
                    # loss = F.mse_loss(model(inputs),labels)
                accelerator.backward(loss)
                ## one optimization step
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    progress_bar_postfix_dict.update(dict(loss=f"{loss:.4f}",lr=f"{lr_scheduler.get_last_lr()[0]:6f}"))
                    progress_bar.set_postfix(progress_bar_postfix_dict)
                    completed_steps += 1
                    if hasattr(args,'max_grad_norm'): accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    if not accelerator.optimizer_step_was_skipped:lr_scheduler.step()
                    optimizer.zero_grad()
                    accelerator.log({"trained_samples": trained_samples}, step=completed_steps)
                    accelerator.log({"training_loss": loss}, step=completed_steps)
                    accelerator.log({"lr": lr_scheduler.get_last_lr()[0]}, step=completed_steps)
                    
                    if completed_steps % EVAL_STEPS == 0:
                        dev_mse,dev_mae   = validate(model,dev_dataloader,accelerator)
                        test_mse,test_mae = validate(model,test_dataloader,accelerator)
                        model.train()
                        accelerator.log({"epoch": epoch+1}, step=completed_steps)
                        accelerator.log({"dev_mse": dev_mse}, step=completed_steps)
                        if dev_mse < BEST_DEV_MSE:
                            PATIENCE = args.num_patience
                            BEST_DEV_MSE = dev_mse
                            BEST_TEST_MAE = test_mae
                            BEST_TEST_MSE = test_mse
                            accelerator.log({"test_mse":test_mse}, step=completed_steps)
                            progress_bar_postfix_dict.update(dict(test_mse=f"{test_mse:.4f}"))
                            accelerator.wait_for_everyone()
                            if accelerator.is_local_main_process:
                                unwrapped_model = accelerator.unwrap_model(model)
                                unwrapped_model.save_pretrained(os.path.join(LOG_DIR,"ckpt"))
                            accelerator.wait_for_everyone()
                        else:
                            PATIENCE -= 1
                            if PATIENCE <= 0:
                                SHOULD_BREAK = True
                                break
        if SHOULD_BREAK:break       

    accelerator.log({"final mse":BEST_TEST_MSE}, step=completed_steps)
    if accelerator.is_local_main_process:
        wandb_tracker.finish()
        print(f"test mse:{BEST_TEST_MSE:.4f} test_mae:{BEST_TEST_MAE:.4f}")
    accelerator.end_training()

if __name__ == '__main__':
    main()