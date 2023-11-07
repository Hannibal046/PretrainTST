from transformers import PreTrainedModel
from .configuration_patchtst import PatchTSTConfig
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

class RevIN(nn.Module):
    # code from https://github.com/ts-kim/RevIN, with minor modifications
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

class PatchTSTSelfAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self,config):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.attention_dropout = config.attention_dropout
        self.head_dim = self.d_model // self.num_heads

        if (self.head_dim * self.num_heads) != self.d_model:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.d_model}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=config.qkv_bias)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=config.qkv_bias)
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=config.qkv_bias)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=config.qkv_bias)
        self.attention_out_dropout = nn.Dropout(config.dropout)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states,
        attention_mask = None,
        output_attentions = False,
    ):
        bsz, tgt_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)


        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.d_model)

        attn_output = self.out_proj(attn_output)
        attn_output = self.attention_out_dropout(attn_output)

        return attn_output
    
class PatchTSTFFN(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model,config.ffn_dim)
        self.activation_fn = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(config.ffn_dim,config.d_model)

    def forward(self,hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class PatchTSTEncoderLayer(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.self_attention = PatchTSTSelfAttention(config)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.attention_norm = nn.BatchNorm1d(config.d_model)
        
        self.ffn = PatchTSTFFN(config)
        self.ffn_dropout = nn.Dropout(config.dropout)
        self.ffn_norm = nn.BatchNorm1d(config.d_model)
        
    def forward(
        self,
        hidden_states,
    ):  
        ## Self-Attention
        _,seq_len,d_model = hidden_states.shape
        residual = hidden_states 
        hidden_states = self.self_attention(hidden_states)
        hidden_states = residual + self.attention_dropout(hidden_states)
        ## view operation for BatchNorm
        hidden_states = hidden_states.view(-1,d_model)
        hidden_states = self.attention_norm(hidden_states)
        hidden_states = hidden_states.view(-1,seq_len,d_model)

        ## FFN
        residual = hidden_states
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + self.ffn_dropout(hidden_states)
        ## view operation for BatchNorm
        hidden_states = hidden_states.view(-1,d_model)
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = hidden_states.view(-1,seq_len,d_model)
        
        return hidden_states

class PatchTSTFlattenHead(nn.Module):
    def __init__(self,config):
        super().__init__()
        # self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(config.d_model,config.patch_len)
        self.dropout = nn.Dropout(config.head_dropout)
    def forward(
        self,hidden_states #[batch_size*num_channel,num_patch,d_model]
    ):  

        # hidden_states = hidden_states.permute(0,2,1)  
        # hidden_states = self.flatten(hidden_states)
        hidden_states = self.linear(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class PatchTSTAverageHead(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.linear = nn.Linear(config.d_model,config.label_len)
        self.dropout = nn.Dropout(config.head_dropout)
    def forward(
        self,hidden_states #[batch_size*num_feature,num_patch,patch_len]
    ):
        # hidden_states = hidden_states.permute(0,2,1)  
        hidden_states = hidden_states.mean(dim=1)
        hidden_states = self.linear(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states
    
class PatchTSTPreTrainedModel(PreTrainedModel):
    config_class = PatchTSTConfig
    base_model_prefix = "model"
    main_input_name = "input_values"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Embedding):
            module.weight.data.uniform_(-0.02, 0.02)

class PatchTSTEncoder(PatchTSTPreTrainedModel):
    def __init__(self,config):
        super().__init__(config)

        self.layers = nn.ModuleList(
            PatchTSTEncoderLayer(config) for _ in range(config.num_layers)
        )
        self.post_init()
    
    def forward(self,hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states
        
class PatchTSTModel(PatchTSTPreTrainedModel):
    def __init__(self,config):
        super().__init__(config)
        
        self.input_embedding = nn.Linear(config.patch_len,config.d_model)
        self.pos_embedding = nn.Embedding(config.num_patches,config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.encoder = PatchTSTEncoder(config)
        self.post_init()
    
    def forward(
        self,input_values # [bs*num_channels,num_patches,patch_len]
    ) :
        hidden_states = self.dropout(
                    self.input_embedding(input_values)
                    +self.pos_embedding(torch.arange(input_values.shape[1],device=input_values.device)))
        hidden_states = self.encoder(hidden_states)
        return hidden_states
    
class PatchTSTForTimeSeriesPrediction(PatchTSTPreTrainedModel):
    def __init__(self,config):
        super().__init__(config)
        self.patch_len = config.patch_len
        self.stride = config.stride
        self.padding = nn.ReplicationPad1d((0,config.stride))
        self.revin_layers = RevIN(config.num_channels,affine=config.revin_affine)
        self.model = PatchTSTModel(config)
        self.d_model = config.d_model
        self.mask_ratio = config.mask_ratio
        if config.pooling_head == 'flatten'  :self.head = PatchTSTFlattenHead(config)
        elif config.pooling_head == 'average':self.head = PatchTSTAverageHead(config)
        self.post_init()
        
    def patchify(self,input_values,seq_len):
        """
        input_values: [batch_size,num_channels,seq_len]
        """
        if seq_len % self.patch_len != 0:
            input_values =  self.padding(input_values)# [batch_size,num_channels,seq_len+self.stride]
        input_values = input_values.unfold(dimension=-1,size=self.patch_len,step=self.stride) # [batch_size,num_channels,num_patch,patch_len]
        return input_values
    
    def random_masking(self,patched_input_values):
        """
        patched_input_values: [batch_size,num_channels,num_patch,patch_len]
        """
        patched_input_values = patched_input_values.permute(0,2,1,3) #[bs x num_patch x num_channels x patch_len]
        bs, L, nvars, D = patched_input_values.shape
        x = patched_input_values.clone()

        len_keep = int(L * (1 - self.mask_ratio[0]))
            
        noise = torch.rand(bs, L, nvars,device=patched_input_values.device)  # noise in [0, 1], bs x L x nvars
            
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)                                     # ids_restore: [bs x L x nvars]

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep, :]                                              # ids_keep: [bs x len_keep x nvars]         
        x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, D))     # x_kept: [bs x len_keep x nvars  x patch_len]

        # removed x
        x_removed = torch.zeros(bs, L-len_keep, nvars, D, device=patched_input_values.device)                 # x_removed: [bs x (L-len_keep) x nvars x patch_len]
        x_ = torch.cat([x_kept, x_removed], dim=1)                                          # x_: [bs x L x nvars x patch_len]

        # combine the kept part and the removed one
        x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,1,D)) # x_masked: [bs x num_patch x nvars x patch_len]
        x_masked = x_masked.permute(0,2,1,3)
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([bs, L, nvars], device=x.device)                                  # mask: [bs x num_patch x nvars]
        mask[:, :len_keep, :] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)  # [bs x num_patch x num_channel]
        mask = mask.squeeze(2)# [bs x num_patch]
        mask = mask.bool()
        return x_masked, mask

    def forward(
        self,
        input_values, # [batch_size,seq_len]->[bs, seq_len, num_channels]
    ):  
        if input_values.dim() == 2:# [batch_size,seq_len]
            input_values=input_values.unsqueeze(2) # [bs, seq_len, num_channels]

        input_values = self.revin_layers(input_values,'norm')
        input_values = input_values.permute(0,2,1) # [batch_size,num_channels,seq_len]
        seq_len = input_values.shape[2]
        patched_input_values = self.patchify(input_values, seq_len) #[batch_size,num_channels,num_patch,patch_len]

        masked_patched_input_values, mask = self.random_masking(patched_input_values)#[batch_size,num_channels,num_patch,patch_len]# [bs x num_patch]
        batch_size,num_channels,num_patch,patch_len = masked_patched_input_values.shape
        hidden_states = self.model(masked_patched_input_values.view(-1,num_patch,patch_len)) # [bs*num_channels,num_patches,d_model]
        # output = torch.reshape(hidden_states, (-1, num_channels, num_patch, self.d_model))# [bs,num_channels,num_patches,d_model]
        output = self.head(hidden_states) # [batch_size*num_channels,num_patches,patch_len]
        # output = output.view(batch_size,num_channels,-1)
        # output = torch.reshape(hidden_states, (-1,num_channels, num_patch, self.d_model))
        output = output.view(batch_size,num_channels,-1)#[batch_size,num_channels,num_patch*patch_len]
        output = self.revin_layers(output.permute(0,2,1),'denorm')#[batch_size,num_patch*patch_len, num_channels]
        output=output.squeeze(2).view(batch_size,num_patch, patch_len)#[batch_size,num_patch, patch_len]
        if torch.sum(torch.isinf(output)).item()!=0:
            pdb.set_trace()
        return patched_input_values.squeeze(1), output, mask
    
