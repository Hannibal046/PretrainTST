""" PATCHTST model configuration"""

from transformers import PretrainedConfig

class PatchTSTConfig(PretrainedConfig):

    model_type = "patchtst"

    def __init__(
        self,
        num_channels=800,
        seq_len=96,
        label_len=96,
        stride=12,
        patch_len=12,
        d_model=768,
        ffn_dim=3072,
        dropout=0.2,
        num_heads=12,
        attention_dropout=0.0,
        num_layers=12,
        qkv_bias=True,
        init_std=0.2,
        head_dropout=0.0,
        mask_ratio=0.4,
        revin_affine=False,
        pooling_head='flatten',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pooling_head = pooling_head
        self.revin_affine = revin_affine
        self.dropout = dropout
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.seq_len = seq_len
        self.label_len = label_len
        self.stride = stride
        self.patch_len = patch_len
        self.d_model = d_model
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.qkv_bias = qkv_bias
        self.init_std = init_std
        self.head_dropout = head_dropout
        self.mask_ratio = mask_ratio,
        if self.seq_len % self.patch_len != 0:
            self.num_patches = int((self.seq_len-self.patch_len)/self.stride + 1) + 1
        else:
            self.num_patches = int((self.seq_len-self.patch_len)/self.stride + 1)
