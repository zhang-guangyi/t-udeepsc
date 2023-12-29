import math
import numpy as np
from timm.models.registry import register_model
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from transformers.models.bert.modeling_bert import BertEmbeddings
import torch
import torch.nn as nn
from channel import *
from modulator import *
from functools import partial
import torch.nn.functional as F
from utils import batch_index_select
from transformers import BertForSequenceClassification, BertModel


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }

# def noise_gen(is_train):
#     min_snr, max_snr = -6, 18
#     diff_snr = max_snr - min_snr
    
#     min_var, max_var = 10**(-min_snr/20), 10**(-max_snr/20)
#     diff_var = max_var - min_var
#     if is_train:
#         # b = torch.bernoulli(1/5.0*torch.ones(1))
#         # if b > 0.5:
#         #     channel_snr = torch.FloatTensor([20])
#         # else:               
#         #     channel_snr = torch.rand(1)*diff_snr+min_snr
#         # noise_var = 10**(-channel_snr/20)
#         # noise_var = torch.rand(1)*diff_var+min_var  
#         # channel_snr = 10*torch.log10((1/noise_var)**2)
#         # channel_snr = torch.rand(1)*diff_snr+min_snr
#         # noise_var = 10**(-channel_snr/20)
#         channel_snr = torch.FloatTensor([12])
#         noise_var = torch.FloatTensor([1]) * 10**(-channel_snr/20)  
#     else:
#         channel_snr = torch.FloatTensor([12])
#         noise_var = torch.FloatTensor([1]) * 10**(-channel_snr/20)  
#     return channel_snr, noise_var 

def noise_gen(is_train):
    # min_snr, max_snr = -7, 14
    # diff_snr = max_snr - min_snr
    
    # min_std, max_std = 10**(-min_snr/20), 10**(-max_snr/20)
    # diff_var = max_std - min_std
    snr_list = np.arange(-6, 13, 2)
    if is_train:
        # b = torch.bernoulli(1/5.0*torch.ones(1))
        # if b > 0.5:
        #     channel_snr = torch.FloatTensor([20])
        # else:               
        #     channel_snr = torch.rand(1)*diff_snr+min_snr
        # noise_var = 10**(-channel_snr/20)
        # noise_var = torch.rand(1)*diff_var+min_var  
        # channel_snr = 10*torch.log10((1/noise_var)**2)
        # channel_snr = torch.rand(1)*diff_snr+min_snr
        channel_snr = torch.FloatTensor([1]) * np.random.choice(snr_list)
        # channel_snr = torch.FloatTensor([1]) * 12
        noise_std = 10**(-channel_snr/20)
    else:
        channel_snr = torch.FloatTensor([2])
        # channel_snr = torch.FloatTensor([1]) * np.random.choice([-6, 12])
        # print(channel_snr)        
        noise_std = torch.FloatTensor([1]) * 10**(-channel_snr/20)  
    return channel_snr, noise_std  



class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def softmax_with_policy(self, attn, policy, eps=1e-6):
        B, N, _ = policy.size()
        B, H, N, N = attn.size()
        attn_policy = policy.reshape(B, 1, 1, N)  # * policy.reshape(B, 1, N, 1)
        eye = torch.eye(N, dtype=attn_policy.dtype, device=attn_policy.device).view(1, 1, N, N)
        attn_policy = attn_policy + (1.0 - attn_policy) * eye
        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att
        # attn = attn.exp_() * attn_policy
        # return attn / attn.sum(dim=-1, keepdim=True)

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps/N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)

    def forward(self, x, policy):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if policy is None:
            attn = attn.softmax(dim=-1)
        else:
            attn = self.softmax_with_policy(attn, policy)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, policy=None):
        x = x + self.drop_path(self.attn(self.norm1(x), policy=policy))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
def get_sinusoid_encoding_table(n_position, d_hid): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return torch.FloatTensor(sinusoid_table).unsqueeze(0) 


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1) # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model)) #math.log(math.exp(1)) = 1
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) #[1, max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        x = self.dropout(x)
        return x

class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        
        self.dense = nn.Linear(d_model, d_model)
        
        #self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query = self.wq(query).view(nbatches, -1, self.num_heads, self.d_k)
        query = query.transpose(1, 2)
        
        key = self.wk(key).view(nbatches, -1, self.num_heads, self.d_k)
        key = key.transpose(1, 2)
        # print(key.shape)
        value = self.wv(value).view(nbatches, -1, self.num_heads, self.d_k)
        value = value.transpose(1, 2)

        x, self.attn = self.attention(query, key, value, mask=mask)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.num_heads * self.d_k)
             
        x = self.dense(x)
        x = self.dropout(x)
        
        return x
    
    def attention(self, query, key, value, mask=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        #print(mask.shape)
        if mask is not None:
            scores += (mask * -1e9)
            # attention weights
        p_attn = F.softmax(scores, dim = -1)
        return torch.matmul(p_attn, value), p_attn

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = F.relu(x)
        x = self.w_2(x)
        x = self.dropout(x) 
        return x

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, d_model, num_heads, dff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_mha = MultiHeadedAttention(num_heads, d_model, dropout = 0.1)
        self.src_mha = MultiHeadedAttention(num_heads, d_model, dropout = 0.1)
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout = 0.1)
        
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)
        
        #self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, look_ahead_mask, trg_padding_mask):
        "Follow Figure 1 (right) for connections."
        attn_output = self.self_mha(x, x, x, look_ahead_mask)
        x = self.layernorm1(x + attn_output)
     
        src_output = self.src_mha(x, memory, memory, trg_padding_mask) # q, k, v
        x = self.layernorm2(x + src_output)
        
        fnn_output = self.ffn(x)
        x = self.layernorm3(x + fnn_output)
        return x


class Decoder(nn.Module):
    def __init__(self, depth=4, embed_dim=128, num_heads=4, dff=128, drop_rate=0.1):
        super(Decoder, self).__init__()
    
        self.d_model = embed_dim
        self.pos_encoding = PositionalEncoding(embed_dim, drop_rate, 50)
        self.dec_layers = nn.ModuleList([DecoderLayer(embed_dim, num_heads, dff, drop_rate) 
                                            for _ in range(depth)])
        
    def forward(self, x, memory, look_ahead_mask=None, trg_padding_mask=None):
        for dec_layer in self.dec_layers:
            x = dec_layer(x, memory, look_ahead_mask, trg_padding_mask)  
        return x




class ViTEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_learnable_pos_emb=False):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.linear_embed_vqa = nn.Linear(2048, self.embed_dim)
        self.linear_embed_msa = nn.Linear(35, self.embed_dim)
        num_patches = self.patch_embed.num_patches

        # TODO: Add the cls token
        self.cls_token = {}
        self.cls_token['imgr'] = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token['imgc'] = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token['vqa'] = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token['msa'] = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.task_embedd = {}
        self.task_embedd['imgr'] = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.task_embedd['imgc'] = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.task_embedd['vqa'] = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.task_embedd['msa'] = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings 
            self.pos_embed = get_sinusoid_encoding_table(num_patches + 1, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)
        for key in self.cls_token.keys():
            trunc_normal_(self.cls_token[key], std=.02)
            trunc_normal_(self.task_embedd[key], std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x, ta_perform):
        if ta_perform.startswith('vqa'):
            x = self.linear_embed_vqa(x)
            batch_size = x.shape[0]
            cls_tokens = self.cls_token[ta_perform].expand(batch_size, -1, -1).to(x.device) 
            task_embedd = self.task_embedd[ta_perform].expand(batch_size, -1, -1).to(x.device)
            x = torch.cat((cls_tokens, x, task_embedd), dim=1)
        elif ta_perform.startswith('msa'):
            x = self.linear_embed_msa(x)
            batch_size = x.shape[0]
            cls_tokens = self.cls_token[ta_perform].expand(batch_size, -1, -1).to(x.device)
            task_embedd = self.task_embedd[ta_perform].expand(batch_size, -1, -1).to(x.device)
            x = torch.cat((cls_tokens, x, task_embedd), dim=1)
        elif ta_perform.startswith('img'):
            x = self.patch_embed(x)
            batch_size = x.shape[0]
            cls_tokens = self.cls_token[ta_perform].expand(batch_size, -1, -1).to(x.device) 
            task_embedd = self.task_embedd[ta_perform].expand(batch_size, -1, -1).to(x.device) 
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
            x = torch.cat((x, task_embedd), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward(self, x, ta_perform):
        x = self.forward_features(x, ta_perform)
        return x


    



class SPTEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_learnable_pos_emb=False):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.linear_embed = nn.Linear(74, self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.task_embedd = nn.Parameter(torch.zeros(1, 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.task_embedd, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    def forward_features(self, x, ta_perform):
        if ta_perform.startswith('msa'):
            x = self.linear_embed(x)
            batch_size = x.shape[0]
            cls_tokens = self.cls_token.expand(batch_size, -1, -1).to(x.device) 
            task_embedd = self.task_embedd.expand(batch_size, -1, -1).to(x.device) 
            x = torch.cat((cls_tokens, x, task_embedd), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward(self, x, ta_perform=None):
        x = self.forward_features(x, ta_perform)
        return x
    


def get_rate_table(noise_std):  
    """ 
    You can define the rate-snr curve according to requiremets 
    """
    noise_snr = 10*torch.log10((1/noise_std)**2)
    compression_ratio = None
    if noise_snr < 4.:
        compression_ratio = 0.5
    elif noise_snr < 12. and noise_snr >= 4.:
        compression_ratio = 0.3
    elif noise_snr >= 12.:
        compression_ratio = 0.2
    return compression_ratio



class rho_function(nn.Module):
    def __init__(self, intermediate_dim):
        super(rho_function, self).__init__()
        self.f1 = rho_layer(1,16)
        self.f2 = rho_layer(16,16)
        self.f3 = rho_layer(16,16)
        self.f4 = rho_layer(16,intermediate_dim)
        
    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        return x

class ViTEncoder_FSM(nn.Module):
    """ Vision Transformer with FIM
    """
    def __init__(self, img_size=224, patch_size=32, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_learnable_pos_emb=False,num_FSM=2):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.depth = depth
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.linear_embed_vqa = nn.Linear(2048, self.embed_dim)
        self.linear_embed_msa = nn.Linear(35, self.embed_dim)
        num_patches = self.patch_embed.num_patches

        # TODO: Add the cls token
        self.cls_token = {}
        self.cls_token['imgr'] = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token['imgc'] = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token['vqa'] = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token['msa'] = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.task_embedd = {}
        self.task_embedd['imgr'] = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.task_embedd['imgc'] = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.task_embedd['vqa'] = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.task_embedd['msa'] = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings 
            self.pos_embed = get_sinusoid_encoding_table(num_patches + 1, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)
        for key in self.cls_token.keys():
            trunc_normal_(self.cls_token[key], std=.02)
            trunc_normal_(self.task_embedd[key], std=.02)
        self.apply(self._init_weights)

        ####  Configurations of FIM
        self.num_FSM = num_FSM
        self.mask_num_imgc = int((img_size/patch_size)**2)  + 1
        self.mask_num_vqa = 101
       
         
        self.FSM_Dict = nn.ModuleDict({
            'imgc': nn.ModuleList([FSM(self.mask_num_imgc, embed_dim) for _ in range(self.num_FSM)]),
            'vqa': nn.ModuleList([FSM(self.mask_num_vqa, embed_dim) for _ in range(self.num_FSM)])
        })
        
        self.RHO_Dict = nn.ModuleDict({
            'imgc': rho_function(1),
            'vqa': rho_function(1)
        })
        self.noise_func = nn.Sequential(nn.Linear(1,16),nn.ReLU(),nn.Linear(16,64),
                        nn.ReLU(), nn.Linear(64, embed_dim//2),nn.ReLU())   
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x, ta_perform, noise_std):
        if ta_perform.startswith('vqa'):
            x = self.linear_embed_vqa(x)
            batch_size = x.shape[0]
            cls_tokens = self.cls_token[ta_perform].expand(batch_size, -1, -1).to(x.device) 
            task_embedd = self.task_embedd[ta_perform].expand(batch_size, -1, -1).to(x.device)
            x = torch.cat((cls_tokens, x, task_embedd), dim=1)
        elif ta_perform.startswith('msa'):
            x = self.linear_embed_msa(x)
            batch_size = x.shape[0]
            cls_tokens = self.cls_token[ta_perform].expand(batch_size, -1, -1).to(x.device)
            task_embedd = self.task_embedd[ta_perform].expand(batch_size, -1, -1).to(x.device)
            x = torch.cat((cls_tokens, x, task_embedd), dim=1)
        elif ta_perform.startswith('img'):
            x = self.patch_embed(x)
            batch_size = x.shape[0]
            cls_tokens = self.cls_token[ta_perform].expand(batch_size, -1, -1).to(x.device) 
            task_embedd = self.task_embedd[ta_perform].expand(batch_size, -1, -1).to(x.device) 
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
            x = torch.cat((x, task_embedd), dim=1)
            
        for d in range(self.depth-self.num_FSM):
            x = self.blocks[d](x)
        m_group = []                
        device, batch_size = x.device, x.shape[0]
        mask_num = self.mask_num_imgc if ta_perform.startswith('img') else self.mask_num_vqa
        prev_m = torch.ones(batch_size, mask_num, 1, dtype=x.dtype, device=device)    
        noise_feature = self.noise_func(noise_std)    
          
        rho = self.RHO_Dict[ta_perform](noise_std)
        # rho = torch.tensor(get_rate_table(noise_std, ta_perform))
        # rho = torch.tensor(0.7)
        rho_list = [rho**(i+1) for i in range(self.num_FSM)]
        for d in range(self.depth-self.num_FSM,self.depth):
            index = d-self.depth+self.num_FSM
            x, curr_m = self.FSM_Dict[ta_perform][index](x, noise_feature, prev_m, num_skip=1, ratio=rho_list[index])
            if self.training:
                m_group.append(curr_m)
                cls_m = torch.ones(batch_size, 1, 1, dtype=x.dtype, device=x.device)     
                policy = torch.cat([cls_m, curr_m], dim=1)
                x = self.blocks[d](x, policy)
            else:
                x = self.blocks[d](x)
            prev_m = curr_m
        
        x = self.norm(x)     
        return x, m_group, rho_list
    


class TextEncoder_FSM(nn.Module):
    """ text Transformer with FIM
    """
    def __init__(self, in_chans=3, num_classes=0, embed_dim=768, depth=12,num_heads=12, 
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_learnable_pos_emb=False, mode='tiny', num_FSM=2):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        bert_ckpt = f"/Data1/zhangguangyi/SemanRes2/JSACCode/UDeepSC_Base/pretrained_models/bert-{mode}"
        # self.embeddings = BertEmbeddings( bert_ckpt)
        temp = BertModel.from_pretrained(bert_ckpt)
        self.embeddings = temp.embeddings
        temp = None
        self.depth = depth
        # TODO: Add the cls token
        self.cls_token = {}
        self.cls_token['textr'] = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token['textc'] = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token['vqa'] = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token['msa'] = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.task_embedd = {}
        self.task_embedd['textr'] = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.task_embedd['textc'] = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.task_embedd['vqa'] = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.task_embedd['msa'] = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings 
            self.pos_embed = get_sinusoid_encoding_table(66 + 1, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)
        for key in self.cls_token.keys():
            trunc_normal_(self.cls_token[key], std=.02)
            trunc_normal_(self.task_embedd[key], std=.02)
        self.apply(self._init_weights)

        ####  Configurations of FIM
        self.num_FSM = num_FSM
        self.mask_num_textc = 66
        self.mask_num_vqa = 16
       
        
        self.FSM_Dict = nn.ModuleDict({
            'textc': nn.ModuleList([FSM(self.mask_num_textc, embed_dim) for _ in range(self.num_FSM)]),
            'vqa': nn.ModuleList([FSM(self.mask_num_vqa, embed_dim) for _ in range(self.num_FSM)])
        })
        
        self.RHO_Dict = nn.ModuleDict({
            'textc': rho_function(1),
            'vqa': rho_function(1)})
        self.noise_func = nn.Sequential(nn.Linear(1,16),nn.ReLU(),nn.Linear(16,64),
                        nn.ReLU(), nn.Linear(64, embed_dim//2),nn.ReLU())   
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x, ta_perform, noise_std):
        x = self.embeddings(x)
        batch_size = x.shape[0]
        # cls_tokens = self.cls_token[ta_perform].expand(batch_size, -1, -1).to(x.device) 
        
        # x = torch.cat((cls_tokens, x), dim=1)
        # x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
        task_embedd = self.task_embedd[ta_perform].expand(batch_size, -1, -1).to(x.device)
        x = torch.cat((x, task_embedd), dim=1)

        # print(x.shape)
        for d in range(self.depth-self.num_FSM):
            x = self.blocks[d](x)
        m_group = []                
        device, batch_size = x.device, x.shape[0]
        mask_num = self.mask_num_textc if ta_perform.startswith('text') else self.mask_num_vqa
        prev_m = torch.ones(batch_size, mask_num, 1, dtype=x.dtype, device=device)    
        noise_feature = self.noise_func(noise_std)
        
        rho = self.RHO_Dict[ta_perform](noise_std)
        # rho = torch.tensor(get_rate_table(noise_std,ta_perform))
        # rho = torch.tensor(0.7)
        rho_list = [rho**(i+1) for i in range(self.num_FSM)]
        for d in range(self.depth-self.num_FSM,self.depth):
            index = d-self.depth+self.num_FSM
            x, curr_m = self.FSM_Dict[ta_perform][index](x, noise_feature, prev_m, num_skip=1, ratio=rho_list[index])
            if self.training:
                m_group.append(curr_m)
                cls_m = torch.ones(batch_size, 1, 1, dtype=x.dtype, device=x.device)     
                policy = torch.cat([cls_m, curr_m], dim=1)
                x = self.blocks[d](x,policy=policy)
            else:
                x = self.blocks[d](x)
            prev_m = curr_m
        
        x = self.norm(x)     
        return x, m_group, rho_list


class mask_gen(nn.Module):
    def __init__(self, embed_dim=384):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )
        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim + embed_dim // 2, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, policy, noise_feature):
        x = self.in_conv(x)
        B, N, C = x.size()
        local_x = x[:,:, :C//2]
        global_x = (x[:,:, C//2:] * policy).sum(dim=1, keepdim=True) / torch.sum(policy, dim=1, keepdim=True)
 
        x = torch.cat([local_x, global_x.expand(B, N, C//2), noise_feature.expand(B,N,C//2)], dim=-1)
        return self.out_conv(x)
    

class rho_layer(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(rho_layer, self).__init__()
        self.H = nn.Parameter(torch.ones(output_channel, input_channel))
        self.b = nn.Parameter(torch.ones(output_channel))
        self.H.data.normal_(0, 0.1)
        self.b.data.normal_(0, 0.1)

    def forward(self, x):
        H = torch.abs(self.H)
        x = F.linear(x,H)
        return torch.tanh(x* 1.7)



class rho_function(nn.Module):
    def __init__(self, intermediate_dim):
        super(rho_function, self).__init__()
        self.f1 = rho_layer(1,16)
        self.f2 = rho_layer(16,16)
        self.f3 = rho_layer(16,16)
        self.f4 = rho_layer(16,intermediate_dim)
        
    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        return x
    
    
class FSM(nn.Module):
    def __init__(self, mask_num=64, embed_dim=384):
        super().__init__()
        self.mask_generator = mask_gen(embed_dim)
        self.mask_num = mask_num
        
    def forward(self, input_feature, 
                noise_feature, 
                prev_m, 
                num_skip=1,
                ratio=1.): 
        batch_size = input_feature.shape[0]
        prob = self.mask_generator(input_feature[:,num_skip:,:], prev_m, noise_feature).reshape(batch_size, -1, 2)  # Z^g Z^l Z^c
        temp_feature = torch.zeros_like(input_feature).to(input_feature.device)
        if self.training:
            curr_m = F.gumbel_softmax(prob, hard=True)[:, :, 0:1] * prev_m
            return input_feature, curr_m
        else:
          
            prob_kept = prob[:,:,0]    # Obtain the first one
            # prob_kept = torch.randn(prob_kept.shape).cuda()
            num_kept = int(np.round(self.mask_num * ratio))
            curr_m = F.gumbel_softmax(prob, hard=True)[:, :, 0:1] * prev_m
            keep_index = torch.argsort(prob_kept, dim=1, descending=True)[:, :num_kept]    
            print(keep_index[0])
            skip_index = torch.zeros(batch_size, num_skip, dtype=keep_index.dtype, device=keep_index.device)
            full_m = torch.cat([skip_index, keep_index + num_skip], dim=1)
            input_feature = batch_index_select(input_feature, full_m)
            curr_m = batch_index_select(prev_m, keep_index)
          
            return input_feature, curr_m





def array_to_binaryarray(input_array, num_bits):
    binary_matrix = (np.right_shift(input_array[:, None], np.arange(num_bits)[::-1]) & 1).astype(np.int32)
    return binary_matrix

def binaryarray_to_array(binary_array, num_bits):
    powers_of_2 = 2 ** np.arange(num_bits)[::-1]
    return np.sum(binary_array * powers_of_2, axis=1)



# Settings
M =16  # modulation order
        # signal-to-noise ratio
channel = 'awgn'  # channel type
mapping_table, demapping_table = qam_mod(M)
print(mapping_table)

def commun_sim(data_dec, snr=18, quan_bits=8):
    data_dec = data_dec.cpu().numpy().flatten()
    data_bin = array_to_binaryarray(data_dec, quan_bits)

    shape = data_bin.shape
    tx_bits = np.hstack(data_bin)
    
    # Communication Process
    tx_symbols = qam_mapper(tx_bits, mapping_table)
    # Wireless Channel
    rx_symbols = channel_Awgn(tx_symbols, snr=snr)
    # rx_symbols = channel_Rayleigh(tx_symbols, snr=snr)
    # rx_symbols = channel_Rician(tx_symbols, snr=snr)
    # M-QAM Demodulation
    rx_bits = qam_demapper(rx_symbols, demapping_table)
    rx_bits = rx_bits[: len(tx_bits)]
    # # Calculate BER
    # ber = bit_error_rate(tx_bits, rx_bits)
    # print(f"Bit Error Rate: {ber}")
    data_recover = binaryarray_to_array(rx_bits.reshape(shape), quan_bits)
    # print(data_recover)
    # print(f"Data error number {np.sum(data_dec!=data_recover)/len(data_recover)}" )
    return data_recover


class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 quan_bits: int = 4,
                 beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta
        self.quan_bits = quan_bits
        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents, snr=5):
        latents = latents # [B x L x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BL x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BL x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BL, 1]
        shape = encoding_inds.shape
        Rx_signal = commun_sim(encoding_inds, snr=snr, quan_bits=self.quan_bits)

        encoding_inds = torch.from_numpy(Rx_signal).cuda().reshape(shape)
  
        
        # Convert to one-hot encodings
        device = latents.device
        
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BL x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BL, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x L x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()
        #print((quantized_latents - latents))
        return quantized_latents.contiguous(), vq_loss  # [B x L x D]
    




class Channels():
    def AWGN(self, Tx_sig, n_std):
        device = Tx_sig.device
        noise = torch.normal(0, n_std/math.sqrt(2), size=Tx_sig.shape).to(device)
        Rx_sig = Tx_sig + noise
        return Rx_sig

    def Rayleigh(self, Tx_sig, n_std):
        device = Tx_sig.device
        shape = Tx_sig.shape
        H_real = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
        H_imag = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, n_std)
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)
        return Rx_sig

    def Rician(self, Tx_sig, n_std, K=1):
        device = Tx_sig.device
        shape = Tx_sig.shape
        mean = math.sqrt(K / (K + 1))
        std = math.sqrt(1 / (K + 1))
        H_real = torch.normal(mean, std, size=[1]).to(device)
        H_imag = torch.normal(mean, std, size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, n_std)
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)

        return Rx_sig

