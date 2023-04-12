import math
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from channel import *
from model_util import *
from functools import partial
from utils import batch_index_select
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from typing import List, Callable, Union, Any, TypeVar, Tuple
from transformers import BertForSequenceClassification, BertModel
from model_util import Block, _cfg, PatchEmbed, get_sinusoid_encoding_table
from base_args import IMGC_NUMCLASS,TEXTC_NUMCLASS,IMGR_LENGTH,TEXTR_NUMCLASS,VQA_NUMCLASS,MSA_NUMCLASS


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

__all__ = [
    'TDeepSC_imgc_model' ,
    'TDeepSC_textc_model']


class UDeepSC(nn.Module):
    def __init__(self,mode='tiny',
                 img_size=224, patch_size=16, encoder_in_chans=3, encoder_num_classes=0, 
                 img_embed_dim=384, text_embed_dim=384, speech_embed_dim=128, img_encoder_depth=4, 
                 text_encoder_depth=4, speech_encoder_depth=4, encoder_num_heads=12, decoder_num_classes=768, 
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=8, mlp_ratio=4., 
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, init_values=0.,use_learnable_pos_emb=False,num_classes=0, 
                 ):

        super().__init__()
        self.img_encoder = ViTEncoder(img_size=img_size, patch_size=patch_size, in_chans=encoder_in_chans, 
                                num_classes=encoder_num_classes, embed_dim=img_embed_dim,depth=img_encoder_depth,
                                num_heads=encoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,drop_rate=drop_rate, 
                                drop_path_rate=drop_path_rate,norm_layer=norm_layer, init_values=init_values,
                                use_learnable_pos_emb=use_learnable_pos_emb)
        
        bert_ckpt = f"/Data1/zhangguangyi/SemanRes2/JSACCode/UDeepSC_Base/pretrained_models/bert-{mode}"
        self.text_encoder = BertModel.from_pretrained(bert_ckpt)
        
        self.spe_encoder = SPTEncoder(in_chans=encoder_in_chans,num_classes=encoder_num_classes, embed_dim=speech_embed_dim,
                                depth=speech_encoder_depth,num_heads=encoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,drop_rate=drop_rate, 
                                drop_path_rate=drop_path_rate,norm_layer=norm_layer, init_values=init_values,
                                use_learnable_pos_emb=use_learnable_pos_emb)
        
        if mode=='tiny':
            text_embed_dim = 128
        elif mode=='small':
            text_embed_dim = 512
        else:
            text_embed_dim = 512
            
        self.text_en2c = nn.Linear(text_embed_dim, 32)
        self.img_en2c = nn.Linear(img_embed_dim, 32)
        self.spe_en2c = nn.Linear(speech_embed_dim, 32)

        self.text_c2de = nn.Linear(32, decoder_embed_dim)
        self.img_c2de = nn.Linear(32, decoder_embed_dim)
        self.spe_c2de = nn.Linear(32, decoder_embed_dim)


        self.task_dict = nn.ModuleDict()
        self.task_dict['imgc'] = nn.Embedding(25, decoder_embed_dim)
        self.task_dict['textc'] = nn.Embedding(25, decoder_embed_dim)
        self.task_dict['vqa'] = nn.Embedding(25, decoder_embed_dim)
        self.task_dict['msa']  = nn.Embedding(25, decoder_embed_dim)
        self.task_dict['textr'] = nn.Embedding(66, decoder_embed_dim)


        self.head = nn.ModuleDict()
        self.head['imgc'] = nn.Linear(decoder_embed_dim, IMGC_NUMCLASS)
        self.head['textc'] = nn.Linear(decoder_embed_dim, TEXTC_NUMCLASS)
        self.head['textr'] = nn.Linear(decoder_embed_dim, TEXTR_NUMCLASS)
        self.head['vqa'] = nn.Linear(decoder_embed_dim, VQA_NUMCLASS)
        self.head['imgr'] = nn.Linear(decoder_embed_dim, IMGR_LENGTH)
        self.head['msa'] = nn.Linear(decoder_embed_dim, MSA_NUMCLASS)


        self.decoder = Decoder(depth=decoder_depth,embed_dim=decoder_embed_dim, 
                                num_heads=decoder_num_heads, dff=mlp_ratio*decoder_embed_dim, 
                                drop_rate=drop_rate)
        self.channel = Channels()
        self.sigmoid_layer = nn.Sigmoid()

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
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, text=None, img=None, speech=None, ta_perform=None):
        noise_snr, noise_var = noise_gen(self.training)
        noise_var,noise_snr = noise_var.cuda(), noise_snr.cpu().item()
        if text is not None:
            x_text = self.text_encoder(ta_perform, text, return_dict=False)[0]
            x_text = self.text_en2c(x_text)
            x_text = power_norm_batchwise(x_text)
            x_text = self.channel.AWGN(x_text, noise_var.item())
            x_text = self.text_c2de(x_text)
        if img is not None:
            x_img = self.img_encoder(img, ta_perform)
            x_img = self.img_en2c(x_img)
            x_img = power_norm_batchwise(x_img)
            x_img = self.channel.AWGN(x_img, noise_var.item())
            x_img = self.img_c2de(x_img)
        if speech is not None:
            x_spe = self.spe_encoder(speech, ta_perform)
            x_spe = self.spe_en2c(x_spe)
            x_spe = power_norm_batchwise(x_spe)
            x_spe = self.channel.AWGN(x_spe, noise_var.item())
            x_spe = self.spe_c2de(x_spe)
        
        if ta_perform.startswith('imgc'):
            x = x_img[:,0,:].unsqueeze(1)
        elif ta_perform.startswith('imgr'):
            x = x_img[:,1:-1,:]
        elif ta_perform.startswith('textc'):
            x = x_text[:,0,:].unsqueeze(1)
        elif ta_perform.startswith('textr'):
            x = x_text[:,1:-1,:]  
        elif ta_perform.startswith('vqa'):
            x = torch.cat([x_img[:,0:,:],x_text[:,0].unsqueeze(1),], dim=1)

        elif ta_perform.startswith('msa'):
            x = torch.cat([x_img[:,0,:].unsqueeze(1),x_text[:,0].unsqueeze(1),x_spe[:,0,:].unsqueeze(1)], dim=1)

        batch_size = x.shape[0]
        if ta_perform.endswith('r'):
            x = self.decoder(x, x, None, None) 
            x = self.head[ta_perform](x)
            return x
        else:
            query_embed = self.task_dict[ta_perform].weight.unsqueeze(0).repeat(batch_size, 1, 1)
            x = self.decoder(query_embed, x, None, None) 
            if ta_perform.startswith('textr'): 
                x = self.head[ta_perform](x)
            else:
                x = self.head[ta_perform](x.mean(1))
            if ta_perform.startswith('vqa'):
                x = self.sigmoid_layer(x)
            return x



@register_model
def UDeepSC_model(pretrained=False, **kwargs):
    model = UDeepSC(
        mode='medium',
        img_size=32,
        patch_size=4,
        img_embed_dim=384,
        text_embed_dim=384,
        speech_embed_dim=128,

        img_encoder_depth=4,
        text_encoder_depth=4,
        speech_encoder_depth=4,
        encoder_num_heads=6,
        decoder_embed_dim=384,
        decoder_depth=6,
        decoder_num_heads=4,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

