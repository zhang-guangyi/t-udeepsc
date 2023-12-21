import math
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


from channel import *
from model_util import *
from functools import partial
from trans_decoder import Decoder
from transformers import  BertModel
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from typing import List, Callable, Union, Any, TypeVar, Tuple
from model_util import Block, _cfg, PatchEmbed, get_sinusoid_encoding_table
from model_util import PositionalEncoding, ViTEncoder_imgcr, SPTEncoder,ViTEncoder_vqa,ViTEncoder_msa
from base_args import IMGC_NUMCLASS,TEXTC_NUMCLASS,IMGR_LENGTH,TEXTR_NUMCLASS,VQA_NUMCLASS,MSA_NUMCLASS


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

__all__ = [
    'TDeepSC_imgc_model' ,
    'TDeepSC_textc_model']

class TDeepSC_imgc(nn.Module):
    def __init__(self,
                 img_size=224, patch_size=16, encoder_in_chans=3, encoder_num_classes=0, 
                 encoder_embed_dim=768, encoder_depth=12,encoder_num_heads=12, decoder_num_classes=768, 
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=8, mlp_ratio=4., 
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, init_values=0.,use_learnable_pos_emb=False,num_classes=0, 
                 ):
        super().__init__()
        self.img_encoder = ViTEncoder_imgcr(img_size=img_size, patch_size=patch_size, in_chans=encoder_in_chans, 
                                num_classes=encoder_num_classes, embed_dim=encoder_embed_dim,depth=encoder_depth,
                                num_heads=encoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,drop_rate=drop_rate, 
                                drop_path_rate=drop_path_rate,norm_layer=norm_layer, init_values=init_values,
                                use_learnable_pos_emb=use_learnable_pos_emb)

        self.num_symbol = 16
        
        self.encoder_to_channel = nn.Linear(encoder_embed_dim, self.num_symbol)
        self.channel = Channels()
        self.channel_to_decoder = nn.Linear(self.num_symbol, decoder_embed_dim)
        
        self.head = nn.Linear(decoder_embed_dim, IMGC_NUMCLASS)
        self.decoder = Decoder(depth=decoder_depth,embed_dim=decoder_embed_dim, 
                                   num_heads=decoder_num_heads, dff=mlp_ratio*decoder_embed_dim, drop_rate=drop_rate)

        self.query_embedd = nn.Embedding(25, decoder_embed_dim)

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

    def forward(self, text=None, img=None, ta_perform=None,
                      test_snr=torch.FloatTensor([-2])):

        if self.training:
            noise_snr, noise_std = noise_gen(self.training)
            noise_std, noise_snr = noise_std.cuda(), noise_snr.cpu().item()
        else:
            noise_std = torch.FloatTensor([1]) * 10**(-test_snr/20)  
        x = self.img_encoder(img, ta_perform)
        batch_size = x.shape[0]
        x = self.encoder_to_channel(x[:,0])
        x = power_norm_batchwise(x.unsqueeze(1))
        x = self.channel.Rayleigh(x, noise_std.item())
        x = self.channel_to_decoder(x)

        query_embed = self.query_embedd.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        x = self.decoder(query_embed, x, None, None, None) 
        
        x = self.head(x.mean(1))
        return x


class TDeepSC_imgr(nn.Module):
    def __init__(self,
                 img_size=224, patch_size=16, encoder_in_chans=3, encoder_num_classes=0, 
                 encoder_embed_dim=768, encoder_depth=12,encoder_num_heads=12, decoder_num_classes=768, 
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=8, mlp_ratio=4., 
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, init_values=0.,use_learnable_pos_emb=False,num_classes=0, 
                 ):
        super().__init__()
        self.img_encoder = ViTEncoder_imgcr(img_size=img_size, patch_size=patch_size, in_chans=encoder_in_chans, 
                                num_classes=encoder_num_classes, embed_dim=encoder_embed_dim,depth=encoder_depth,
                                num_heads=encoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,drop_rate=drop_rate, 
                                drop_path_rate=drop_path_rate,norm_layer=norm_layer, init_values=init_values,
                                use_learnable_pos_emb=use_learnable_pos_emb)
        self.head = nn.Linear(decoder_embed_dim, IMGR_LENGTH)

        self.num_symbol = 16

        self.encoder_to_channel = nn.Linear(encoder_embed_dim, self.num_symbol)
        self.channel_to_decoder = nn.Linear(self.num_symbol, decoder_embed_dim)

        self.decoder = Decoder(depth=decoder_depth,embed_dim=decoder_embed_dim, 
                                   num_heads=decoder_num_heads, dff=mlp_ratio*decoder_embed_dim, drop_rate=drop_rate)
        self.query_embedd = nn.Embedding(64, decoder_embed_dim)
        self.channel = Channels()
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

    def forward(self, text=None, img=None, ta_perform=None, test_snr=torch.FloatTensor([-2])):
        if self.training:
            noise_snr, noise_std = noise_gen(self.training)
            noise_std, noise_snr = noise_std.cuda(), noise_snr.cpu().item()
        else:
            noise_std = torch.FloatTensor([1]) * 10**(-test_snr/20) 
        x = self.img_encoder(img, ta_perform)[:,1:]
        batch_size = x.shape[0]
        x = self.encoder_to_channel(x)
        x = power_norm_batchwise(x)
        x = self.channel.Rayleigh(x, noise_std.item())
        x = self.channel_to_decoder(x)

        # query_embed = self.query_embedd.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        x = self.decoder(x, x, None, None, None) 
        x = self.head(x[:,0:])
        return x
    
class TDeepSC_vqa(nn.Module):
    def __init__(self,mode='tiny',
                 img_size=224, patch_size=16, encoder_in_chans=3, encoder_num_classes=0, 
                 encoder_embed_dim=768, encoder_depth=12,encoder_num_heads=12, decoder_num_classes=768, 
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=8, mlp_ratio=4., 
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, init_values=0.,use_learnable_pos_emb=False,num_classes=0, 
                 ):
        super().__init__()
        self.img_encoder = ViTEncoder_vqa(img_size=img_size, patch_size=patch_size, in_chans=encoder_in_chans, 
                                num_classes=encoder_num_classes, embed_dim=encoder_embed_dim,depth=encoder_depth,
                                num_heads=encoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,drop_rate=drop_rate, 
                                drop_path_rate=drop_path_rate,norm_layer=norm_layer, init_values=init_values,
                                use_learnable_pos_emb=use_learnable_pos_emb)
        
        bert_ckpt = f"/Data1/zhangguangyi/SemanRes2/JSACCode/TDeepSC_Base/pretrain_models/bert-{mode}/"
        self.text_encoder = BertModel.from_pretrained(bert_ckpt)
        if mode=='tiny':
            encoder_dim_text = 128
        elif mode=='small':
            encoder_dim_text = 512
        else:
            encoder_dim_text = 512

        self.num_symbols_img = 16  # Keep all feature vectors
        self.num_symbols_text = 6   # Keep all feature vectors
        
        self.img_encoder_to_channel = nn.Linear(encoder_embed_dim, self.num_symbols_img)
        self.text_encoder_to_channel = nn.Linear(encoder_dim_text, self.num_symbols_text)
        self.img_channel_to_decoder =  nn.Linear(self.num_symbols_img, decoder_embed_dim)
        self.text_channel_to_decoder = nn.Linear(self.num_symbols_text, decoder_embed_dim)

        self.decoder = Decoder(depth=decoder_depth,embed_dim=decoder_embed_dim, 
                                   num_heads=decoder_num_heads, dff=mlp_ratio*decoder_embed_dim, drop_rate=drop_rate)
        self.query_embedd = nn.Embedding(25, decoder_embed_dim)

        

        self.channel = Channels()
        self.head = nn.Linear(decoder_embed_dim, VQA_NUMCLASS)

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

    def forward(self, text=None, img=None, ta_perform=None, test_snr=torch.FloatTensor([12])):
        if self.training:
            noise_snr, noise_std = noise_gen(self.training)
            noise_std,noise_snr = noise_std.cuda(), noise_snr.cpu().item()
        else:
            noise_std = torch.FloatTensor([1]) * 10**(-test_snr/20)  
        x_img = self.img_encoder(img, ta_perform)
        batch_size = x_img.shape[0]
        x_text = self.text_encoder(input_ids=text, return_dict=False)[0]
        x_img = self.img_encoder_to_channel(x_img)
        x_text = self.text_encoder_to_channel(x_text)
        x_img = power_norm_batchwise(x_img[:,0:3])
        x_text = power_norm_batchwise(x_text[:,0:2])
        
        x_img = self.channel.Rayleigh(x_img, noise_std.item())
        x_text= self.channel.Rayleigh(x_text, noise_std.item())
        x_img = self.img_channel_to_decoder(x_img)
        x_text = self.text_channel_to_decoder(x_text)

        x = torch.cat([x_img, x_text], dim=1)

        query_embed = self.query_embedd.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        x = self.decoder(query_embed, x, None, None, None) 
        x = self.head(x.mean(1))
        x = torch.sigmoid(x)
        return x


class TDeepSC_textc(nn.Module):
    def __init__(self,
                 mode='tiny',drop_rate=0., attn_drop_rate=0., drop_path_rate=0., decoder_num_classes=384, 
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=8, mlp_ratio=4., 
                 qkv_bias=False, qk_scale=None,norm_layer=nn.LayerNorm, init_values=0.,
                 ):
        super().__init__()
        bert_ckpt = f"/Data1/zhangguangyi/SemanRes2/JSACCode/TDeepSC_Base/pretrain_models/bert-{mode}/"
        self.text_encoder = BertModel.from_pretrained(bert_ckpt)
        if mode=='tiny':
            encoder_embed_dim = 128
        elif mode=='small':
            encoder_embed_dim = 512
        else:
            encoder_embed_dim = 512

        self.num_symbols = 4

        self.encoder_to_channel = nn.Linear(encoder_embed_dim, self.num_symbols)
        self.channel = Channels()
        self.channel_to_decoder = nn.Linear(self.num_symbols, decoder_embed_dim)

        self.decoder = Decoder(depth=decoder_depth,embed_dim=decoder_embed_dim, 
                                   num_heads=decoder_num_heads, dff=mlp_ratio*decoder_embed_dim, drop_rate=drop_rate)
        self.query_embedd = nn.Embedding(25, decoder_embed_dim)


        self.head = nn.Linear(decoder_embed_dim, TEXTC_NUMCLASS)

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

    def forward(self, text=None, img=None, ta_perform=None, test_snr=torch.FloatTensor([12])):
        x = self.text_encoder(ta_perform=None, input_ids= text, return_dict=False)[0]
        batch_size = x.shape[0]
        if self.training:
            noise_snr, noise_std = noise_gen(self.training)
            noise_std, noise_snr = noise_std.cuda(), noise_snr.cpu().item()
        else:
            noise_std = torch.FloatTensor([1]) * 10**(-test_snr/20)  
        x = self.encoder_to_channel(x[:,0])
        x = power_norm_batchwise(x)
        x = self.channel.Rayleigh(x, noise_std.item())
        x = self.channel_to_decoder(x)

        query_embed = self.query_embedd.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        x = self.decoder(query_embed, x, None, None, None) 

        x = self.head(x.mean(1))
        return x
    
class TDeepSC_textr(nn.Module):
    def __init__(self,
                 mode='tiny',drop_rate=0., attn_drop_rate=0., drop_path_rate=0., decoder_num_classes=384, 
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=8, mlp_ratio=4., 
                 qkv_bias=False, qk_scale=None,norm_layer=nn.LayerNorm, init_values=0.,
                 ):
        super().__init__()
        bert_ckpt = f"/Data1/zhangguangyi/SemanRes2/JSACCode/TDeepSC_Base/pretrain_models/bert-{mode}/"
        self.text_encoder = BertModel.from_pretrained(bert_ckpt)
        if mode=='tiny':
            encoder_embed_dim = 128
        elif mode=='small':
            encoder_embed_dim = 512
        else:
            encoder_embed_dim = 512

        self.num_symbols = 24
        
        self.encoder_to_channel = nn.Linear(encoder_embed_dim, self.num_symbols)
        self.channel = Channels()
        self.channel_to_decoder = nn.Linear(self.num_symbols, decoder_embed_dim)
        self.head = nn.Linear(decoder_embed_dim, TEXTR_NUMCLASS)

        self.decoder = Decoder(depth=decoder_depth,embed_dim=decoder_embed_dim, 
                                   num_heads=decoder_num_heads, dff=mlp_ratio*decoder_embed_dim, drop_rate=drop_rate)
        self.query_embedd = nn.Embedding(66, decoder_embed_dim)

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

    def forward(self, text=None, img=None, ta_perform=None, test_snr=torch.FloatTensor([12])):
        x = self.text_encoder(input_ids=text, return_dict=False)[0]
        batch_size = x.shape[0]
        if self.training:
            noise_snr, noise_std = noise_gen(self.training)
            noise_std,noise_snr = noise_std.cuda(), noise_snr.cpu().item()
        else:
            noise_std = torch.FloatTensor([1]) * 10**(-test_snr/20)  
        x = self.encoder_to_channel(x)
        x = power_norm_batchwise(x)
        x = self.channel.Rayleigh(x, noise_std.item())
        x = self.channel_to_decoder(x)

        query_embed = self.query_embedd.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        x = self.decoder(x, x, None, None, None) 

        x = self.head(x[:,0:,])
        return x

class TDeepSC_msa(nn.Module):
    def __init__(self,mode='tiny',
                 img_size=224, patch_size=16, encoder_in_chans=3, encoder_num_classes=0, 
                 encoder_embed_dim=768, encoder_depth=12,encoder_num_heads=12, decoder_num_classes=768, 
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=8, mlp_ratio=4., 
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, init_values=0.,use_learnable_pos_emb=False,num_classes=0, 
                 ):
        super().__init__()
        self.img_encoder = ViTEncoder_msa(img_size=img_size, patch_size=patch_size, in_chans=encoder_in_chans, 
                                num_classes=encoder_num_classes, embed_dim=encoder_embed_dim,depth=encoder_depth,
                                num_heads=encoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,drop_rate=drop_rate, 
                                drop_path_rate=drop_path_rate,norm_layer=norm_layer, init_values=init_values,
                                use_learnable_pos_emb=use_learnable_pos_emb)
        
        self.spe_encoder = SPTEncoder(in_chans=encoder_in_chans,num_classes=encoder_num_classes, embed_dim=128,
                                depth=encoder_depth,num_heads=encoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,drop_rate=drop_rate, 
                                drop_path_rate=drop_path_rate,norm_layer=norm_layer, init_values=init_values,
                                use_learnable_pos_emb=use_learnable_pos_emb)
        
        bert_ckpt = f"/Data1/zhangguangyi/SemanRes2/JSACCode/TDeepSC_Base/pretrain_models/bert-{mode}"
        self.text_encoder = BertModel.from_pretrained(bert_ckpt)
        if mode=='tiny':
            encoder_dim_text = 128
        elif mode=='small':
            encoder_dim_text = 512
        else:
            encoder_dim_text = 512

        self.num_symbols_img = 16
        self.num_symbols_text = 6
        self.num_symbols_spe = 16

        
        self.img_encoder_to_channel = nn.Linear(encoder_embed_dim, self.num_symbols_img)
        self.text_encoder_to_channel = nn.Linear(encoder_dim_text, self.num_symbols_text)
        self.spe_encoder_to_channel = nn.Linear(128, self.num_symbols_spe)

        self.channel = Channels()

        self.img_channel_to_decoder = nn.Linear(self.num_symbols_img, decoder_embed_dim)
        self.text_channel_to_decoder = nn.Linear(self.num_symbols_text, decoder_embed_dim)
        self.spe_channel_to_decoder = nn.Linear(self.num_symbols_spe, decoder_embed_dim)


        self.decoder = Decoder(depth=decoder_depth,embed_dim=decoder_embed_dim, 
                                   num_heads=decoder_num_heads, dff=mlp_ratio*decoder_embed_dim, drop_rate=drop_rate)
        self.query_embedd = nn.Embedding(25, decoder_embed_dim)
        self.head = nn.Linear(decoder_embed_dim, MSA_NUMCLASS)

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

    def forward(self, text=None, img=None, speech=None, ta_perform=None, test_snr=torch.FloatTensor([-2])):
        x_text = self.text_encoder(input_ids=text, return_dict=False)[0]
        x_img = self.img_encoder(img, ta_perform)
        x_spe = self.spe_encoder(speech, ta_perform)

        batch_size = x_img.shape[0]

        if self.training:
            noise_snr, noise_std = noise_gen(self.training)
            noise_std,noise_snr = noise_std.cuda(), noise_snr.cpu().item()
        else:
            noise_std = torch.FloatTensor([1]) * 10**(-test_snr/20) 


        x_img = self.img_encoder_to_channel(x_img)
        x_text = self.text_encoder_to_channel(x_text)
        x_spe = self.spe_encoder_to_channel(x_spe)

        x_img = power_norm_batchwise(x_img[:,0].unsqueeze(1))
        x_text = power_norm_batchwise(x_text[:,0].unsqueeze(1))
        x_spe = power_norm_batchwise(x_spe[:,0].unsqueeze(1))

        x_img = self.channel.Rayleigh(x_img, noise_std.item())
        x_text = self.channel.Rayleigh(x_text, noise_std.item())
        x_spe = self.channel.Rayleigh(x_spe, noise_std.item())

        x_img = self.img_channel_to_decoder(x_img)
        x_text = self.text_channel_to_decoder(x_text)
        x_spe = self.spe_channel_to_decoder(x_spe)

        x = torch.cat([x_text,x_img,x_spe], dim=1)
        query_embed = self.query_embedd.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        x = self.decoder(query_embed, x, None, None, None) 
        x = self.head(x.mean(1))
        return x


@register_model
def TDeepSC_imgc_model(pretrained=False, **kwargs):
    model = TDeepSC_imgc(
        img_size=32,
        patch_size=4,
        encoder_embed_dim=384,
        encoder_depth=4,
        encoder_num_heads=6,
        decoder_embed_dim=128,
        decoder_depth=2,
        decoder_num_heads=4,
        mlp_ratio=4,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def TDeepSC_imgr_model(pretrained=False, **kwargs):
    model = TDeepSC_imgr(
        img_size=32,
        patch_size=4,
        encoder_embed_dim=384,
        encoder_depth=4,
        encoder_num_heads=6,
        decoder_embed_dim=128,
        decoder_depth=4,
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


@register_model
def TDeepSC_textc_model(pretrained=False, **kwargs):
    model = TDeepSC_textc(
        mode='small',
        decoder_embed_dim=128,
        decoder_depth=2,
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


@register_model
def TDeepSC_textr_model(pretrained=False, **kwargs):
    model = TDeepSC_textr(
        mode='small',
        decoder_embed_dim=128,
        decoder_depth=2,
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


@register_model
def TDeepSC_vqa_model(pretrained=False, **kwargs):
    model = TDeepSC_vqa(
        mode='small',
        img_size=32,
        patch_size=4,
        encoder_embed_dim=384,
        encoder_depth=4,
        encoder_num_heads=6,
        decoder_embed_dim=128,
        decoder_depth=2,
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


@register_model
def TDeepSC_msa_model(pretrained=False, **kwargs):
    model = TDeepSC_msa(
        mode='small',
        img_size=32,
        patch_size=4,
        encoder_embed_dim=384,
        encoder_depth=4,
        encoder_num_heads=6,
        decoder_embed_dim=128,
        decoder_depth=2,
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