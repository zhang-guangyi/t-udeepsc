import math
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from channel import *
from model_util import *
from functools import partial
from trans_deocer import Decoder
from utils import batch_index_select
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from typing import List, Callable, Union, Any, TypeVar, Tuple
from transformers import BertForSequenceClassification, BertModel
from model_util import Block, _cfg, PatchEmbed, get_sinusoid_encoding_table
from base_args import IMGC_NUMCLASS,TEXTC_NUMCLASS,IMGR_LENGTH,TEXTR_NUMCLASS,VQA_NUMCLASS,MSA_NUMCLASS

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

__all__ = [
    'UDeepSC_model']


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
        self.img_encoder = ViTEncoder_FSM(img_size=img_size, patch_size=patch_size, in_chans=encoder_in_chans, 
                                num_classes=encoder_num_classes, embed_dim=img_embed_dim,depth=img_encoder_depth,
                                num_heads=encoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,drop_rate=drop_rate, 
                                drop_path_rate=drop_path_rate,norm_layer=norm_layer, init_values=init_values,
                                use_learnable_pos_emb=use_learnable_pos_emb)
        
        bert_ckpt = f"/Data1/zhangguangyi/SemanRes2/JSACCode/UDeepSC_Base/pretrained_models/bert-{mode}"
        
        
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
        
        self.text_encoder = TextEncoder_FSM(in_chans=encoder_in_chans, 
                                num_classes=encoder_num_classes, embed_dim=text_embed_dim,depth=text_encoder_depth,
                                num_heads=2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,drop_rate=drop_rate, 
                                drop_path_rate=drop_path_rate,norm_layer=norm_layer, init_values=init_values,
                                use_learnable_pos_emb=use_learnable_pos_emb, mode=mode)
        
        
        # self.num_symbols_img = 16
        # self.num_symbols_text = 8
        # self.num_symbols_spe = 16

        # self.text_encoder_to_channel = nn.Linear(text_embed_dim, self.num_symbols_text)
        # self.img_encoder_to_channel = nn.Linear(img_embed_dim, self.num_symbols_img)
        # self.spe_encoder_to_channel = nn.Linear(speech_embed_dim, self.num_symbols_spe)

        self.text_channel_to_decoder = nn.Linear(text_embed_dim, decoder_embed_dim)
        self.img_channel_to_decoder = nn.Linear(img_embed_dim, decoder_embed_dim)
        self.spe_channel_to_decoder = nn.Linear(speech_embed_dim, decoder_embed_dim)


        self.task_dict = nn.ModuleDict()
        self.task_dict['imgc'] = nn.Embedding(25, decoder_embed_dim)
        self.task_dict['imgr'] = nn.Embedding(64, decoder_embed_dim)
        self.task_dict['textc'] = nn.Embedding(25, decoder_embed_dim)
        self.task_dict['vqa'] = nn.Embedding(25, decoder_embed_dim)
        self.task_dict['msa']  = nn.Embedding(25, decoder_embed_dim)
        self.task_dict['textr'] = nn.Embedding(66, decoder_embed_dim)


        self.head = nn.ModuleDict()
        self.head['imgc']  = nn.Linear(decoder_embed_dim, IMGC_NUMCLASS)
        self.head['textc'] = nn.Linear(decoder_embed_dim, TEXTC_NUMCLASS)
        self.head['textr'] = nn.Linear(decoder_embed_dim, TEXTR_NUMCLASS)
        self.head['vqa']   = nn.Linear(decoder_embed_dim, VQA_NUMCLASS)
        self.head['imgr']  = nn.Linear(decoder_embed_dim, IMGR_LENGTH)
        self.head['msa']   = nn.Linear(decoder_embed_dim, MSA_NUMCLASS)

        
        self.codebook = nn.ModuleDict()
        self.bit_per_digit = 4
        
        self.codebook['img'] =  VectorQuantizer(num_embeddings=2**self.bit_per_digit,
                                        embedding_dim=img_embed_dim,
                                        quan_bits=self.bit_per_digit)
        self.codebook['text'] = VectorQuantizer(num_embeddings=2**self.bit_per_digit,
                                        embedding_dim=text_embed_dim,
                                        quan_bits=self.bit_per_digit)
        self.codebook['spe'] = VectorQuantizer(num_embeddings=2**self.bit_per_digit,
                                        embedding_dim=speech_embed_dim,
                                        quan_bits=self.bit_per_digit)
        
       
        
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

    def forward(self, text=None, img=None, speech=None, ta_perform=None, test_snr=torch.FloatTensor([12])):
        if self.training:
            noise_snr, noise_std = noise_gen(self.training)
            noise_std,noise_snr = noise_std.cuda(), noise_snr.cpu().item()
        else:
            noise_std = torch.FloatTensor([1]) * 10**(-test_snr/20)
            noise_snr = test_snr
            noise_std = noise_std.cuda()
        # noise_snr, noise_std = noise_gen(self.training)
        # noise_std, noise_snr = noise_std.cuda(), noise_snr.cpu().item()
        m_dict, rho_dict, codebook_loss = {},{},{}

    
        
        codebook_loss['img'] = torch.tensor(0.)
        codebook_loss['text'] = torch.tensor(0.)
        codebook_loss['spe'] = torch.tensor(0.)

        ######  Compute the encoder and mapping pof codebook
        if ta_perform.startswith('textc'):
            # print(x_text.shape)
            x_text, m_dict['text'], rho_dict['text'] = self.text_encoder(text, ta_perform, noise_std)
            x_text, codebook_loss['text'] = self.codebook['text'](x_text, noise_snr)
            x_text = self.text_channel_to_decoder(x_text) 
        elif ta_perform.startswith('imgc'):
            x_img, m_dict['img'], rho_dict['img'] = self.img_encoder(img, ta_perform, noise_std)
            x_img, codebook_loss['img'] = self.codebook['img'](x_img, noise_snr)
            x_img = self.img_channel_to_decoder(x_img)
        elif ta_perform.startswith('vqa'):
        
            x_text, m_dict['text'], rho_dict['text'] = self.text_encoder(text, ta_perform,noise_std)
            x_text, codebook_loss['text'] = self.codebook['text'](x_text, noise_snr)
            x_text = self.text_channel_to_decoder(x_text) 
            x_img, m_dict['img'], rho_dict['img'] = self.img_encoder(img, ta_perform, noise_std)
            x_img, codebook_loss['img'] = self.codebook['img'](x_img, noise_snr)
            x_img = self.img_channel_to_decoder(x_img)
        if speech is not None:
            x_spe, m_dict['spe'], rho_dict['spe'] = self.spe_encoder(speech, ta_perform)
            x_spe = x_spe[:,0:-1,:]
            x_spe, codebook_loss['spe'] = self.codebook['spe'](x_spe, noise_snr)
            x_spe = self.spe_channel_to_decoder(x_spe)
            
 
        #######  Compute the policy vectors    
        if ta_perform.startswith('img'):
            x = x_img
            if self.training:
                cls_m = torch.ones(x.shape[0], 1, 1, dtype=x.dtype, device=x.device) 
                curr_m = m_dict['img'][-1]
                policy = torch.cat([cls_m, curr_m], dim=1)
        elif ta_perform.startswith('text'):
            x = x_text
            if self.training:
                cls_m = torch.ones(x.shape[0], 1, 1, dtype=x.dtype, device=x.device) 
                curr_m = m_dict['text'][-1]
                policy = torch.cat([cls_m, curr_m], dim=1)
        elif ta_perform.startswith('vqa'):
            x = torch.cat([x_img, x_text], dim=1)
            if self.training:
                cls_m = torch.ones(x.shape[0], 1, 1, dtype=x.dtype, device=x.device) 
                curr_m_text = m_dict['text'][-1]
                curr_m_img = m_dict['img'][-1]
                policy_text = torch.cat([cls_m, curr_m_text], dim=1)
                policy_img = torch.cat([cls_m, curr_m_img], dim=1)
                policy = torch.cat([policy_img, policy_text], dim=1)
        elif ta_perform.startswith('msa'):
            x = torch.cat([x_img,x_text,x_spe], dim=1)
            if self.training:
                cls_m = torch.ones(x.shape[0], 1, 1, dtype=x.dtype, device=x.device) 
                curr_m_text = m_dict['text'][-1]
                curr_m_img = m_dict['img'][-1]
                curr_m_spe = m_dict['spe'][-1]
                policy_text = torch.cat([cls_m, curr_m_text], dim=1)
                policy_img = torch.cat([cls_m, curr_m_img], dim=1)
                policy_spe = torch.cat([cls_m, curr_m_spe], dim=1)
                policy = torch.cat([policy_img, policy_text, policy_spe], dim=1)
     
        #######  Decoding process

        query_embed = self.task_dict[ta_perform].weight.unsqueeze(0).repeat(x.shape[0], 1, 1)
        x = self.decoder(query_embed, x, policy, None, None) if self.training else self.decoder(query_embed, x, None, None, None)
        if ta_perform.startswith('textr'): 
            x = self.head[ta_perform](x)
        else:
            x = self.head[ta_perform](x.mean(1))
        if ta_perform.startswith('vqa'):
            x = self.sigmoid_layer(x)
        return (x, m_dict, rho_dict, codebook_loss) if self.training else {'outputs': x, 'rho': rho_dict, 'mask': m_dict}




@register_model
def UDeepSC_model(pretrained=False, **kwargs):
    model = UDeepSC(
        mode='small',
        img_size=224,
        patch_size=32,
        img_embed_dim=384,
        text_embed_dim=384,
        speech_embed_dim=128,
        img_encoder_depth=6,
        text_encoder_depth=4,
        speech_encoder_depth=4,
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
