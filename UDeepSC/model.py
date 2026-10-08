import torch
import torch.nn as nn
import numpy as np
import os

from functools import partial
from trans_deocer import Decoder
from timm.models.registry import register_model
from transformers import BertModel

from base_args import (
    IMGC_NUMCLASS,
    IMGR_LENGTH,
    MSA_NUMCLASS,
    TEXTC_NUMCLASS,
    TEXTR_NUMCLASS,
    VQA_NUMCLASS,
)
from channel import power_norm_batchwise
from model_util import _cfg, Channels, SPTEncoder, ViTEncoder, noise_gen


BERT_ROOT = os.environ.get(
    "BERT_ROOT", "/8T2/zhangguangyi/UDeepSC_renew/pretrain_models"
)
VIT_BASE_PATCH32_224_URL = (
    "https://storage.googleapis.com/vit_models/augreg/"
    "B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0"
    "--imagenet2012-steps_20k-lr_0.03-res_224.npz"
)
TEXT_EMBED_DIMS = {
    "tiny": 128,
    "small": 512,
    "medium": 512,
}
TASK_TOKEN_LENGTHS = {
    "imgc": 25,
    "imgr": 64,
    "textc": 25,
    "vqa": 25,
    "msa": 2,
    "textr": 66,
}
TASK_HEAD_DIMS = {
    "imgc": IMGC_NUMCLASS,
    "textc": TEXTC_NUMCLASS,
    "textr": TEXTR_NUMCLASS,
    "vqa": VQA_NUMCLASS,
    "imgr": IMGR_LENGTH,
    "msa": MSA_NUMCLASS,
}
TEXT_TASKS = ("textc", "textr", "vqa", "msa")

__all__ = ["UDeepSC_model", "UDeepSC_new_model"]


def _bert_checkpoint(mode):
    return f"{BERT_ROOT}/bert-{mode}"


def _text_embed_dim(mode):
    return TEXT_EMBED_DIMS.get(mode, 512)


def _task_embeddings(decoder_embed_dim):
    return nn.ModuleDict({
        task: nn.Embedding(length, decoder_embed_dim)
        for task, length in TASK_TOKEN_LENGTHS.items()
    })


def _task_heads(decoder_embed_dim):
    return nn.ModuleDict({
        task: nn.Linear(decoder_embed_dim, output_dim)
        for task, output_dim in TASK_HEAD_DIMS.items()
    })


def _n2p(w, transpose=True):
    if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
        w = w.flatten()
    if transpose:
        if w.ndim == 4:
            w = w.transpose([3, 2, 0, 1])
        elif w.ndim == 3:
            w = w.transpose([2, 0, 1])
        elif w.ndim == 2:
            w = w.transpose([1, 0])
    return torch.from_numpy(w)


@torch.no_grad()
def _load_vit_base_patch32_img_branch(img_encoder, checkpoint_path):
    """Load Google ViT-B/32 .npz weights into the imgc branch."""
    weights = np.load(checkpoint_path)
    prefix = 'opt/target/' if 'opt/target/embedding/kernel' in weights else ''

    img_encoder.patch_embed_imgc.proj.weight.copy_(
        _n2p(weights[f'{prefix}embedding/kernel']))
    img_encoder.patch_embed_imgc.proj.bias.copy_(
        _n2p(weights[f'{prefix}embedding/bias']))
    img_encoder.cls_token['imgc'].copy_(_n2p(weights[f'{prefix}cls'], False))
    img_encoder.pos_embed_imgc.copy_(_n2p(
        weights[f'{prefix}Transformer/posembed_input/pos_embedding'], False))
    img_encoder.norm.weight.copy_(
        _n2p(weights[f'{prefix}Transformer/encoder_norm/scale']))
    img_encoder.norm.bias.copy_(
        _n2p(weights[f'{prefix}Transformer/encoder_norm/bias']))

    for i, block in enumerate(img_encoder.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(weights[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(weights[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(weights[f'{mha_prefix}{name}/kernel'], False).flatten(1).T
            for name in ('query', 'key', 'value')
        ]))
        if block.attn.q_bias is not None:
            block.attn.q_bias.copy_(
                _n2p(weights[f'{mha_prefix}query/bias'], False).reshape(-1))
        if block.attn.v_bias is not None:
            block.attn.v_bias.copy_(
                _n2p(weights[f'{mha_prefix}value/bias'], False).reshape(-1))
        block.attn.proj.weight.copy_(
            _n2p(weights[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(weights[f'{mha_prefix}out/bias']))
        block.mlp.fc1.weight.copy_(
            _n2p(weights[f'{block_prefix}MlpBlock_3/Dense_0/kernel']))
        block.mlp.fc1.bias.copy_(
            _n2p(weights[f'{block_prefix}MlpBlock_3/Dense_0/bias']))
        block.mlp.fc2.weight.copy_(
            _n2p(weights[f'{block_prefix}MlpBlock_3/Dense_1/kernel']))
        block.mlp.fc2.bias.copy_(
            _n2p(weights[f'{block_prefix}MlpBlock_3/Dense_1/bias']))
        block.norm2.weight.copy_(_n2p(weights[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(weights[f'{block_prefix}LayerNorm_2/bias']))


class _UDeepSCBase(nn.Module):
    def _init_text_task_embeddings(self):
        hidden_size = self.text_encoder.config.hidden_size
        self.text_task_embedd = nn.ParameterDict({
            task: nn.Parameter(torch.zeros(1, 1, hidden_size))
            for task in TEXT_TASKS
        })
        for task_embedd in self.text_task_embedd.values():
            nn.init.trunc_normal_(task_embedd, std=.02)

    def _encode_text(self, text, ta_perform):
        """Run BERT with a learned task token appended to its input sequence."""
        if ta_perform not in self.text_task_embedd:
            raise ValueError(f"Unsupported text task: {ta_perform}")

        inputs_embeds = self.text_encoder.embeddings.word_embeddings(text)
        batch_size = inputs_embeds.shape[0]
        task_embedd = self.text_task_embedd[ta_perform].expand(
            batch_size, -1, -1).to(
                device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        inputs_embeds = torch.cat((inputs_embeds, task_embedd), dim=1)
        return self.text_encoder(
            inputs_embeds=inputs_embeds, return_dict=False)[0]

    def _noise_std(self, train_snr, test_snr):
        device = next(self.parameters()).device
        if self.training:
            _, noise_std = noise_gen(
                self.training, train_snr=train_snr, device=device)
            return noise_std
        return 10 ** (-torch.as_tensor(
            [test_snr], dtype=torch.float32, device=device) / 20)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    def transmit(self, input_signal, noise_std, encoder_to_channel,
                 channel_to_decoder):
        x = encoder_to_channel(input_signal)
        x = power_norm_batchwise(x)
        x = self.channel.AWGN(x, noise_std.item())
        return channel_to_decoder(x)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def _select_modal_features(self, ta_perform, x_img=None, x_text=None,
                               x_spe=None):
        if ta_perform.startswith('img'):
            return x_img
        if ta_perform.startswith('text'):
            return x_text
        if ta_perform.startswith('vqa'):
            return torch.cat([x_img, x_text], dim=1)
        if ta_perform.startswith('msa'):
            return torch.cat([x_img, x_text, x_spe], dim=1)
        raise ValueError(f"Unsupported task: {ta_perform}")

    def _decode(self, x, ta_perform):
        if ta_perform.endswith('r'):
            x = self.decoder(x, x, None, None, None)
            return self.head[ta_perform](x)

        batch_size = x.shape[0]
        query_embed = self.task_dict[ta_perform].weight.unsqueeze(0).repeat(
            batch_size, 1, 1)
        x = self.decoder(query_embed, x, None, None, None)
        x = self.head[ta_perform](x.mean(1))
        if ta_perform.startswith('vqa'):
            x = self.sigmoid_layer(x)
        return x


class UDeepSC_M1(_UDeepSCBase):
    def __init__(
        self,
        mode='tiny',
        img_size=224,
        patch_size=16,
        encoder_in_chans=3,
        encoder_num_classes=0,
        img_embed_dim=384,
        text_embed_dim=384,
        speech_embed_dim=128,
        img_encoder_depth=4,
        text_encoder_depth=4,
        speech_encoder_depth=4,
        encoder_num_heads=12,
        decoder_num_classes=768,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=8,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer=nn.LayerNorm,
        init_values=0.,
        use_learnable_pos_emb=False,
        num_classes=0,
    ):

        super().__init__()
        self.img_encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes,
            embed_dim=img_embed_dim,
            depth=img_encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            use_learnable_pos_emb=use_learnable_pos_emb)

        self.text_encoder = BertModel.from_pretrained(_bert_checkpoint(mode))
        self._init_text_task_embeddings()

        self.spe_encoder = SPTEncoder(
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes,
            embed_dim=speech_embed_dim,
            depth=speech_encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            use_learnable_pos_emb=use_learnable_pos_emb)

        text_embed_dim = _text_embed_dim(mode)

        self.num_symbols_img = 16
        self.num_symbols_text = 6
        self.num_symbols_spe = 16

        self.text_encoder_to_channel = nn.Linear(
            text_embed_dim, self.num_symbols_text)
        self.img_encoder_to_channel = nn.Linear(
            img_embed_dim, self.num_symbols_img)
        self.spe_encoder_to_channel = nn.Linear(
            speech_embed_dim, self.num_symbols_spe)

        self.text_channel_to_decoder = nn.Linear(
            self.num_symbols_text, decoder_embed_dim)
        self.img_channel_to_decoder = nn.Linear(
            self.num_symbols_img, decoder_embed_dim)
        self.spe_channel_to_decoder = nn.Linear(
            self.num_symbols_spe, decoder_embed_dim)

        self.task_dict = _task_embeddings(decoder_embed_dim)
        self.head = _task_heads(decoder_embed_dim)

        self.decoder = Decoder(
            depth=decoder_depth,
            embed_dim=decoder_embed_dim,
            num_heads=decoder_num_heads,
            dff=mlp_ratio * decoder_embed_dim,
            drop_rate=drop_rate)
        self.channel = Channels()
        self.sigmoid_layer = nn.Sigmoid()

    def forward(
            self,
            text=None,
            img=None,
            speech=None,
            ta_perform=None,
            train_snr=12.0,
            test_snr=12.0):
        noise_std = self._noise_std(train_snr, test_snr)
        x_img = x_text = x_spe = None
        if text is not None:
            x_text = self._encode_text(text, ta_perform)
            x_text = self.text_encoder_to_channel(x_text)

            if ta_perform.startswith('textc'):
                x_text = x_text[:, 0, :].unsqueeze(1)
            elif ta_perform.startswith('textr'):
                # Drop the original [CLS]/last padded position and the new
                # task token, preserving the pre-patch reconstruction length.
                x_text = x_text[:, 1:-2, :]
            elif ta_perform.startswith('vqa'):
                x_text = x_text[:, 0:2, :]
            elif ta_perform.startswith('msa'):
                x_text = x_text[:, 0].unsqueeze(1)

            x_text = power_norm_batchwise(x_text)
            x_text = self.channel.AWGN(x_text, noise_std.item())
            x_text = self.text_channel_to_decoder(x_text)
        if img is not None:
            x_img = self.img_encoder(img, ta_perform)
            x_img = self.img_encoder_to_channel(x_img)
            if ta_perform.startswith('imgc'):
                x_img = x_img[:, 0, :].unsqueeze(1)
            elif ta_perform.startswith('imgr'):
                x_img = x_img[:, 1:-1, :]
            elif ta_perform.startswith('vqa'):
                x_img = x_img[:, 0:3, :]
            elif ta_perform.startswith('msa'):
                x_img = x_img[:, 0, :].unsqueeze(1)
            x_img = power_norm_batchwise(x_img)
            x_img = self.channel.AWGN(x_img, noise_std.item())
            x_img = self.img_channel_to_decoder(x_img)

        if speech is not None:
            x_spe = self.spe_encoder(speech, ta_perform)
            x_spe = self.spe_encoder_to_channel(x_spe)
            x_spe = x_spe[:, 0, :].unsqueeze(1)
            x_spe = power_norm_batchwise(x_spe)
            x_spe = self.channel.AWGN(x_spe, noise_std.item())
            x_spe = self.spe_channel_to_decoder(x_spe)

        x = self._select_modal_features(
            ta_perform, x_img=x_img, x_text=x_text, x_spe=x_spe)
        return self._decode(x, ta_perform)


class UDeepSC_M2(_UDeepSCBase):
    def __init__(
        self,
        mode='tiny',
        img_size=224,
        patch_size=16,
        encoder_in_chans=3,
        encoder_num_classes=0,
        img_embed_dim=384,
        text_embed_dim=384,
        speech_embed_dim=128,
        img_encoder_depth=4,
        text_encoder_depth=4,
        speech_encoder_depth=4,
        encoder_num_heads=12,
        decoder_num_classes=768,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=8,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer=nn.LayerNorm,
        init_values=0.,
        use_learnable_pos_emb=False,
        num_classes=0,
    ):

        super().__init__()
        self.img_encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes,
            embed_dim=img_embed_dim,
            depth=img_encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            use_learnable_pos_emb=use_learnable_pos_emb)

        self.text_encoder = BertModel.from_pretrained(_bert_checkpoint(mode))
        self._init_text_task_embeddings()

        self.spe_encoder = SPTEncoder(
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes,
            embed_dim=speech_embed_dim,
            depth=speech_encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            use_learnable_pos_emb=use_learnable_pos_emb)

        text_embed_dim = _text_embed_dim(mode)

        self.num_symbols_imgc = 16
        self.num_symbols_imgr = 16
        self.num_symbols_textc = 4
        self.num_symbols_textr = 24
        self.num_symbols_vqa_img = 16
        self.num_symbols_vqa_text = 6
        self.num_symbols_msa_img = 16
        self.num_symbols_msa_text = 6
        self.num_symbols_msa_spe = 16

        self.textc_encoder_to_channel = nn.Linear(
            text_embed_dim, self.num_symbols_textc)
        self.imgc_encoder_to_channel = nn.Linear(
            img_embed_dim, self.num_symbols_imgc)
        self.textr_encoder_to_channel = nn.Linear(
            text_embed_dim, self.num_symbols_textr)
        self.imgr_encoder_to_channel = nn.Linear(
            img_embed_dim, self.num_symbols_imgr)
        self.vqa_img_encoder_to_channel = nn.Linear(
            img_embed_dim, self.num_symbols_vqa_img)
        self.vqa_text_encoder_to_channel = nn.Linear(
            text_embed_dim, self.num_symbols_vqa_text)
        self.msa_img_encoder_to_channel = nn.Linear(
            img_embed_dim, self.num_symbols_msa_img)
        self.msa_text_encoder_to_channel = nn.Linear(
            text_embed_dim, self.num_symbols_msa_text)
        self.msa_spe_encoder_to_channel = nn.Linear(
            speech_embed_dim, self.num_symbols_msa_spe)

        self.textc_channel_to_decoder = nn.Linear(
            self.num_symbols_textc, decoder_embed_dim)
        self.imgc_channel_to_decoder = nn.Linear(
            self.num_symbols_imgc, decoder_embed_dim)
        self.textr_channel_to_decoder = nn.Linear(
            self.num_symbols_textr, decoder_embed_dim)
        self.imgr_channel_to_decoder = nn.Linear(
            self.num_symbols_imgr, decoder_embed_dim)
        self.vqa_img_channel_to_decoder = nn.Linear(
            self.num_symbols_vqa_img, decoder_embed_dim)
        self.vqa_text_channel_to_decoder = nn.Linear(
            self.num_symbols_vqa_text, decoder_embed_dim)
        self.msa_img_channel_to_decoder = nn.Linear(
            self.num_symbols_msa_img, decoder_embed_dim)
        self.msa_text_channel_to_decoder = nn.Linear(
            self.num_symbols_msa_text, decoder_embed_dim)
        self.msa_spe_channel_to_decoder = nn.Linear(
            self.num_symbols_msa_spe, decoder_embed_dim)

        self.task_dict = _task_embeddings(decoder_embed_dim)
        self.head = _task_heads(decoder_embed_dim)

        self.decoder = Decoder(
            depth=decoder_depth,
            embed_dim=decoder_embed_dim,
            num_heads=decoder_num_heads,
            dff=mlp_ratio * decoder_embed_dim,
            drop_rate=drop_rate)
        self.channel = Channels()
        self.sigmoid_layer = nn.Sigmoid()

        # self.LN = nn.LayerNorm(text_embed_dim)

    def forward(
            self,
            text=None,
            img=None,
            speech=None,
            ta_perform=None,
            train_snr=12.0,
            test_snr=12.0):
        noise_std = self._noise_std(train_snr, test_snr)
        x_img = x_text = x_spe = None
        if text is not None:
            x_text = self._encode_text(text, ta_perform)
            # x_text = self.LN(x_text)
            if ta_perform.startswith('textc'):
                x_text = x_text[:, 0, :].unsqueeze(1)
                x_text = self.transmit(
                    x_text,
                    noise_std,
                    self.textc_encoder_to_channel,
                    self.textc_channel_to_decoder)
            elif ta_perform.startswith('textr'):
                # Drop the original [CLS]/last padded position and the new
                # task token, preserving the pre-patch reconstruction length.
                x_text = x_text[:, 1:-2, :]
                x_text = self.transmit(
                    x_text,
                    noise_std,
                    self.textr_encoder_to_channel,
                    self.textr_channel_to_decoder)
            elif ta_perform.startswith('vqa'):
                x_text = x_text[:, 0:2, :]
                x_text = self.transmit(
                    x_text,
                    noise_std,
                    self.vqa_text_encoder_to_channel,
                    self.vqa_text_channel_to_decoder)
            elif ta_perform.startswith('msa'):
                x_text = x_text[:,-2:-1,:]
                x_text = self.transmit(
                    x_text,
                    noise_std,
                    self.msa_text_encoder_to_channel,
                    self.msa_text_channel_to_decoder)

        if img is not None:
            x_img = self.img_encoder(img, ta_perform)
            if ta_perform.startswith('imgc'):
                x_img = x_img[:, 0, :].unsqueeze(1)
                x_img = self.transmit(
                    x_img,
                    noise_std,
                    self.imgc_encoder_to_channel,
                    self.imgc_channel_to_decoder)

            elif ta_perform.startswith('imgr'):
                x_img = x_img[:, 1:-1, :]
                x_img = self.transmit(
                    x_img,
                    noise_std,
                    self.imgr_encoder_to_channel,
                    self.imgr_channel_to_decoder)

            elif ta_perform.startswith('vqa'):
                x_img = x_img[:, 0:3, :]
                x_img = self.transmit(
                    x_img,
                    noise_std,
                    self.vqa_img_encoder_to_channel,
                    self.vqa_img_channel_to_decoder)

            elif ta_perform.startswith('msa'):
                x_img = x_img[:,0,:].unsqueeze(1)
                x_img = self.transmit(
                    x_img,
                    noise_std,
                    self.msa_img_encoder_to_channel,
                    self.msa_img_channel_to_decoder)

        if speech is not None:
            x_spe = self.spe_encoder(speech, ta_perform)
            x_spe = x_spe[:,0,:].unsqueeze(1)
            x_spe = self.transmit(
                x_spe,
                noise_std,
                self.msa_spe_encoder_to_channel,
                self.msa_spe_channel_to_decoder)

        x = self._select_modal_features(
            ta_perform, x_img=x_img, x_text=x_text, x_spe=x_spe)
        return self._decode(x, ta_perform)


@register_model
def UDeepSC_model(pretrained=False, **kwargs):
    init_ckpt = kwargs.pop("init_ckpt", None)
    model = UDeepSC_M1(
        mode='small',
        img_size=32,
        patch_size=4,
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
            init_ckpt, map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def UDeepSC_new_model(pretrained=False, **kwargs):
    init_ckpt = kwargs.pop("init_ckpt", None)
    model = UDeepSC_M2(
        mode='medium',
        img_size=224,
        patch_size=32,
        img_embed_dim=768,
        text_embed_dim=384,
        speech_embed_dim=128,
        img_encoder_depth=12,
        text_encoder_depth=4,
        speech_encoder_depth=4,
        encoder_num_heads=12,
        decoder_embed_dim=128,
        decoder_depth=6,
        decoder_num_heads=4,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_learnable_pos_emb=True,
        **kwargs)
    model.default_cfg = _cfg(url=VIT_BASE_PATCH32_224_URL)
    if pretrained:
        if init_ckpt is None:
            raise ValueError("init_ckpt is required when pretrained=True")
        if init_ckpt.endswith(".npz"):
            _load_vit_base_patch32_img_branch(model.img_encoder, init_ckpt)
        else:
            checkpoint = torch.load(init_ckpt, map_location="cpu")
            model.load_state_dict(checkpoint["model"])
    return model
