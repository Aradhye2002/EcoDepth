import torch.nn as nn
import torch
import random
from transformers import ViTImageProcessor, ViTForImageClassification
from timm.models.layers import trunc_normal_, DropPath
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, normal_init)
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import torch.nn.functional as F
from ecodepth.models import UNetWrapper, EmbeddingAdapter

from transformers import logging
logging.set_verbosity_error()

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class EcoDepthEncoder(nn.Module):
    def __init__(self, out_dim=1024, ldm_prior=[320, 640, 1280+1280], sd_path=None, emb_dim=768,
                 dataset='nyu', args=None):
        super().__init__()

        self.args = args

        self.layer1 = nn.Sequential(
            nn.Conv2d(ldm_prior[0], ldm_prior[0], 3, stride=2, padding=1),
            nn.GroupNorm(16, ldm_prior[0]),
            nn.ReLU(),
            nn.Conv2d(ldm_prior[0], ldm_prior[0], 3, stride=2, padding=1),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(ldm_prior[1], ldm_prior[1], 3, stride=2, padding=1),
        )

        self.out_layer = nn.Sequential(
            nn.Conv2d(sum(ldm_prior), out_dim, 1),
            nn.GroupNorm(16, out_dim),
            nn.ReLU(),
        )
        self.apply(self._init_weights)
        
        self.cide_module = CIDE(args, emb_dim)
        
        self.config = OmegaConf.load('./v1-inference.yaml')
        if sd_path is None:
            self.config.model.params.ckpt_path = '../checkpoints/v1-5-pruned-emaonly.ckpt'
        else:
            self.config.model.params.ckpt_path = f'../{sd_path}'

        sd_model = instantiate_from_config(self.config.model)
        self.encoder_vq = sd_model.first_stage_model

        self.unet = UNetWrapper(sd_model.model, use_attn=False)
        
        del sd_model.cond_stage_model
        del self.encoder_vq.decoder
        del self.unet.unet.diffusion_model.out

        for param in self.encoder_vq.parameters():
            param.requires_grad = False

        


    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, feats):
        x =  self.ldm_to_net[0](feats[0])
        for i in range(3):
            if i > 0:
                x = x + self.ldm_to_net[i](feats[i])
            x = self.layers[i](x)
            x = self.upsample_layers[i](x)
        return self.out_conv(x)

    def forward(self, x):
        # Use torch.no_grad() to prevent gradient computation on application of VQ encoder since it is frozen
        # Refer to paper for more info
        with torch.no_grad():
            # convert the input image to latent space and scale.
            latents = self.encoder_vq.encode(x).mode().detach() * self.config.model.params.scale_factor

        conditioning_scene_embedding = self.cide_module(x)

        t = torch.ones((x.shape[0],), device=x.device).long()

        outs = self.unet(latents, t, c_crossattn=[conditioning_scene_embedding])

        feats = [outs[0], outs[1], torch.cat([outs[2], F.interpolate(outs[3], scale_factor=2)], dim=1)]
        x = torch.cat([self.layer1(feats[0]), self.layer2(feats[1]), feats[2]], dim=1)
        return self.out_layer(x)

class CIDE(nn.Module):
    def __init__(self, args, emb_dim):
        super().__init__()
        self.args = args
        self.vit_processor = ViTImageProcessor.from_pretrained(args.vit_model, resume_download=True)
        self.vit_model = ViTForImageClassification.from_pretrained(args.vit_model, resume_download=True)
        for param in self.vit_model.parameters():
            param.requires_grad = False
        
        self.fc = nn.Sequential(
            nn.Linear(1000, 400),
            nn.GELU(),
            nn.Linear(400, args.no_of_classes)
        )
        self.dim = emb_dim
        self.m = nn.Softmax(dim=1)
        
        self.embeddings = nn.Parameter(torch.randn(self.args.no_of_classes, self.dim))
        self.embedding_adapter = EmbeddingAdapter(emb_dim=self.dim)
        
        self.gamma = nn.Parameter(torch.ones(self.dim) * 1e-4)
        
    def pad_to_make_square(self, x):
        y = 255*((x+1)/2)
        y = torch.permute(y, (0,2,3,1))
        bs, _, h, w = x.shape
        if w>h:
            patch = torch.zeros(bs, w-h, w, 3).to(x.device)
            y = torch.cat([y, patch], axis=1)
        else:
            patch = torch.zeros(bs, h, h-w, 3).to(x.device)
            y = torch.cat([y, patch], axis=2)
        return y.to(torch.int)
    
    def forward(self, x):
        
        # make the image of dimension 480*640 into a square and downsample to utilize pretrained knowledge in the ViT
        y = self.pad_to_make_square(x)
        # use torch.no_grad() to prevent gradient flow through the ViT since it is kept frozen
        with torch.no_grad():
            inputs = self.vit_processor(images=y, return_tensors="pt").to(x.device)
            vit_outputs = self.vit_model(**inputs)
            vit_logits = vit_outputs.logits
            
        class_probs = self.fc(vit_logits)
        class_probs = self.m(class_probs)
        
        class_embeddings = class_probs @ self.embeddings
        conditioning_scene_embedding = self.embedding_adapter(class_embeddings, self.gamma) 
        
        return conditioning_scene_embedding
        
        
class EcoDepth(nn.Module):
    def __init__(self, args=None, min_depth = 0.1):
        super().__init__()
        self.max_depth = args.max_depth
        self.min_depth = args.min_depth

        self.args = args
        embed_dim = 192
        channels_in = embed_dim*8
        channels_out = embed_dim

        self.encoder = EcoDepthEncoder(out_dim=channels_in, dataset='nyu', args = args)
        self.decoder = Decoder(channels_in, channels_out, args)
        self.decoder.init_weights()

        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, 1, kernel_size=3, stride=1, padding=1))
            
        for m in self.last_layer_depth.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)



    def forward(self, x):  
        b, c, h, w = x.shape
        x = x*2.0 - 1.0  # normalize to [-1, 1]

        conv_feats = self.encoder(x)
        if h == 480 or h == 352:
            conv_feats = conv_feats[:, :, :-1, :-1]

        out = self.decoder([conv_feats])
        out_depth = self.last_layer_depth(out)            
        out_depth = torch.sigmoid(out_depth) * self.max_depth

        return {'pred_d': out_depth}

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, args):
        super().__init__()
        self.deconv = args.num_deconv
        self.in_channels = in_channels
        self.args = args    
        self.deconv_layers = self._make_deconv_layer(
            args.num_deconv,
            args.num_filters,
            args.deconv_kernels,
        )
    
        conv_layers = []
        conv_layers.append(
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels=args.num_filters[-1],
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1))
        conv_layers.append(
            build_norm_layer(dict(type='BN'), out_channels)[1])
        conv_layers.append(nn.ReLU(inplace=True))
        self.conv_layers = nn.Sequential(*conv_layers)
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, conv_feats):
        # import ipdb;ipdb.set_trace()
        out = self.deconv_layers(conv_feats[0])
        out = self.conv_layers(out)

        out = self.up(out)
        out = self.up(out)

        return out

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        
        layers = []
        in_planes = self.in_channels
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),
                    in_channels=in_planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            in_planes = planes

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding

    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)

