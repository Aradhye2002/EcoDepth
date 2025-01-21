import torch.nn as nn
import torch
from transformers import ViTImageProcessor, ViTForImageClassification, ViTConfig
from timm.models.layers import trunc_normal_
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import torch.nn.functional as F
from einops import repeat
import lightning as L
from utils import pad, unpad, silog
from optimizer import get_optimizer
from metrics import compute_metrics
from utils import eigen_crop, garg_crop, custom_crop, no_crop

NUM_DECONV = 3
NUM_FILTERS = [32, 32, 32]
DECONV_KERNELS = [2, 2, 2]
VIT_MODEL = 'google/vit-base-patch16-224'


def pad_to_make_square(x):
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


class EmbeddingAdapter(nn.Module):
    def __init__(self, emb_dim=768):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, texts, gamma):
        emb_transformed = self.fc(texts)
        texts = texts + gamma * emb_transformed
        texts = repeat(texts, 'n c -> n b c', b=1)
        return texts

class EcoDepthEncoder(nn.Module):
    def __init__(
        self, 
        out_dim=1024, 
        ldm_prior=[320, 640, 1280+1280], 
        sd_path=None, 
        emb_dim=768, 
        args=None,
        train_from_scratch=False,
    ):
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
        
        if train_from_scratch:
            self.apply(self._init_weights)
        
        self.cide_module = CIDE(args, emb_dim, train_from_scratch)
        
        self.config = OmegaConf.load('../v1-inference.yaml')
        unet_config = self.config.model.params.unet_config
        first_stage_config = self.config.model.params.first_stage_config
        
        if train_from_scratch:
            if sd_path is None:
                sd_path = '../../checkpoints/v1-5-pruned-emaonly.ckpt'
            # unet_config.params.ckpt_path = sd_path
        
        self.unet = instantiate_from_config(unet_config)
        self.encoder_vq = instantiate_from_config(first_stage_config)
        del self.encoder_vq.decoder
        del self.unet.out

        for param in self.encoder_vq.parameters():
            param.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        with torch.no_grad():
            # convert the input image to latent space and scale.
            latents = self.encoder_vq.encode(x).mode().detach() * self.config.model.params.scale_factor

        conditioning_scene_embedding = self.cide_module(x)

        t = torch.ones((x.shape[0],), device=x.device).long()
        outs = self.unet(latents, t, context=conditioning_scene_embedding)

        feats = [outs[0], outs[1], torch.cat([outs[2], F.interpolate(outs[3], scale_factor=2)], dim=1)]
        x = torch.cat([self.layer1(feats[0]), self.layer2(feats[1]), feats[2]], dim=1)
        return self.out_layer(x)

class CIDE(nn.Module):
    def __init__(self, args, emb_dim, train_from_scratch):
        super().__init__()
        self.args = args
        self.vit_processor = ViTImageProcessor.from_pretrained(VIT_MODEL, resume_download=True)
        if train_from_scratch:
            vit_config = ViTConfig(num_labels=1000)
            self.vit_model = ViTForImageClassification(vit_config)
        else:
            self.vit_model = ViTForImageClassification.from_pretrained(VIT_MODEL, resume_download=True)
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
    
    def forward(self, x):
        y = pad_to_make_square(x)
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
        
class EcoDepth(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.max_depth = args.max_depth

        self.args = args
        embed_dim = 192
        channels_in = embed_dim * 8
        channels_out = embed_dim

        self.encoder = EcoDepthEncoder(out_dim=channels_in, args = args, train_from_scratch=args.train_from_scratch)
        self.decoder = Decoder(channels_in, channels_out, args)
        
        if args.eval_crop == "eigen":
            self.eval_crop = eigen_crop
        elif args.eval_crop == "garg":
            self.eval_crop = garg_crop
        elif args.eval_crop == "custom":
            self.eval_crop = custom_crop
        else:
            self.eval_crop = no_crop
        
        # Only support finetuning for now
        assert not args.train_from_scratch
        
        if args.train_from_scratch:
            self.decoder.init_weights()

        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, 1, kernel_size=3, stride=1, padding=1))
            
        for m in self.last_layer_depth.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x must be a pytorch tensor of shape (bs, 3, h, w)
        # and the each value ranges between [0, 1]
        _, _, h, _ = x.shape
        x = x*2.0 - 1.0  # normalize to [-1, 1]
        
        x, padding = pad(x, 64)
        conv_feats = self.encoder(x)
        out = self.decoder([conv_feats])
        out = unpad(out, padding)
        out_depth = self.last_layer_depth(out)            
        pred = torch.sigmoid(out_depth) * self.max_depth
        # pred is a pt of shape (bs, 1, h, w)
        # where each value ranges between [0, self.max_depth]
        return pred
    
    def training_step(self, batch, batch_idx):
        image, depth = batch["image"], batch["depth"]
        pred = self(image)
        loss = silog(pred, depth)
        return loss
    
    def _shared_eval_step(self, batch, batch_idx, prefix):
        image, depth = batch["image"], batch["depth"]
        depth = self.eval_crop(depth)
        image_concat = torch.cat([image, image.flip(-1)])
        pred_concat = self(image_concat)
        pred = ((pred_concat[0] + pred_concat[1].flip(-1))/2).unsqueeze(0)
        loss = silog(pred, depth)
        metrics = compute_metrics(pred, depth, self.args)
        self.log(f"{prefix}_loss", loss)
        self.log_dict(metrics)
        return loss, metrics
        
    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, "val")
    
    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, "test")
        
    def configure_optimizers(self):
        optimizer = get_optimizer(self, self.args)
        return optimizer
    
        
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, args):
        super().__init__()
        self.deconv = NUM_DECONV
        self.in_channels = in_channels
        self.args = args    
        self.deconv_layers = self._make_deconv_layer(
            NUM_DECONV,
            NUM_FILTERS,
            DECONV_KERNELS,
        )
    
        conv_layers = []
        conv_layers.append(
            nn.Conv2d(
                in_channels=NUM_FILTERS[-1],
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1))
        conv_layers.append(nn.BatchNorm2d(out_channels))
        conv_layers.append(nn.ReLU(inplace=True))
        self.conv_layers = nn.Sequential(*conv_layers)
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, conv_feats):
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
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
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
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


