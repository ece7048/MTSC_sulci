"""3D model wrappers used by pre-training and classification workflows."""

from __future__ import division, print_function

import torch
import torch.nn.functional as F
import tf2onnx
from monai.networks.nets import Discriminator, Generator
from onnx2pytorch import ConvertModel

from MTSC_sulci.utilities.DiffModel import DiffusionModelUNet
from MTSC_sulci.utilities.SwiftUnet3D import SwinUNETR


def pytorch_model(model):
    """Convert a Keras model object to an equivalent PyTorch module."""
    onnx_model, _ = tf2onnx.convert.from_keras(model)
    return ConvertModel(onnx_model)


class SwinMT(torch.nn.Module):
    """SwinUNETR backbone with reconstruction and class-prediction heads."""

    def __init__(self, f=112, classes=2, roi=64, upscale=64):
        super().__init__()
        self.f = f
        self.roi = roi
        self.classes = classes
        self.upscale = upscale
        self.dim = 1536
        up = 1563
        depth = 2
        self.swnet = SwinUNETR(
            img_size=(roi, roi, roi),
            in_channels=1,
            out_channels=1,
            feature_size=f,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=True,
            roi_size=roi,
        )
        self.linear_encod = torch.nn.Linear(self.dim, self.classes)
        self.linear_encod2 = torch.nn.Linear(up, self.classes)
        self.maxpool = torch.nn.MaxPool3d(depth)

    def forward(self, x, one_hot=False):
        """Return reconstructed volume logits and log-probability class scores."""
        out1, enc3 = self.swnet(x)

        if self.roi >= 64:
            out2 = self.maxpool(enc3)
        else:
            out2 = enc3
        out2 = out2.view(out2.shape[0], -1)

        if (self.roi != self.upscale) and (out2.shape[1] != self.dim):
            out2 = self.linear_encod2(out2)
            print("upscale patch use...")
        else:
            out2 = self.linear_encod(out2)
        return out1, F.log_softmax(out2, -1)


class Discrim(torch.nn.Module):
    """3D discriminator used by adversarial pre-training."""

    def __init__(self, roi=64):
        super().__init__()
        self.disc_net = Discriminator(
            in_shape=(1, roi, roi, roi),
            channels=(8, 16, 32, 64, 1),
            strides=(2, 2, 2, 2, 2, 1),
            num_res_units=1,
            kernel_size=5,
        )
        self.gen_net = Generator(
            latent_shape=64,
            start_shape=(64, 8, 8),
            channels=[32, 16, 8, 1],
            strides=[2, 2, 2, 1],
        )

    def forward(self, x):
        """Score a generated or real 3D volume."""
        return self.disc_net(x)


class DiffusionUnet(torch.nn.Module):
    """Diffusion U-Net with an auxiliary class-prediction head."""

    def __init__(self, f=112, classes=2):
        super().__init__()
        self.f = f
        self.classes = classes
        self.difnet = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            num_channels=[32, 32, 64],
            attention_levels=[False, False, True],
            num_head_channels=[0, 0, 64],
            num_res_blocks=2,
            with_conditioning=True,
            cross_attention_dim=32,
        )
        self.linear_encod = torch.nn.Linear(512, classes)
        self.maxpool = torch.nn.MaxPool3d(4)
        self.m = torch.nn.Flatten()

    def forward(self, x, timesteps, context):
        """Return denoised volume logits and log-probability class scores."""
        out1, enc3 = self.difnet(x, timesteps, context)
        out2 = self.maxpool(enc3)
        out2 = self.m(out2)
        out2 = self.linear_encod(out2)
        return out1, F.log_softmax(out2, -1)
