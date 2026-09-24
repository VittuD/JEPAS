"""ViTDet-style Simple Feature Pyramid neck for a single-scale ViT grid.

None of this repo's backbones are hierarchical (multi-scale) like a ResNet or
Swin, so there's nothing to hand torchvision's usual FPN out of the box --
this builds a 4-level pyramid {p2..p5} out of one (B, D, H, W) grid the same
way "Exploring Plain Vision Transformer Backbones for Object Detection"
(Li et al. 2022, ViTDet) does: two deconvs for the finest level, one deconv
for the next, identity, then a maxpool for the coarsest -- each followed by a
1x1 lateral conv + a 3x3 smoothing conv to a common out_channels. Norm here
is a plain channel-wise LayerNorm2d rather than the paper's exact recipe;
close enough for our purposes, this is our own eval harness, not a
reproduction of a specific published checkpoint.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn

from experiments.dense_eval.backbones import GridBackbone


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class SimpleFeaturePyramid(nn.Module):
    """Frozen grid backbone + trainable 4-level pyramid neck.

    Satisfies torchvision's custom-backbone contract directly (forward(x) ->
    OrderedDict[str, Tensor], plus an .out_channels attribute) -- no need for
    torchvision's BackboneWithFPN/IntermediateLayerGetter machinery, which
    assumes a plain multi-stage CNN with named submodules that doesn't apply
    to a single-scale ViT grid.
    """

    def __init__(self, backbone: GridBackbone, *, out_channels: int = 256) -> None:
        super().__init__()
        self.backbone = backbone
        self.out_channels = out_channels
        d = backbone.embed_dim

        # Four levels at scale {x4, x2, x1, x0.5} relative to the backbone's
        # native grid, matching ViTDet's simple pyramid.
        self.to_p2 = nn.Sequential(
            nn.ConvTranspose2d(d, d // 2, kernel_size=2, stride=2),
            LayerNorm2d(d // 2),
            nn.GELU(),
            nn.ConvTranspose2d(d // 2, d // 4, kernel_size=2, stride=2),
        )
        self.to_p3 = nn.ConvTranspose2d(d, d // 2, kernel_size=2, stride=2)
        self.to_p4 = nn.Identity()
        self.to_p5 = nn.MaxPool2d(kernel_size=2, stride=2)

        stage_dims = {"p2": d // 4, "p3": d // 2, "p4": d, "p5": d}
        self.lateral = nn.ModuleDict(
            {name: nn.Conv2d(dim, out_channels, kernel_size=1) for name, dim in stage_dims.items()}
        )
        self.smooth = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    LayerNorm2d(out_channels),
                )
                for name in stage_dims
            }
        )

    def forward(self, x: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
        grid = self.backbone(x)  # (B, D, H, W); frozen, produced under no_grad
        stages = {"p2": self.to_p2(grid), "p3": self.to_p3(grid), "p4": self.to_p4(grid), "p5": self.to_p5(grid)}
        out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        for name, feat in stages.items():
            out[name] = self.smooth[name](self.lateral[name](feat))
        return out
