from typing import List, Optional, Tuple

import torch
from mmcv.ops import QueryAndGroup, furthest_point_sample, gather_points
from torch import nn
from torch.nn import functional as F


class _SharedMLP(nn.Sequential):
    def __init__(self, channels: List[int], bn: bool) -> None:
        super().__init__()
        for index, (in_channels, out_channels) in enumerate(
                zip(channels, channels[1:])):
            layer = nn.Sequential()
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                             bias=not bn)
            nn.init.kaiming_normal_(conv.weight)
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0)
            layer.add_module("conv", conv)
            if bn:
                layer.add_module("bn", nn.BatchNorm2d(out_channels))
            layer.add_module("activation", nn.ReLU(inplace=True))
            self.add_module(f"layer{index}", layer)


class PointnetSAModuleVotes(nn.Module):
    def __init__(
        self,
        *,
        mlp: List[int],
        npoint: int,
        radius: float,
        nsample: int,
        bn: bool = True,
        use_xyz: bool = True,
        pooling: str = "max",
        sigma: Optional[float] = None,
        normalize_xyz: bool = False,
        sample_uniformly: bool = False,
        ret_unique_cnt: bool = False,
    ) -> None:
        super().__init__()
        if pooling not in {"max", "avg", "rbf"}:
            raise ValueError(f"Unsupported pooling mode: {pooling}")
        if ret_unique_cnt and not sample_uniformly:
            raise ValueError("ret_unique_cnt requires sample_uniformly=True")

        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.pooling = pooling
        self.sigma = sigma if sigma is not None else radius / 2
        self.ret_unique_cnt = ret_unique_cnt
        self.normalize_xyz = normalize_xyz
        channels = list(mlp)
        if use_xyz:
            channels[0] += 3
        self.grouper = QueryAndGroup(
            radius,
            nsample,
            use_xyz=use_xyz,
            return_grouped_xyz=True,
            normalize_xyz=normalize_xyz,
            uniform_sample=sample_uniformly,
            return_unique_cnt=ret_unique_cnt,
        )
        self.mlp_module = _SharedMLP(channels, bn=bn)

    def forward(
        self,
        xyz: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        inds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if inds is None:
            inds = furthest_point_sample(xyz, self.npoint)
        elif inds.shape[1] != self.npoint:
            raise ValueError("inds must contain one index per sampled point")

        new_xyz = gather_points(xyz.transpose(1, 2).contiguous(), inds).transpose(
            1, 2).contiguous()
        grouped = self.grouper(xyz, new_xyz, features)
        grouped_features, grouped_xyz, *extra = grouped
        new_features = self.mlp_module(grouped_features)

        if self.pooling == "max":
            new_features = F.max_pool2d(new_features, [1, new_features.size(3)])
        elif self.pooling == "avg":
            new_features = F.avg_pool2d(new_features, [1, new_features.size(3)])
        else:
            grouped_xyz = grouped_xyz - new_xyz.transpose(1, 2).unsqueeze(-1)
            if self.normalize_xyz:
                grouped_xyz = grouped_xyz / self.radius
            rbf = torch.exp(-grouped_xyz.pow(2).sum(1) / (2 * self.sigma**2))
            new_features = (new_features * rbf.unsqueeze(1)).sum(
                -1, keepdim=True
            ) / float(self.nsample)
        new_features = new_features.squeeze(-1)

        if self.ret_unique_cnt:
            return new_xyz, new_features, inds, extra[0]
        return new_xyz, new_features, inds
