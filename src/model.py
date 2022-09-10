"""Ref-NeRF 模型，开模！
"""

import ref_utils
import nerf_utils
import nerf

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional, Tuple

# 继承自 NeRF
class RefNeRF(nerf.NeRF):
    def __init__(self,
        sph_harm_level,
        bottle_neck_dim = 128,
        hidden_unit = 256,
        output_dim = 256,
        use_srgb = False,
        perturb_bottle_neck_w = 0.1,
        ) -> None:
        super().__init__(lambda x: x)
        self.sph_harm_level = sph_harm_level
        self.bottle_neck_dim = bottle_neck_dim
        self.dir_enc_dim = ((1 << sph_harm_level) - 1 + sph_harm_level) << 1

        # spatial mlp
        self.encoding = nerf_utils.hash_encoding(3)
        spatial_modules = nerf_utils.add_mlp(self.encoding.n_output_dims, hidden_unit)
        for _ in range(3):
            spatial_modules.extend(nerf_utils.add_mlp(hidden_unit, hidden_unit))
        self.spatial_module1 = nn.Sequential(*spatial_modules)
        self.spatial_module2 = nn.Sequential(
            *nerf_utils.add_mlp(hidden_unit + self.encoding.n_output_dims, hidden_unit),
            *nerf_utils.add_mlp(hidden_unit, hidden_unit), *nerf_utils.add_mlp(hidden_unit, hidden_unit),
            *nerf_utils.add_mlp(hidden_unit, output_dim)
            )
        
        self.rho_tau_head = nn.Linear(output_dim, 2)
        self.norm_color_tint_head = nn.Linear(output_dim, 9)
        self.bottle_neck = nn.Linear(output_dim, bottle_neck_dim)
        self.specular_rgb_head = nn.Sequential(*nerf_utils.add_mlp(output_dim, 3, nn.Sigmoid()))

        # directional mlp
        dir_input_dim = 1 + bottle_neck_dim + self.dir_enc_dim
        directional_modules = nerf_utils.add_mlp(dir_input_dim, hidden_unit)
        for _ in range(3):
            directional_modules.extend(nerf_utils.add_mlp(hidden_unit, hidden_unit))
        self.directional_module1 = nn.Sequential(*directional_modules)
        self.directional_module2 = nn.Sequential(
            *nerf_utils.add_mlp(hidden_unit + dir_input_dim, hidden_unit),
            *nerf_utils.add_mlp(hidden_unit, hidden_unit), *nerf_utils.add_mlp(hidden_unit, hidden_unit),
            *nerf_utils.add_mlp(hidden_unit, output_dim)
            )
        
        self.use_srgb = use_srgb
        self.perturb_bottle_neck_w = perturb_bottle_neck_w
        self.integrated_dir_enc = ref_utils.generate_ide_fn(sph_harm_level)
        self.apply(self.init_weight)

    def forward(self, pts, ray_d: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded_pts = self.encoding(pts)

        # spatial mlp
        pts_tmp = self.spatial_module1(encoded_pts)
        encoded_pts = torch.cat((encoded_pts, pts_tmp), dim = -1)
        intermediate = self.spatial_module2(encoded_pts)

        [normal, diffuse_rgb, tint] = self.norm_color_tint_head(intermediate).split((3, 3, 3), dim=-1)
        rho, tau = self.rho_tau_head(intermediate).split((1, 1), dim=-1)
        rho = F.softplus(rho-1)
        b = self.bottle_neck(intermediate)
        if self.training:
            b = b + torch.normal(0, self.perturb_bottle_neck_w, b.shape, device = b.device)
        normal = -normal / (normal.norm(dim=-1, keepdim=True) + 1e-6)

        # directional mlp
        ray_d = pts[..., 3:] if ray_d is None else ray_d
        reflect = ref_utils.reflect(ray_d, normal)
        ide = self.integrated_dir_enc(-reflect, rho)
        dot_norm_view = np.sum(normal * -ray_d, dim=-1, keepdim=True)

        dir_input = torch.cat((b, ide, dot_norm_view), dim=-1)
        dir_tmp = self.directional_module1(dir_input)
        dir_input = torch.cat((dir_input, dir_tmp), dim=-1)

        specular_rgb = self.specular_rgb_head(self.directional_module2(dir_input) * F.sigmond(tint))
        if self.use_srgb:
            diffuse_rgb = torch.sigmoid(diffuse_rgb - np.log(3.0))
            rgb = nerf_utils.linear_to_srgb(specular_rgb + diffuse_rgb)
        else:
            diffuse_rgb = torch.sigmoid(diffuse_rgb)
            rgb = specular_rgb + diffuse_rgb
        return torch.cat((rgb, tau), dim=-1), normal

    @staticmethod
    def coarse_grad_select(fine_grads: torch.Tensor, sort_inds: torch.Tensor, c_pnum: int) -> torch.Tensor:
        ray_num, all_pnum, _ = fine_grads.shape
        target_device = fine_grads.device 
        selector = torch.cat([
            torch.full((ray_num, all_pnum - c_pnum), False, device = target_device),
            torch.full((ray_num, c_pnum), True, device = target_device),
        ], dim = -1)
        selector = torch.gather(selector, -1, sort_inds)
        return fine_grads[selector].reshape(ray_num, c_pnum, -1)

    @staticmethod
    def get_grad(func_val: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:  # remember: grad goes from low to high
        grad, = torch.autograd.grad(func_val, inputs, 
            torch.ones_like(func_val, device=func_val.device), retain_graph=True
        )
        return grad / grad.norm(dim=-1, keepdim=True)

class WeightedNormalLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    # weight (ray_num, point_num)
    def forward(self, weight:torch.Tensor, d_norm: torch.Tensor, p_norm: torch.Tensor) -> torch.Tensor:
        dot_diff = 1. - torch.sum(d_norm * p_norm, dim=-1)
        return torch.mean(weight * dot_diff)

class BackFaceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    # 注意，可以使用pts[..., 3:] 作为输入
    def forward(self, weight:torch.Tensor, normal: torch.Tensor, ray_d: torch.Tensor) -> torch.Tensor:
        return torch.mean(weight * F.relu(torch.sum(normal * ray_d, dim = -1)))