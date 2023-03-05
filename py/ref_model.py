#-*-coding:utf-8-*-

"""
    Ref NeRF network details. To be finished ...
"""
import torch
from torch import nn
from numpy import log
from py.nerf_base import NeRF
from typing import Optional, Tuple
from torch.nn import functional as F
from py.ref_func import generate_ide_fn
from py.nerf_helper import makeMLP, hash_encoding, linear_to_srgb, positional_encoding

""" 
NeRF ---+---------> Mip-NeRF
        +---------> Ref-NeRF
"""

# Inherited from NeRF (base class), Mip-NeRF is not concerned with this

class RefNeRF(NeRF):
    def __init__(self, 
        position_flevel, sh_max_level, # position_flevel = 10
        bottle_neck_dim = 128,
        hidden_unit = 256, 
        output_dim = 256, 
        use_srgb = False,
        cat_origin = False,
        perturb_bottle_neck_w = 0.1,
        instant_ngp = False
    ) -> None:
        super().__init__(position_flevel, cat_origin, lambda x: x)          # density is not activated during render
        self.sh_max_level = sh_max_level
        self.bottle_neck_dim = bottle_neck_dim
        self.dir_enc_dim = ((1 << sh_max_level) - 1 + sh_max_level) << 1
        self.instant_ngp = instant_ngp

        self.encoding = hash_encoding(3)
        extra_width = 3 if cat_origin else 0
        if instant_ngp:
            hidden_unit = 64
            output_dim = 64 # 256 without instant-ngp, turn it into small network
            bottle_neck_dim = 32
            spatial_module_list = makeMLP(self.encoding.n_output_dims + extra_width, hidden_unit)
        else:
            spatial_module_list = makeMLP(6 * position_flevel + extra_width, hidden_unit)
        for _ in range(3):
            spatial_module_list.extend(makeMLP(hidden_unit, hidden_unit))

        # spatial MLP part (spa_xxxx)
        # hash encoding is used in it
        self.spa_block1 = nn.Sequential(*spatial_module_list)       # MLP before skip connection
        if instant_ngp:
                self.spa_block2 = nn.Sequential(
                # *makeMLP(hidden_unit + 6 * position_flevel + extra_width, hidden_unit),
                *makeMLP(hidden_unit + self.encoding.n_output_dims + extra_width, hidden_unit),
                *makeMLP(hidden_unit, hidden_unit), *makeMLP(hidden_unit, hidden_unit),
                *makeMLP(hidden_unit, output_dim)
            )
        else:
            self.spa_block2 = nn.Sequential(
                *makeMLP(hidden_unit + 6 * position_flevel + extra_width, hidden_unit),
                # *makeMLP(hidden_unit + self.encoding.n_output_dims + extra_width, hidden_unit),
                *makeMLP(hidden_unit, hidden_unit), *makeMLP(hidden_unit, hidden_unit),
                *makeMLP(hidden_unit, output_dim)
            )

        self.rho_tau_head = nn.Linear(output_dim, 2)
        self.norm_col_tint_head = nn.Linear(output_dim, 9)  # output normal prediction, color, tint (all 3)
        self.bottle_neck = nn.Linear(output_dim, bottle_neck_dim) # bottle_neck

        self.spec_rgb_head = nn.Sequential(*makeMLP(output_dim, 3, nn.Sigmoid()))

        dir_input_dim = 1 + bottle_neck_dim + self.dir_enc_dim
        directional_module_list = makeMLP(dir_input_dim, hidden_unit)
        for _ in range(3):
            directional_module_list.extend(makeMLP(hidden_unit, hidden_unit))

        self.dir_block1 = nn.Sequential(*directional_module_list)
        self.dir_block2 = nn.Sequential(                   # skip connection ()
            *makeMLP(hidden_unit + dir_input_dim, hidden_unit),
            *makeMLP(hidden_unit, hidden_unit), *makeMLP(hidden_unit, output_dim),
            *makeMLP(hidden_unit, output_dim)
        )
        # \rho is roughness coefficient, \tau is density

        self.use_srgb = use_srgb
        self.perturb_bottle_neck_w = perturb_bottle_neck_w
        self.integrated_dir_enc = generate_ide_fn(sh_max_level)
        self.apply(self.init_weight)

    # for coarse network, input is obtained by sampling, sampling result is (ray_num, point_num, 9), (depth) (ray_num, point_num)
    # core part
    def forward(self, pts:torch.Tensor, ray_d: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # maybe the positional encoding should not be abandoned
        # how to insert hash-encoding ?
        if self.instant_ngp:
            # encoding networks
            # [important]todo: add positional encoding, making it work with hash_encoding
            modify_pts = pts[:, :, :3].view([-1, 3])
            encoded_x = self.encoding(modify_pts)
            encoded_x = encoded_x.view(pts.shape[0], pts.shape[1], self.encoding.n_output_dims)
        else:
            position_dim = 6 * self.position_flevel
            # [important]todo: need change it into integrated positional encoding(IPE)
            encoded_x = positional_encoding(pts[:, :, :3], self.position_flevel)
            encoded_x = encoded_x.view(pts.shape[0], pts.shape[1], position_dim)

        if self.cat_origin:
            encoded_x = torch.cat((pts[:, :, :3], encoded_x), -1)
        # todo: check whether spa_block1 and spa_block2 is correct
        x_tmp = self.spa_block1(encoded_x)
        encoded_x = torch.cat((encoded_x, x_tmp), dim = -1)
        intermediate = self.spa_block2(encoded_x)               # output of spatial network
        # todo: check whether norm_col_tint_head is correct
        [normal, diffuse_rgb, spec_tint] = self.norm_col_tint_head(intermediate).split((3, 3, 3), dim = -1)
        # todo[finished]: check whether rho_tau_head is correct, check it in predict_density_and_grad_fn
        # res = raw + bias
        # raw = roughness_activation(...)
        roughness, density = self.rho_tau_head(intermediate).split((1, 1), dim = -1)
        # todo: check whether roughness is correct
        roughness = F.softplus(roughness - 1.)
        # todo: check whether bottle_neck is correct
        # dense_layar
        spa_info_b = self.bottle_neck(intermediate)
        if self.training == True:
            # noise added?
            spa_info_b = spa_info_b + torch.normal(0, self.perturb_bottle_neck_w, spa_info_b.shape, device = spa_info_b.device)

        normal = -normal / (normal.norm(dim = -1, keepdim = True) + 1e-7)
        # needs further validation
        ray_d = pts[..., 3:] if ray_d is None else ray_d
        # reflect_r = ray_d - 2. * torch.sum(ray_d * normal, dim = -1, keepdim = True) * normal
        # todo[finished]: change reflection function
        reflect_r = 2. * torch.sum(ray_d * normal, dim=-1, keepdim=True) * normal - ray_d
        # todo: check whether IDE is correct, check it in dir_enc_fn
        wr_ide = self.integrated_dir_enc(reflect_r, roughness)
        # todo: check dot_product [finished]
        nv_dot = torch.sum(normal * ray_d, dim = -1, keepdim = True)

        # todo: check whether all inputs is correct
        all_inputs = torch.cat((spa_info_b, wr_ide, nv_dot), dim = -1)
        # todo: check whether dir_block1 is correct
        r_tmp = self.dir_block1(all_inputs)
        all_inputs = torch.cat((all_inputs, r_tmp), dim = -1)
        # todo: check whether spec_rgb_head is correct, check it in rgb_activation
        # there always has a bias, check it
        specular_rgb = self.spec_rgb_head(self.dir_block2(all_inputs)) * F.sigmoid(spec_tint) 
        if self.use_srgb == True:
            diffuse_rgb = torch.sigmoid(diffuse_rgb - log(3.))
            rgb = linear_to_srgb(specular_rgb + diffuse_rgb)
        else:
            diffuse_rgb = torch.sigmoid(diffuse_rgb)
            rgb = specular_rgb + diffuse_rgb
        return torch.cat((rgb, density), dim = -1), normal      # output (ray_num, point_num, 4) + (ray_num, point_num, 3)
    
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
            torch.ones_like(func_val, device = func_val.device), retain_graph = True
        )
        # return grad / grad.norm(dim = -1, keepdim = True)
        grad_norm = grad.norm(dim = -1, keepdim = True)
        return grad / torch.maximum(torch.full_like(grad_norm, 1e-5), grad_norm)


class WeightedNormalLoss(nn.Module):
    def __init__(self, size_average = False):
        super().__init__()
        self.size_average = size_average        # average (per point, not per ray)
    
    # weight (ray_num, point_num)
    def forward(self, weight:torch.Tensor, d_norm: torch.Tensor, p_norm: torch.Tensor) -> torch.Tensor:
        dot_diff = 1. - torch.sum(d_norm * p_norm, dim = -1)
        return torch.mean(weight * dot_diff) if self.size_average == True else torch.sum(weight * dot_diff)

class BackFaceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    # 注意，可以使用pts[..., 3:] 作为输入
    def forward(self, weight:torch.Tensor, normal: torch.Tensor, ray_d: torch.Tensor) -> torch.Tensor:
        return torch.mean(weight * F.relu(torch.sum(normal * ray_d, dim = -1)))
