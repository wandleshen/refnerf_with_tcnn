"""NeRF 基本工具
"""
import torch
import tinycudann as tcnn
import commentjson as json

def hash_encoding(dim:int):
    with open("configs/config_hash.json") as f:
        config = json.load(f)
    encoding = tcnn.Encoding(dim, config['encoding'], dtype=torch.float32)
    return encoding

def save_model(model, path:str, optimizer = None, extra_params:dict = None):
    """保存模型"""
    state_dict = {'model': model.state_dict()}
    if optimizer is not None:
        state_dict['optimizer'] = optimizer.state_dict()
    state_dict.update(extra_params)
    torch.save(state_dict, path)

def add_mlp(in_chan, out_chan, act = torch.nn.ReLU(), batch_norm = False):
    """添加 MLP 层"""
    layers = [torch.nn.Linear(in_chan, out_chan).cuda()]
    if batch_norm:
        layers.append(torch.nn.BatchNorm1d(out_chan).cuda())
    if act is not None:    
        layers.append(act.cuda())
    return layers

def linear_to_srgb(linear: torch.Tensor, eps: float = None) -> torch.Tensor:
    """代码来自 https://github.com/google-research/multinerf 并修改为 PyTorch 版本"""
    if eps is None:
        eps = torch.full((1, ), torch.finfo(torch.float32).eps, device = linear.device)
    srgb0 = 323 / 25 * linear
    srgb1 = (211 * torch.maximum(eps, linear)**(5 / 12) - 11) / 200
    return torch.where(linear <= 0.0031308, srgb0, srgb1)