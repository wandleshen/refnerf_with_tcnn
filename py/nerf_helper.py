"""
    NeRF helper functions
    @author Enigmatisms @date 2022.4.24
"""
import torch
import commentjson as json
import tinycudann as tcnn
# todo: install jax
import jax.numpy as jnp

# hashing encoding, mind where it is used
def hash_encoding(dim:int):
    with open("configs/config_hash.json") as f:
        config = json.load(f)
    encoding = tcnn.Encoding(dim, config['encoding'], dtype=torch.float32)
    return encoding

def saveModel(model, path:str, other_stuff: dict = None, opt = None, amp = None):
    checkpoint = {'model': model.state_dict(),}
    if not amp is None:
        checkpoint['amp'] =  amp.state_dict()
    if not opt is None:
        checkpoint['optimizer'] = opt.state_dict()
    if not other_stuff is None:
        checkpoint.update(other_stuff)
    torch.save(checkpoint, path)
    
def makeMLP(in_chan, out_chan, act = torch.nn.ReLU(), batch_norm = False):
    modules = [torch.nn.Linear(in_chan, out_chan)]
    if batch_norm == True:
        modules.append(torch.nn.BatchNorm1d(out_chan))
    if not act is None:
        modules.append(act)
    return modules

# from pytorch.org https://discuss.pytorch.org/t/finding-source-of-nan-in-forward-pass/51153/2
def nan_hook(self, inp, output):
    if not isinstance(output, tuple):
        outputs = [output]
    else:
        outputs = output

    for i, out in enumerate(outputs):
        nan_mask = torch.isnan(out)
        if nan_mask.any():
            print("In", self.__class__.__name__)
            raise RuntimeError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:", out[nan_mask.nonzero()[:, 0].unique(sorted=True)])

# todo: apply lift_gaussian and cylinder_to_gaussian to this project
# variable rays in multinerf is input that packet all input together
def lift_gaussian(d, t_mean, t_var, r_var, diag):
  """Lift a Gaussian defined along a ray to 3D coordinates."""
  mean = d[..., None, :] * t_mean[..., None]

  d_mag_sq = jnp.maximum(1e-10, jnp.sum(d**2, axis=-1, keepdims=True))

  if diag:
    d_outer_diag = d**2
    null_outer_diag = 1 - d_outer_diag / d_mag_sq
    t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :]
    xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]
    cov_diag = t_cov_diag + xy_cov_diag
    return mean, cov_diag
  else:
    d_outer = d[..., :, None] * d[..., None, :]
    eye = jnp.eye(d.shape[-1])
    null_outer = eye - d[..., :, None] * (d / d_mag_sq)[..., None, :]
    t_cov = t_var[..., None, None] * d_outer[..., None, :, :]
    xy_cov = r_var[..., None, None] * null_outer[..., None, :, :]
    cov = t_cov + xy_cov
    return mean, cov
def cylinder_to_gaussian(d, t0, t1, radius, diag):
  """Approximate a cylinder as a Gaussian distribution (mean+cov).

  Assumes the ray is originating from the origin, and radius is the
  radius. Does not renormalize `d`.

  Args:
    d: jnp.float32 3-vector, the axis of the cylinder
    t0: float, the starting distance of the cylinder.
    t1: float, the ending distance of the cylinder.
    radius: float, the radius of the cylinder
    diag: boolean, whether or the Gaussian will be diagonal or full-covariance.

  Returns:
    a Gaussian (mean and covariance).
  """
  t_mean = (t0 + t1) / 2
  r_var = radius**2 / 4
  t_var = (t1 - t0)**2 / 12
  return lift_gaussian(d, t_mean, t_var, r_var, diag)

# todo: finish this funciton, realizing IPE in Mip-NeRF
def integrated_positional_encoding(x:torch.Tensor, freq_level:int) -> torch.Tensor:
    result = []
    encoded = torch.cat(result, dim = -1)

    return encoded

# it's the encoding of NeRF, not integrated encoding in Mip-NeRF
def positional_encoding(x:torch.Tensor, freq_level:int) -> torch.Tensor:
    result = []
    for fid in range(freq_level):
        freq = 2. ** fid
        for func in (torch.sin, torch.cos):
            result.append(func(freq * x))
    encoded = torch.cat(result, dim = -1)
    if x.dim() > 2:
        ray_num, point_num = x.shape[0], x.shape[1]
        encoded = encoded.view(ray_num, point_num, -1)
    return encoded

def linear_to_srgb(linear: torch.Tensor, eps: float = None) -> torch.Tensor:
  """From JAX multiNeRF official repo: https://github.com/google-research/multinerf"""
  if eps is None:
    eps = torch.full((1, ), torch.finfo(torch.float32).eps, device = linear.device)
  srgb0 = 323 / 25 * linear
  srgb1 = (211 * torch.maximum(eps, linear)**(5 / 12) - 11) / 200
  return torch.where(linear <= 0.0031308, srgb0, srgb1)