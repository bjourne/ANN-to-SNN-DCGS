import torch
import torch.nn as nn
from torch import fx
from torch import nn
from tqdm import tqdm
from typing import Tuple
import numpy as np
import os
import utils
from torch.nn import Module

def set_voltagehook_under_graph(fx_model, mode='Max', momentum=0.1):
    hook_cnt = -1
    for node in fx_model.graph.nodes:
        if node.op != 'call_module':
            continue
        if type(fx_model.get_submodule(node.target)) is nn.ReLU:
            hook_cnt += 1
            target = 'snn tailor.' + str(hook_cnt) + '.0'  # voltage_hook
            m = ThreHook(momentum=momentum, mode=mode)
            new_node = Threshold_Getter._add_module_and_node(fx_model, target, node, m, (node,))
    fx_model.graph.lint()
    fx_model.recompile()
    return fx_model

class Threshold_Getter(Module):
    def __init__(
        self,
        loader,
        mode='99.9',
        level = 'layer',
        device=None,
        momentum=0.1,
        output_fx=False
    ):
        super().__init__()
        self.loader = loader
        self.mode = mode
        self.level = level
        self.device = device
        self.momentum = momentum
        self.output_fx = output_fx

    def forward(self, model):
        assert self.device
        if self.output_fx:
            model = fx.symbolic_trace(model).to(self.device)
            model.eval()
            model_with_hook = set_voltagehook_under_graph(
                model, mode=self.mode, momentum=self.momentum
            ).to(self.device)
        else:
            model.eval()
            model_with_hook = replace_nonlinear_by_hook(
                model,
                self.momentum,
                self.mode,
                self.level
            ).to(self.device)
        for _, (xs, _) in enumerate(tqdm(self.loader)):
            xs = xs.to(self.device)
            model_with_hook(xs)
        return model_with_hook

    @staticmethod
    def get_scale_from_var(model, T = 64):
        for name, module in model._modules.items():
            cname = module.__class__.__name__.lower()
            if cname == "threhook":
                model._modules[name].get_scale_from_var(T=T)
            elif hasattr(module, "_modules"):
                Threshold_Getter.get_scale_from_var(module, T=T)
        return model

    @staticmethod
    def _add_module_and_node(
            fx_model: fx.GraphModule,
            target: str,
            after: fx.Node,
            m: nn.Module,
            args: Tuple
    ) -> fx.Node:
        fx_model.add_submodule(target=target, m=m)
        with fx_model.graph.inserting_after(n=after):
            new_node = fx_model.graph.call_module(module_name=target, args=args)
        return new_node

    @staticmethod
    def load_model(model_path, mode_fx, code_path=None,model=None):
        if mode_fx:
            return Threshold_Getter.load_fx_model(model_path+'.pt')
        else:
            if model is None:
                print("Must give a model to load state dict")
                exit(0)
            return Threshold_Getter.load_module_model(model=model, model_path=model_path+'.pth')

    @staticmethod
    def load_module_model(model, model_path):
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        return model

    @staticmethod
    def save_fx_model(fx_model, model_path, code_path=None):
        model_dir = os.path.dirname(model_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)  # Create the directory if it doesn't exist
        if code_path is not None:
            with open(code_path, "w") as f:
                f.write(fx_model.code)
        torch.save(fx_model, model_path)

    @staticmethod
    def load_fx_model(model_path, code_path=None):
        code = None
        model = torch.load(model_path)
        if code_path is not None:
            with open(code_path, "r") as f:
                code = f.read()
            return model, code
        return model

def replace_nonlinear_by_hook(model, momentum, mode, level):
    cnames = [
        "relu",
        'gelu',
        'silu',
        'layernorm','groupnorm','softmax','myat',
        'linear',
        'conv2d'
    ]
    for name, module in model._modules.items():
        cname = module.__class__.__name__.lower()
        if hasattr(module, "_modules"):
            replace_nonlinear_by_hook(module, momentum, mode, level)
        if cname in cnames:
            model._modules[name] = ThreHook(
                mode=mode,
                momentum=momentum,
                out_layer=module,
                level=level
            )
    return model


def save_model(model, model_path, mode_fx):
    if mode_fx:
        return Threshold_Getter.save_fx_model(model, model_path+'.pt')
    else:
        torch.save(model.state_dict(), "%s.pth" % model_path)

def merge_dims(tensor, dims):
    dims = sorted(dims)
    shape = list(tensor.shape)
    merged_size = 1
    for dim in dims:
        merged_size *= shape[dim]

    # 构建新形状
    new_shape = [merged_size] + [shape[i] for i in range(len(shape)) if i not in dims]
    # 调整顺序并 reshape
    # print(tensor.shape)
    permuted_tensor = tensor.permute(*dims, *[i for i in range(tensor.ndim) if i not in dims]).contiguous()
    # print(permuted_tensor.shape)
    return permuted_tensor.view(*new_shape).contiguous(), shape, dims

def restore_dims(percentile_tensor, original_shape, dims):
    """
    根据原始形状和 dims，将百分位数结果还原，合并的维度大小变为 1。
    """
    dims = sorted(dims)
    new_shape = [1 if i in dims else original_shape[i] for i in range(len(original_shape))]
    return percentile_tensor.view(*new_shape).contiguous()

def calculate_better(mean, var, scale, n=64):
    device = mean.device
    dtype = mean.dtype

    sigma = torch.sqrt(var)
    theta = scale
    pi = torch.tensor(torch.pi, dtype=dtype, device=device)  # π 值

    # 生成 i 的序列 (1, 2, ..., n)
    i = torch.arange(1, n + 1, dtype=dtype, device=device).view(*([1] * mean.dim()), n)
    # 例如，如果 mean 的形状是 (batch_size, channels, height, width)，则 i 的形状将是 (1, 1, 1, 1, n)

    # 计算 coeff，确保维度匹配
    coeff = ((2 * i - 1) / (2 * n)) * theta.unsqueeze(-1)  # 在最后一个维度添加 n 的维度

    # 计算 (coeff - mean) / (sqrt(2) * sigma)
    sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))
    erf_input = (coeff - mean.unsqueeze(-1)) / (sqrt2 * sigma.unsqueeze(-1))
    erf_term = torch.erf(erf_input)

    term1_sum = torch.sum((1 / n) * erf_term, dim=-1)
    term2_sum = torch.sum(((2 * i - 1) / n**2) * erf_term, dim=-1)

    # 计算 exp 项
    exp_input = -((coeff - mean.unsqueeze(-1))**2) / (2 * var.unsqueeze(-1))
    exp_term = torch.exp(exp_input)
    term3_sum = torch.sum((1 / n) * exp_term, dim=-1)

    # 分母
    denominator = 1 - term2_sum

    # 计算部分1和部分2
    part1 = (mean / theta) * (1 - term1_sum) / denominator
    sqrt_pi_over_2 = torch.sqrt(pi / 2)
    part2 = (sigma / (sqrt_pi_over_2 * theta)) * term3_sum / denominator

    # 合并得到 k1
    k1 = part1 + part2
    return k1

class ThreHook(nn.Module):
    def __init__(
        self,
        scale=1.0,
        mode='Max',
        momentum=0.1,
        out_layer='Identitiy',
        level='layer'
    ):
        super().__init__()
        self.register_buffer('scale', None)
        self.register_buffer('scale2', None)
        self.register_buffer('var', None)
        self.register_buffer('mean', None)
        self.mode = mode
        self.register_buffer('num_batches_tracked', torch.tensor(0))
        self.momentum = momentum
        self.out = out_layer

        self.level=level

    def get_scale_from_var(self, T=64):
        self.scale = torch.ones_like(self.mean)
        tmp = self.scale.clone().detach()
        for idx in range(100):
            self.scale = self.scale*calculate_better(
                self.mean, self.var, self.scale, n=T
            )
        nan_mask = torch.isnan(self.scale)
        if nan_mask.any():
            print(f"Found {nan_mask.sum().item()} NaN values, replacing with original thresholds")
            tmp_values = tmp[nan_mask].to(self.scale.dtype)
            self.scale[nan_mask] = tmp_values
        else:
            print(f"Found 0 NaN values, replacing with original thresholds")


    def forward(self, x, *args):
        assert len(args) == 0
        if self.level == 'layer':
            dims = [i for i in range(x.ndim)]
        elif self.level == 'channel':
            dims = [i for i in range(x.ndim) if i != 1]
        elif self.level=='neuron':
            dims = [0]
        else:
            assert False
        err_msg = 'You have used a non-defined Method to get Threshold.'

        out_cname = self.out.__class__.__name__.lower()
        if out_cname == 'myat':
            if self.mode[-1] == '%':
                percentile_val = float(self.mode[:-1])
                assert 0 < percentile_val < 100

                merged_x, original_shape, dims = merge_dims(x.clone().detach(), dims)
                s_t = torch.quantile(merged_x, percentile_val / 100.0, dim=0, keepdim=True).detach()
                s_t = restore_dims(s_t, original_shape, dims)

                merged_x2, original_shape2, dims2 = merge_dims(args[0].clone().detach(), dims)
                s_t2 = torch.quantile(merged_x2, percentile_val / 100.0, dim=0, keepdim=True).detach()
                s_t2 = restore_dims(s_t2, original_shape2, dims2)

                if self.scale is None:
                    self.scale = s_t.clone()
                    self.scale2 = s_t2.clone()
                else:
                    self.momentum = x.shape[0]/(self.num_batches_tracked+x.shape[0])
                    self.scale = (1 - self.momentum) * self.scale + self.momentum * s_t
                    self.scale2 = (1 - self.momentum) * self.scale2 + self.momentum * s_t2
                self.num_batches_tracked += x.shape[0]
            elif self.mode.lower() in ['max']:
                s_t = x.clone().detach().amax(dim=dims, keepdim=True).detach()
                s_t2 = args[0].clone().detach().amax(dim=dims, keepdim=True).detach()
                if self.scale is None:
                    self.scale = s_t.clone()
                    self.scale2 = s_t2.clone()
                else:
                    self.momentum = x.shape[0]/(self.num_batches_tracked+x.shape[0])
                    self.scale = (1 - self.momentum) * self.scale + self.momentum * s_t
                    self.scale2 = (1 - self.momentum) * self.scale2 + self.momentum * s_t2
                self.num_batches_tracked += x.shape[0]
            else:
                raise NotImplementedError(err_msg)
            x = self.out(x,args[0])
            return x
        else:
            if self.mode.lower() in ['var']:
                mean_t = x.mean(dim=dims, keepdim=True).detach()
                var_t = x.var(dim=dims, keepdim=True, unbiased=False).detach()
                if self.mean is None:
                    self.mean = mean_t.clone()
                    self.var = var_t.clone()
                else:
                    delta = mean_t - self.mean
                    self.momentum = x.shape[0]/(self.num_batches_tracked+x.shape[0])
                    self.mean = (1 - self.momentum) * self.mean + self.momentum * mean_t
                    self.var = (1 - self.momentum) * self.var + self.momentum * (var_t + (1-self.momentum) * delta * delta)
                self.num_batches_tracked += x.shape[0]

            if out_cname not in ['linear','conv2d']:
                print("OUT", self.out)
                x = self.out(x)
            if self.mode[-1] == '%':
                percentile_val = float(self.mode[:-1])
                assert 0 < percentile_val < 100

                merged_x, original_shape, dims = merge_dims(x.clone().detach(), dims)
                s_t = torch.quantile(
                    merged_x,
                    percentile_val / 100.0,
                    dim=0,
                    keepdim=True
                ).detach()
                s_t = restore_dims(s_t, original_shape, dims)
                if self.scale is None:
                    self.scale = s_t.clone()
                else:
                    self.momentum = x.shape[0]/(self.num_batches_tracked+x.shape[0])
                    self.scale = (1 - self.momentum) * self.scale + self.momentum * s_t
                self.num_batches_tracked += x.shape[0]
            elif self.mode.lower() in ['max']:
                s_t = x.clone().detach().amax(dim=dims, keepdim=True).detach()
                if self.scale is None:
                    self.scale = s_t.clone()
                else:
                    self.momentum = x.shape[0]/(self.num_batches_tracked+x.shape[0])
                    self.scale = (1 - self.momentum) * self.scale + self.momentum * s_t
                self.num_batches_tracked += x.shape[0]
            else:
                if self.mode.lower() not in ['var']:
                    raise NotImplementedError(err_msg)

            if out_cname in ['linear','conv2d']:
                x = self.out(x)
            return x

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs
    ):
        for key in state_dict:
            if key.endswith('.scale'):
                saved_scale = state_dict[key]
                if self.scale is None or saved_scale.shape != self.scale.shape:
                    self.scale = saved_scale.new_zeros(*saved_scale.shape)
            if key.endswith('.scale2'):
                saved_scale2 = state_dict[key]
                if self.scale2 is None or saved_scale2.shape != self.scale2.shape:
                    self.scale2 = saved_scale2.new_zeros(*saved_scale2.shape)
            if key.endswith('.var'):
                saved_var = state_dict[key]
                if self.var is None or saved_var.shape != self.var.shape:
                    self.var = saved_var.new_zeros(*saved_var.shape)
            if key.endswith('.mean'):
                saved_mean = state_dict[key]
                if self.mean is None or saved_mean.shape != self.mean.shape:
                    self.mean = saved_mean.new_zeros(*saved_mean.shape)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs
        )
