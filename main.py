# I think we only need step_mode=="s"

import os
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist

from argparse import ArgumentParser
from converter import Converter, Threshold_Getter
from converter.threshold_getter import replace_nonlinear_by_hook, save_model
from copy import deepcopy
from datasets.getdataloader import GetCifar10, GetCifar100
from forwards import add_dimention
from models.VGG import *
from neurons.nonlinear_neuron import maxpool_neuron
from torch.nn import (
    BatchNorm2d, Conv2d, Linear,
    Flatten,
    Identity,
    MaxPool2d,
    ReLU,
    Sequential
)
from torch.nn.init import (
    constant_, kaiming_normal_, normal_, uniform_, zeros_
)
from torchinfo import summary
from types import MethodType
from utils import (
    MergeTemporalDim, ExpandTemporalDim,
    seed_all, get_logger, get_modules
)

def replace_modules(mod, match_fun, new_fun):
    for name, submod in mod.named_children():
        if match_fun(submod):
            setattr(mod, name, new_fun(submod))
        replace_modules(submod, match_fun, new_fun)

def change_maxpool_before_relu(model):
    for name, module in model._modules.items():
        cname = module.__class__.__name__.lower()
        if hasattr(module, "_modules"):
            change_maxpool_before_relu(module)
        if 'relu' in cname or ("threhook" in cname and module.out.__class__.__name__.lower()=='relu'):
            tmp_name = name
        if 'maxpool' in cname:
            tmp = model._modules[tmp_name]
            model._modules[tmp_name] = Identity()
            model._modules[name] = Sequential(model._modules[name],tmp)
            print("change a maxpool before relu")
    #return model

def replace_by_maxpool_neuron(
    model,
    T,
    step_mode,
    coding_type
):
    replace_modules(
        model,
        lambda m: isinstance(m, MaxPool2d),
        lambda m: maxpool_neuron(m, T, step_mode, coding_type)
    )


def datapool(args):
    if args.dataset == 'cifar10':
        return GetCifar10(args)
    elif args.dataset == 'cifar100':
        return GetCifar100(args)
    elif args.dataset == 'imagenet':
        return GetImageNet(args)
    elif args.dataset == 'coco':
        return GetCOCO(args)
    assert False

def reset(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = reset(module)
        if hasattr(module, "reset"):
            model._modules[name].reset()
    return model

def val_snn_classfication(model, l_te, device, args=None):
    correct = 0
    total = 0
    model.eval()
    all_correct = [0 for i in range(model.T)]
    all_total = [0 for i in range(model.T)]
    with torch.no_grad():
        for bi, (xs, targets) in enumerate(l_te):
            reset(model)
            xs = xs.to(device)
            yhats = model(xs)
            for i in range(model.T):
                yhats_T = yhats[:i+1].mean(0)
                _, predicted = yhats_T.cpu().max(1)
                all_correct[i] += float(predicted.eq(targets).sum().item())
                all_total[i] += float(targets.size(0))

            print(bi, 100 * all_correct[-1] / all_total[-1])

            per_step = ["%4.1f" % (100 * all_correct[i] / all_total[i])
                        for i in range(model.T)]
            print(" ".join(per_step))
        final_acc = 100 * all_correct[-1] / all_total[-1]
    return final_acc


def val_ann_classfication(model, l_te, device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for bi, (xs, targets) in enumerate(l_te):
            xs = xs.to(device)
            yhats = model(xs)
            _, predicted = yhats.cpu().max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())
            print(bi, 100 * correct / total)
        final_acc = 100 * correct / total
    return final_acc

def get_args():
    parser = ArgumentParser(description='Conversion Frame')

    # Model configuration
    parser.add_argument(
        '--model_name', default='vgg16_bn',
        type=str, help='Model class name'
    )
    parser.add_argument(
        '--load_name', '-load',
        type=str, help='Path to the model state_dict file'
    )
    parser.add_argument(
        '--mode', choices=[
            'test_ann',
            'get_threshold',
            'test_snn',
            'train_snn'
        ],
        default='test_ann',
        type=str,
        help='Mode of operation'
    )
    parser.add_argument(
        '--sop',
        action='store_true',
        help="whether to static sop"
    )
    parser.add_argument(
        '--save_name', '-save',
        default='checkpoint', type=str, help='Name for saving the model'
    )

    # Threshold configuration
    parser.add_argument(
        '--threshold_mode', '-thre',
        default='99.9%', type=str, help='Threshold mode'
    )
    parser.add_argument(
        '--threshold_level',
        default='layer',
        choices = [
            'layer', 'channel', 'neuron'
        ],
        type=str, help='Threshold level'
    )
    parser.add_argument(
        '--fx',
        action='store_true',
        help="Whether to use fx output graph"
    )

    # Neuron conversion configuration
    parser.add_argument(
        '--neuron_name', '-neuron',
        choices=[
            'IF', 'IF_with_neg', 'IF_diff', 'IF_line','IF_diff_line',
            'LIF', 'LIF_with_neg', 'LIF_diff',
            'MTH', 'MTH_with_neg', 'MTH_diff', 'MTH_line','MTH_diff_line'
        ],
        default='IF', type=str, help='Neuron model name'
    )
    parser.add_argument('--tau', default=0.98, type=float, help='Parameter tau')
    parser.add_argument('--num_thresholds', default=8, type=int, help='num_thresholds')
    parser.add_argument(
        '--step_mode',
        choices=['s', 'm'],
        default='s',
        type=str, help='Step_mode'
    )
    parser.add_argument(
        '--coding_type', '-coding',
        choices=[
            'rate', 'leaky_rate', 'diff_rate', 'diff_leaky_rate'
        ], default='rate',
        type=str,
        help='Coding type'
    )
    parser.add_argument(
        '--fuse', action='store_true', help="Whether to fuse"
    )

    # Dataset configuration
    parser.add_argument(
        '--dataset', '-data', default='cifar10',
        type=str, help='Dataset name'
    )
    parser.add_argument(
        '--dataset_path', default='../data', type=str, help='Dataset path'
    )
    parser.add_argument(
        '--batchsize', '-b', default=25,
        type=int, metavar='N', help='Batch size'
    )

    # Device configuration
    parser.add_argument(
        '--device',
        '-dev',
        default='0',
        type=str,
        help='CUDA device ID (default: 0)'
    )

    # Device configuration only for imagenet
    # eg.torchrun --nproc_per_node=1 main.py --logger --dataset imagenet --batchsize 64 --distributed
    parser.add_argument(
        '--distributed', action='store_true', help="Enable distributed (default: False)"
    )

    # Training and Testing configuration
    parser.add_argument('--seed', default=2024, type=int, help='Random seed for training initialization')
    parser.add_argument('--time', '-T', type=int, default=0, help='SNN simulation time')

    # YAML configuration
    parser.add_argument(
        '--config', default='configs/config.yaml',
        type=str,
        help="Path to the YAML configuration file"
    )

    # Parse arguments
    args = parser.parse_args()

    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    if args.distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", init_method="env://")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0

    return args, device

def load_model_from_dict(model, model_path, device):
    state_dict = torch.load(
        os.path.join(model_path),
        map_location=torch.device('cpu')
    )
    for model_key in ['model','module']:
        if model_key in state_dict:
            state_dict = state_dict[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            break

    model.load_state_dict(state_dict)
    model.to(device)
    return model

def load_model_from_model(model, model_path, device):
    return torch.load(model_path)

# Build the VGG16 layers
def build_vgg_layers(n_cls):
    vgg16_layers = [
        64, 64, "M",
        128, 128, "M",
        256, 256, 256, "M",
        512, 512, 512, "M",
        512, 512, 512, "M",
    ]
    n_chans_in = 3
    for v in vgg16_layers:
        if type(v) == int:
            # Batch norm so no bias.
            yield Conv2d(n_chans_in, v, 3, padding=1, bias=False)
            yield BatchNorm2d(v)
            yield ReLU(True)
            n_chans_in = v
        elif v == "M":
            yield MaxPool2d(2)
        else:
            assert False
    yield Flatten()
    yield Linear(512, 4096)
    yield ReLU(inplace = True)
    yield Linear(4096, 4096)
    yield ReLU(inplace = True)
    yield Linear(4096, n_cls)

def my_vgg16(n_cls):
    net = Sequential(*build_vgg_layers(n_cls))
    for m in net.modules():
        if isinstance(m, Conv2d):
            kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                constant_(m.bias, 0)
        elif isinstance(m, BatchNorm2d):
            constant_(m.weight, 1)
            constant_(m.bias, 0)
        elif isinstance(m, Linear):
            normal_(m.weight, 0, 0.01)
            constant_(m.bias, 0)
    return net

def modelpool(args):
    if args.dataset == 'imagenet':
        num_classes = 1000
    elif args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset == 'cifar10':
        num_classes = 10
    else:
        assert False

    ctors = {
        "vgg16_bn" : vgg16,
        "my_vgg16" : my_vgg16
    }
    return ctors[args.model_name](num_classes)

    if n == 'vgg16_bn':
        return vgg16(num_classes=num_classes)
    elif n == "my_vgg16":
        return my_vgg16(num_classes)
    elif args.model_name == 'vgg16_wobn':
        return vgg16_wobn(num_classes=num_classes)
    elif args.model_name == 'vgg19_bn':
        return vgg19(num_classes=num_classes)
    elif args.model_name == 'resnet18':
        return resnet18(num_classes=num_classes)
    elif args.model_name == 'resnet20':
        return resnet20(num_classes=num_classes)
    elif args.model_name == 'resnet34':
        return resnet34(num_classes=num_classes)
    elif args.model_name == 'resnet50':
        return resnet34(num_classes=num_classes)
    elif args.model_name == 'resnet152':
        return resnet34(num_classes=num_classes)
    elif args.model_name == 'vit_small':
        return vit_small_patch16_224(num_classes=num_classes)
    elif args.model_name == 'vit_base':
        return vit_base_patch16_224(num_classes=num_classes)
    elif args.model_name == 'vit_large':
        return vit_large_patch16_224(num_classes=num_classes)
    elif args.model_name == 'eva02_tiny':
        return eva02_tiny_patch14_336(num_classes=num_classes)
    elif args.model_name == 'eva02_small':
        return eva02_small_patch14_336(num_classes=num_classes)
    elif args.model_name == 'eva02_base':
        return eva02_base_patch14_448(num_classes=num_classes)
    elif args.model_name == 'eva02_large':
        return eva02_large_patch14_448(num_classes=num_classes)
    assert False


def decodeoutput(x):
    out = torch.zeros_like(x[1:])
    T = x.shape[0]-1
    exp_in = x[0].clone().detach()
    for t in range(T):
        out[t]= exp_in + x[t+1] - x[0]
        exp_in = exp_in + (x[t+1] - x[0])/(t+1)
    return out

def forward_snn_rate_s(net, x):
    output = []
    for i in range(net.T):
        tmp = net.init_forward(deepcopy(x))
        output.append(deepcopy(tmp))
    return torch.stack(output, dim=0)

def forward_snn_diff_rate_s(net, x):
    output = []
    tmp = net.init_forward(torch.zeros_like(x))
    output.append(deepcopy(tmp))
    tmp = net.init_forward(x)
    output.append(deepcopy(tmp))
    for i in range(net.T-1):
        tmp = net.init_forward(torch.zeros_like(x))
        output.append(deepcopy(tmp))
    return decodeoutput(torch.stack(output, dim=0))

def forward_snn_rate_m(self, x):
    x = add_dimention(x, self.T)
    x = self.merge(x)
    out = self.init_forward(x)
    out = self.expand(out)
    return out

def forward_replace(args, model):
    model.coding_type = args.coding_type
    model.step_mode = args.step_mode
    model.T = args.time

    if args.coding_type=='rate':
        if args.step_mode=='s':
            model.init_forward = model.forward
            model.forward = MethodType(forward_snn_rate_s, model)
        elif args.step_mode=='m':
            model.merge = MergeTemporalDim()
            model.expand = ExpandTemporalDim(model.T)
            model.init_forward = model.forward
            model.forward = MethodType(forward_snn_rate_m, model)
        else:
            assert False
        return model
    elif args.coding_type=='leaky_rate':
        if args.step_mode=='s':
            model.tau = args.tau
            model.init_forward = model.forward
            model.forward = MethodType(forward_snn_leaky_rate_s, model)
        elif args.step_mode=='m':
            model.tau = args.tau
            model.merge = MergeTemporalDim()
            model.expand = ExpandTemporalDim(model.T)
            model.init_forward = model.forward
            model.forward = MethodType(forward_snn_leaky_rate_m, model)
        else:
            print("Unexpected step mode")
        return model
    elif args.coding_type=='diff_rate':
        if args.step_mode=='s':
            model.init_forward = model.forward
            model.forward = MethodType(forward_snn_diff_rate_s, model)
        elif args.step_mode=='m':
            model.merge = MergeTemporalDim()
            model.expand = ExpandTemporalDim(model.T+1)
            model.init_forward = model.forward
            model.forward = MethodType(forward_snn_diff_rate_m, model)
        else:
            print("Unexpected step mode")
        return model
    elif args.coding_type=='diff_leaky_rate':
        if args.step_mode=='s':
            model.tau = args.tau
            model.init_forward = model.forward
            model.forward = MethodType(forward_snn_diff_leaky_rate_s, model)
        elif args.step_mode=='m':
            model.tau = args.tau
            model.merge = MergeTemporalDim()
            model.expand = ExpandTemporalDim(model.T)
            model.init_forward = model.forward
            model.forward = MethodType(forward_snn_diff_leaky_rate_m, model)
        else:
            print("Unexpected step mode")
        return model
    else:
        assert False

def main():
    args, device = get_args()
    seed_all(args.seed)

    train_loader, l_te = datapool(args)
    model = modelpool(args)

    keys = [
        "fuse", "fx",
        "coding_type",
        "mode", "neuron_name",
        "save_name", "sop", "step_mode",
        "threshold_level", "threshold_mode", "time"
    ]
    pad = max(len(k) for k in keys)
    print("* Parameters")
    fmt = "  %%-%ds: %%s" % pad
    for key in keys:
        print(fmt % (key, getattr(args, key)))

    # Perform training or testing based on args.mode
    if args.mode == 'test_ann':
        model = load_model_from_dict(model, args.load_name, device)
        print(model)
        print("Successfully load ann state dict")
        model.to(device)
        model.eval()
        val_ann_classfication(model, l_te, device)
    elif args.mode == 'get_threshold':
        model = load_model_from_dict(model, args.load_name, device)
        model.to(device)
        model.eval()
        print(model)
        model = change_maxpool_before_relu(model)
        print(model)

        model_converter = Threshold_Getter(
            l_te,
            mode=args.threshold_mode,
            level=args.threshold_level,
            device=device,
            momentum=0.1,
            output_fx=args.fx
        )
        model_with_threshold = model_converter(model)
        save_model(model_with_threshold, args.save_name, args.fx)
        print("Successfully Save Model with Threshold")
    elif args.mode == 'test_snn':
        assert args.time > 0
        change_maxpool_before_relu(model)
        replace_by_maxpool_neuron(
            model,
            args.time,
            args.step_mode,
            args.coding_type
        )
        model = replace_nonlinear_by_hook(
            model, 0.1, args.threshold_mode, args.threshold_level
        )
        model = load_model_from_dict(model, args.load_name, device)
        if args.threshold_mode=="var":
            if args.neuron_name.startswith('MTH'):
                model = Threshold_Getter.get_scale_from_var(model, T=args.time*(2**args.num_thresholds))
            else:
                model = Threshold_Getter.get_scale_from_var(model, T=args.time)
        model_converter = Converter(
            args.neuron_name,
            args,
            args.time,
            args.step_mode,
            args.fuse
        )
        model = model_converter(model)
        model = forward_replace(args, model)
        model.to(device)
        model.eval()

        if args.sop:
            val = val_snn_classfication_with_sop
        else:
            val = val_snn_classfication
        val(model, l_te, device, args)
    else:
        assert False
    if args.distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
