import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist

from argparse import ArgumentParser
from utils import seed_all, get_logger, get_modules
from datasets import datapool
from train_val_functions import valpool
from converter import Threshold_Getter,Converter
from forwards import forward_replace
from models.VGG import *

def get_args():
    parser = ArgumentParser(description='Conversion Frame')

    # Model configuration
    parser.add_argument('--model_name', default='vgg16_bn', type=str, help='Model class name')
    parser.add_argument('--load_name', '-load', type=str, help='Path to the model state_dict file')
    parser.add_argument(
        '--mode', choices=[
            'test_ann', 'get_threshold', 'test_snn', 'train_snn'
        ],
        default='test_ann',
        type=str,
        help='Mode of operation'
    )
    parser.add_argument('--sop', action='store_true', help="whether to static sop")
    parser.add_argument('--save_name', '-save', default='checkpoint', type=str, help='Name for saving the model')

    # Threshold configuration
    parser.add_argument(
        '--threshold_mode', '-thre',
        default='99.9%', type=str, help='Threshold mode'
    )
    parser.add_argument('--threshold_level', default='layer', choices=['layer', 'channel', 'neuron'], type=str, help='Threshold level')
    parser.add_argument('--fx', action='store_true', help="Whether to use fx output graph")

    # Neuron conversion configuration
    parser.add_argument(
        '--neuron_name', '-neuron',
        choices=[
            'IF', 'IF_with_neg', 'IF_diff', 'IF_line','IF_diff_line'
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
    parser.add_argument('--fuse', action='store_true', help="Whether to fuse")

    # Task configuration
    parser.add_argument('--task', choices=['classification','object_detection'], default='classification', type=str, help='Task type')

    # Dataset configuration
    parser.add_argument('--dataset', '-data', default='cifar10', type=str, help='Dataset name')
    parser.add_argument('--dataset_path', default='../data', type=str, help='Dataset path')
    parser.add_argument('--batchsize', '-b', default=25, type=int, metavar='N', help='Batch size')

    # Device configuration
    parser.add_argument('--device', '-dev', default='0', type=str, help='CUDA device ID (default: 0)')
    # Device configuration only for imagenet
    # eg.torchrun --nproc_per_node=1 main.py --logger --dataset imagenet --batchsize 64 --distributed
    parser.add_argument('--distributed', action='store_true', help="Enable distributed (default: False)")

    # Logger configuration
    parser.add_argument('--logger', action='store_true', help="Enable logging (default: False)")
    parser.add_argument('--logger_path', type=str, default="logs/log.txt", help="Path to save the log file")

    # Training and Testing configuration
    parser.add_argument('--seed', default=2024, type=int, help='Random seed for training initialization')
    parser.add_argument('--time', '-T', type=int, default=0, help='SNN simulation time')

    # YAML configuration
    parser.add_argument('--config', default='configs/config.yaml', type=str, help="Path to the YAML configuration file")

    # Parse arguments
    args = parser.parse_args()

    # Load configuration from YAML if specified
    if args.config:
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
        for key, value in config.items():
            setattr(args, key, value)

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
    model = torch.load(model_path)
    return model

def modelpool(args):
    if args.task == 'classification':
        if args.dataset == 'imagenet':
            num_classes = 1000
        elif args.dataset == 'cifar100':
            num_classes = 100
        elif args.dataset == 'cifar10':
            num_classes = 10
        else:
            print("still not support this dataset")
            exit(0)
        if args.model_name == 'vgg16_bn':
            return vgg16(num_classes=num_classes)
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
        else:
            print("still not support this model")
            exit(0)
    elif args.task == 'object_detection':
        if args.dataset == 'coco':
            num_classes = 91
        else:
            print("error dataset")
        if args.model_name == 'fcos_resnet50_fpn':
            return fcos_resnet50_fpn(num_classes=num_classes)
        elif args.model_name == 'retinanet_resnet50_fpn':
            return retinanet_resnet50_fpn(num_classes=num_classes)
        elif args.model_name == 'retinanet_resnet50_fpn_v2':
            return retinanet_resnet50_fpn_v2(num_classes=num_classes)
    else:
        assert False

def main():
    args, device = get_args()
    seed_all(args.seed)
    logger = get_logger(args.logger,args.logger_path)

    train_loader, test_loader = datapool(args)
    model = modelpool(args)

    print(model)

    get_modules(111, model)

    print("* Parameters")
    print("  mode: %s" % args.mode)

    # Perform training or testing based on args.mode
    if args.mode == 'test_ann':
        print("Test ANN Mode")
        model = load_model_from_dict(model, args.load_name, device)
        print(model)
        print("Successfully load ann state dict")
        model.to(device)
        model.eval()
        val = valpool(args)
        print(type(test_loader))
        val(model, test_loader, device, args)
    elif args.mode == 'get_threshold':
        print("Get Threshold for SNN Neuron Mode")
        model = load_model_from_dict(model, args.load_name, device)
        model.to(device)
        model.eval()
        model = Converter.change_maxpool_before_relu(model)
        model_converter = Threshold_Getter(
            dataloader=test_loader,
            mode=args.threshold_mode,
            level=args.threshold_level,
            device=device,
            momentum=0.1,
            output_fx=args.fx
        )
        model_with_threshold = model_converter(model)
        Threshold_Getter.save_model(model=model_with_threshold, model_path=args.save_name, mode_fx=args.fx)
        print("Successfully Save Model with Threshold")
    elif args.mode == 'test_snn':
        print("Test SNN Mode")
        model = Converter.change_maxpool_before_relu(model)
        model = Converter.replace_by_maxpool_neuron(model,T=args.time,step_mode=args.step_mode,coding_type=args.coding_type)
        model = Threshold_Getter.replace_nonlinear_by_hook(model=model, momentum=0.1, mode=args.threshold_mode, level=args.threshold_level)
        model = load_model_from_dict(model, args.load_name, device)
        if args.threshold_mode=="var":
            if args.neuron_name.startswith('MTH'):
                model = Threshold_Getter.get_scale_from_var(model, T=args.time*(2**args.num_thresholds))
            else:
                model = Threshold_Getter.get_scale_from_var(model, T=args.time)
        model_converter = Converter(
            neuron=args.neuron_name,
            args=args,
            T=args.time,
            step_mode=args.step_mode,fuse_flag=args.fuse
        )
        model = model_converter(model)
        model = forward_replace(args, model)
        model.to(device)
        model.eval()
        val = valpool(args)# using args.coding_type
        val(model, test_loader, device, args)
    elif args.mode == 'train_snn':
        print("Train SNN Mode")
    else:
        assert False
    if args.distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
