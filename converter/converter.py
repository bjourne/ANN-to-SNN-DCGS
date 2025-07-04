import torch
from torch import fx
from torch import nn
from tqdm import tqdm
from typing import Tuple
import numpy as np
import os


from neurons import *
from torch.nn import *
from utils import MyatSequential


def replace_relu_by_LIF(model,step_mode,T,tau,neuron):
    for name, module in model._modules.items():
        if module.__class__.__name__.lower()=="relu":
            model._modules[name] = neuron(T=T, thresh=1.0, tau = tau, step_mode=step_mode)
        elif "threhook" in module.__class__.__name__.lower() and module.out.__class__.__name__.lower()=='relu':
            thre = model._modules[name].scale
            thre = thre*(thre>=0).float()
            model._modules[name] = neuron(T=T, thresh=thre, tau = tau, step_mode=step_mode)
        elif hasattr(module, "_modules"):
            model._modules[name] = replace_relu_by_LIF(module,step_mode,T,tau,neuron)
    return model

def replace_relu_by_MTH(model,step_mode,T,neuron,num_thresholds,args):
    for name, module in model._modules.items():
        cname = module.__class__.__name__.lower()
        if cname == "relu":
            model._modules[name] = neuron(T=T, thresh=1.0, step_mode=step_mode,num_thresholds=num_thresholds)
        elif args.model_name in ['fcos_resnet50_fpn', 'retinanet_resnet50_fpn', 'retinanet_resnet50_fpn_v2']:
            if "threhook" in module.__class__.__name__.lower() and module.out.__class__.__name__.lower() in ['linear','conv2d']:
                model._modules[name] = module.out
            elif "threhook" in module.__class__.__name__.lower() and module.out.__class__.__name__.lower()=='relu':
                thre = model._modules[name].scale
                thre = thre*(thre>=0).float()
                model._modules[name] = neuron(T=T, thresh=thre, step_mode=step_mode,num_thresholds=num_thresholds)
            elif "threhook" in module.__class__.__name__.lower():
                model._modules[name] = exp_comp_neuron(module.out,T=args.time,step_mode=args.step_mode,coding_type=args.coding_type)
            elif hasattr(module, "_modules"):
                model._modules[name] = Converter.replace_relu_by_MTH(module,step_mode,T,neuron,num_thresholds,args)
        elif args.model_name in ["my_vgg16", 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'resnet18', 'resnet20', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
            if "threhook" in module.__class__.__name__.lower() and module.out.__class__.__name__.lower()=='relu':
                thre = model._modules[name].scale
                thre = thre*(thre>=0).float()
                model._modules[name] = neuron(T=T, thresh=thre, step_mode=step_mode,num_thresholds=num_thresholds)
            elif "threhook" in module.__class__.__name__.lower() and module.out.__class__.__name__.lower() in ['linear','conv2d']:
                model._modules[name] = module.out
            elif hasattr(module, "_modules"):
                model._modules[name] = Converter.replace_relu_by_MTH(module,step_mode,T,neuron,num_thresholds,args)
        else:
            if "threhook" in module.__class__.__name__.lower():
                scale = 8
                thre = model._modules[name].scale
                thre = thre*(thre>=0).float()*scale
                if module.out.__class__.__name__.lower()=='myat':
                    thre2 = model._modules[name].scale2
                    thre2 = thre2*(thre2>=0).float()*scale
                    model._modules[name] = MyatSequential(neuron(T=T, thresh=thre, step_mode=step_mode,num_thresholds=num_thresholds),
                                                          neuron(T=T, thresh=thre2, step_mode=step_mode,num_thresholds=num_thresholds),
                                                          AtNeuron(T=args.time,step_mode=args.step_mode,coding_type=args.coding_type))
                elif module.out.__class__.__name__.lower() in ['linear','conv2d']:
                    model._modules[name] = nn.Sequential(neuron(T=T, thresh=thre, step_mode=step_mode,num_thresholds=num_thresholds),
                                                         module.out)
                else:
                    model._modules[name] = exp_comp_neuron(module.out,T=args.time,step_mode=args.step_mode,coding_type=args.coding_type)
            elif module.__class__.__name__.lower() == 'mymul':
                model._modules[name] = MulNeuron(T=args.time,step_mode=args.step_mode,coding_type=args.coding_type)
            elif hasattr(module, "_modules"):
                model._modules[name] = replace_relu_by_MTH(module,step_mode,T,neuron,num_thresholds,args)
    return model

class Converter:
    def __init__(self, neuron, args, T=0, step_mode='s', fuse_flag=False):
        self.neuron = neuron
        self.fuse_flag = fuse_flag
        self.step_mode = step_mode
        self.T=T
        self.args=args

    @staticmethod
    def fuse(
        fx_model: torch.fx.GraphModule,
        fuse_flag: bool = True
    ) -> torch.fx.GraphModule:
        print("FUSE")
        def matches_module_pattern(pattern: Iterable[Type], node: fx.Node, modules: Dict[str, Any]) -> bool:
            if len(node.args) == 0:
                return False
            nodes: Tuple[Any, fx.Node] = (node.args[0], node)
            for expected_type, current_node in zip(pattern, nodes):
                if not isinstance(current_node, fx.Node):
                    return False
                if current_node.op != 'call_module':
                    return False
                if not isinstance(current_node.target, str):
                    return False
                if current_node.target not in modules:
                    return False
                if type(modules[current_node.target]) is not expected_type:
                    return False
            return True

        def replace_node_module(node: fx.Node, modules: Dict[str, Any],
                                new_module: torch.nn.Module):
            def parent_name(target: str) -> Tuple[str, str]:
                """
                Splits a qualname into parent path and last atom.
                For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
                """
                *parent, name = target.rsplit('.', 1)
                return parent[0] if parent else '', name

            assert isinstance(node.target, str)
            parent_name, name = parent_name(node.target)
            modules[node.target] = new_module
            setattr(modules[parent_name], name, new_module)

        if not fuse_flag:
            return fx_model
        patterns = [
            (Conv1d, BatchNorm1d),
            (Conv2d, BatchNorm2d),
            (Conv3d, BatchNorm3d)
        ]

        modules = dict(fx_model.named_modules())

        for pattern in patterns:
            for node in fx_model.graph.nodes:
                if matches_module_pattern(pattern, node,
                                          modules):
                    if len(node.args[0].users) > 1:  # Output of conv is used by other nodes
                        continue
                    conv = modules[node.args[0].target]
                    bn = modules[node.target]
                    fused_conv = fuse_conv_bn_eval(conv, bn)
                    replace_node_module(node.args[0], modules,
                                        fused_conv)
                    node.replace_all_uses_with(node.args[0])
                    fx_model.graph.erase_node(node)
        fx_model.graph.lint()
        fx_model.delete_all_unused_submodules()  # remove unused bn modules
        fx_model.recompile()
        return fx_model


    @staticmethod
    def replace_by_ifnode(fx_model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        hook_cnt = -1
        for node in fx_model.graph.nodes:
            if node.op != 'call_module':
                continue
            if type(fx_model.get_submodule(node.target)) is VoltageHook:
                if type(fx_model.get_submodule(node.args[0].target)) is nn.ReLU:
                    hook_cnt += 1
                    hook_node = node
                    relu_node = node.args[0]
                    if len(relu_node.args) != 1:
                        raise NotImplementedError('The number of relu_node.args should be 1.')
                    s = fx_model.get_submodule(node.target).scale.item()
                    target0 = 'snn tailor.' + str(hook_cnt) + '.0'  # voltage_scaler
                    target1 = 'snn tailor.' + str(hook_cnt) + '.1'  # IF_node
                    target2 = 'snn tailor.' + str(hook_cnt) + '.2'  # voltage_scaler
                    m0 = VoltageScaler(1.0 / s)
                    m1 = neuron.IFNode(v_threshold=1., v_reset=None)
                    m2 = VoltageScaler(s)
                    node0 = Converter._add_module_and_node(fx_model, target0, hook_node, m0,
                                                           relu_node.args)
                    node1 = Converter._add_module_and_node(fx_model, target1, node0, m1
                                                           , (node0,))
                    node2 = Converter._add_module_and_node(fx_model, target2, node1, m2, args=(node1,))

                    relu_node.replace_all_uses_with(node2)
                    node2.args = (node1,)
                    fx_model.graph.erase_node(hook_node)
                    fx_model.graph.erase_node(relu_node)
                    fx_model.delete_all_unused_submodules()
        fx_model.graph.lint()
        fx_model.recompile()
        return fx_model

    @staticmethod
    def _add_module_and_node(fx_model: fx.GraphModule, target: str, after: fx.Node, m: nn.Module,
                             args: Tuple) -> fx.Node:
        fx_model.add_submodule(target=target, m=m)
        with fx_model.graph.inserting_after(n=after):
            new_node = fx_model.graph.call_module(module_name=target, args=args)
        return new_node
