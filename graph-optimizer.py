

import torch
import torch.nn as nn
import torchvision.models as models
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def fuse_conv_bn_relu(conv, bn, relu=None):
    """Fuse Conv2d + BatchNorm2d (+ optional ReLU) into a single Conv2d"""
    fused_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True
    )
    
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight / torch.sqrt(bn.running_var + bn.eps))
    fused_conv.weight.data = torch.mm(w_bn, w_conv).view(fused_conv.weight.size())
  
    if conv.bias is not None:
        b_conv = conv.bias
    else:
        b_conv = torch.zeros(conv.out_channels)
    b_bn = bn.bias - bn.weight * bn.running_mean / torch.sqrt(bn.running_var + bn.eps)
    fused_conv.bias.data = torch.mv(w_bn, b_conv) + b_bn

    return fused_conv

def remove_identity_layers(model):
    """Remove nn.Identity layers from Sequential modules"""
    if isinstance(model, nn.Sequential):
        new_layers = [layer for layer in model if not isinstance(layer, nn.Identity)]
        return nn.Sequential(*new_layers)
    return model

def optimize_model(model):
    """Recursively optimize model by fusing Conv+BN(+ReLU) and removing Identity"""
    for name, module in model.named_children():
        optimize_model(module)
        if isinstance(module, nn.Sequential) and len(module) >= 2:
            conv = module[0] if isinstance(module[0], nn.Conv2d) else None
            bn = module[1] if isinstance(module[1], nn.BatchNorm2d) else None
            relu = module[2] if len(module) > 2 and isinstance(module[2], nn.ReLU) else None
            if conv and bn:
                fused_conv = fuse_conv_bn_relu(conv, bn, relu)
                new_seq = [fused_conv]
                if relu:
                    new_seq.append(relu)
                setattr(model, name, nn.Sequential(*new_seq))
        if isinstance(module, nn.Sequential):
            setattr(model, name, remove_identity_layers(module))
    return model


def profile_inference(model, input_shape=(1,3,224,224), device='cpu', warmup=10, repeat=50):
    """Measure average latency of the model in milliseconds"""
    model.eval().to(device)
    dummy_input = torch.randn(input_shape).to(device)
    
    for _ in range(warmup):
        _ = model(dummy_input)
    
    times = []
    for _ in range(repeat):
        start = time.time()
        _ = model(dummy_input)
        end = time.time()
        times.append(end - start)
    
    return np.mean(times) * 1000  


def print_model_summary(model, indent=0):
    """Print model layers recursively"""
    prefix = " " * indent
    for name, module in model.named_children():
        print(f"{prefix}{name}: {module}")
        if list(module.children()):
            print_model_summary(module, indent + 2)


def plot_latency_and_layers(model_before, model_after, latency_before, latency_after, filename='optimization_summary.png'):
    """Plot layer composition before/after and latency comparison"""
    def layer_counts(m):
        layers = []
        def collect_layers(mod):
            for layer in mod.children():
                layers.append(type(layer).__name__)
                if list(layer.children()):
                    collect_layers(layer)
        collect_layers(m)
        return Counter(layers)

    counts_before = layer_counts(model_before)
    counts_after = layer_counts(model_after)

    layer_types = list(set(list(counts_before.keys()) + list(counts_after.keys())))
    before_vals = [counts_before.get(l, 0) for l in layer_types]
    after_vals = [counts_after.get(l, 0) for l in layer_types]

    fig, axs = plt.subplots(1, 2, figsize=(14,6))

    x = range(len(layer_types))
    axs[0].barh(x, before_vals, color='skyblue', label='Before')
    axs[0].barh(x, after_vals, color='orange', left=before_vals, label='After')
    axs[0].set_yticks(x)
    axs[0].set_yticklabels(layer_types)
    axs[0].set_xlabel('Layer Count')
    axs[0].set_title('Layer Composition Before vs After')
    axs[0].legend()

    axs[1].bar(['Before', 'After'], [latency_before, latency_after], color=['skyblue', 'orange'])
    axs[1].set_ylabel('Latency (ms)')
    axs[1].set_title('Inference Latency Comparison')

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Optimization summary saved as {filename}")
    plt.show()


class Args:
    model = 'resnet18'   
    benchmark = True
    summary = True
    device = 'cpu'      

args = Args()

model = getattr(models, args.model)(pretrained=False)
print(f"Loaded model: {args.model}")

import copy
model_before = copy.deepcopy(model)

if args.benchmark:
    latency_before = profile_inference(model, device=args.device)
    print(f"\nLatency before optimization: {latency_before:.2f} ms")

model = optimize_model(model)

if args.benchmark:
    latency_after = profile_inference(model, device=args.device)
    print(f"Latency after optimization: {latency_after:.2f} ms")
    print(f"Speedup: {latency_before / latency_after:.2f}x")

if args.summary:
    print("\nOptimized Model Summary:")
    print_model_summary(model)

plot_latency_and_layers(model_before, model, latency_before, latency_after)

scripted_model = torch.jit.script(model)
torch.jit.save(scripted_model, f"{args.model}_optimized_scripted.pt")
print(f"TorchScript optimized model saved as {args.model}_optimized_scripted.pt")
