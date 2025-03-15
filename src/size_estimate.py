import torch
import numpy as np
import torch.nn as nn


def flop(model, input_shape, device):
    total = {}

    def count_flops(name):
        def hook(module, input, output):
            "Hook that calculates number of floating point operations"
            flops = {}
            batch_size = input[0].shape[0]
            if isinstance(module, nn.Linear):
                # TODO: fill-in (start)
                input_features = module.in_features
                output_features = module.out_features
                flops['forward'] = batch_size * input_features * output_features * 2
                if module.bias is not None:
                    flops['forward'] += batch_size * output_features
                # TODO: fill-in (end)

            if isinstance(module, nn.Conv2d):
                # TODO: fill-in (start)
                input_shape = input[0].shape
                output_shape = output.shape
                
                kernel_size = module.kernel_size
                in_channels = module.in_channels
                out_channels = module.out_channels
                groups = module.groups
                
                output_height = output_shape[2]
                output_width = output_shape[3]
                
                flops['forward'] = batch_size * output_height * output_width * out_channels * kernel_size[0] * kernel_size[1] * (in_channels / groups) * 2
                
                if module.bias is not None:
                    flops['forward'] += batch_size * output_height * output_width * out_channels
                # TODO: fill-in (end)

            if isinstance(module, nn.BatchNorm1d):
                # TODO: fill-in (start)
                input_shape = input[0].shape
                num_elements = batch_size
                for dim in input_shape[1:]:
                    num_elements *= dim
                flops['forward'] = num_elements * 2
                # TODO: fill-in (end)

            if isinstance(module, nn.BatchNorm2d):
                # TODO: fill-in (start)
                input_shape = input[0].shape
                num_elements = batch_size
                for dim in input_shape[1:]:
                    num_elements *= dim
                flops['forward'] = num_elements * 2
                # TODO: fill-in (end)
            total[name] = flops
        return hook

    handle_list = []
    for name, module in model.named_modules():
        handle = module.register_forward_hook(count_flops(name))
        handle_list.append(handle)
    input = torch.ones(input_shape).to(device)
    model(input)

    # Remove forward hooks
    for handle in handle_list:
        handle.remove()
    return total


def count_trainable_parameters(model):
    """
    Return the total number of trainable parameters for [model]
    :param model:
    :return:
    """
    # TODO: fill-in (start)
    total_params = 0
    for param in model.parameters():
        if param.requires_grad:
            total_params += param.numel()
    return total_params
    # TODO: fill-in (end)


def compute_forward_memory(model, input_shape, device):
    """
    Calculate the memory (in bytes) needed for a forward pass through the model
    :param model: The neural network model
    :param input_shape: Shape of the input tensor
    :param device: Device to run the model on (cpu or cuda)
    :return: Total memory usage in bytes
    """
    
    # TODO: fill-in (start)
    tensor_sizes = {}
    
    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            tensor_sizes[output.data_ptr()] = output.numel() * 4
        elif isinstance(output, tuple):
            for out in output:
                if isinstance(out, torch.Tensor):
                    tensor_sizes[out.data_ptr()] = out.numel() * 4
    
    hooks = []
    for module in model.modules():
        hooks.append(module.register_forward_hook(hook_fn))
    
    # Forward pass
    input_tensor = torch.ones(input_shape).to(device)
    model(input_tensor)
    
    for hook in hooks:
        hook.remove()
    
    total_memory = input_tensor.numel() * 4
    
    for size in tensor_sizes.values():
        total_memory += size
    
    return total_memory
    # TODO: fill-in (end)

