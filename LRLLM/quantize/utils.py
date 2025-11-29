from collections import OrderedDict
from quantize.int_linear import QuantLinear
import torch
from models.transformation import *

def rlq_state_dict(self, destination=None, prefix='', keep_vars=False):
    if destination is None:
        destination = OrderedDict()
    for name, param in self.named_parameters():
        if name.find('smooth') > -1 or name.find('bound_factor') > -1:
            destination[prefix + name] = param if keep_vars else param.detach()
    return destination

def get_rlq_parameters(self, use_shift=True):
    params = []
    template = "smooth" if use_shift else "smooth_scale"
    for n, m in self.named_parameters():
        if n.find('bound_factor') > -1 or n.find(template) > -1:
            params.append(m)
    return iter(params) 

def let_parameters(model, use_shift=True):
    params = []
    template = "smooth" if use_shift else "smooth_scale"
    for n, m in model.named_parameters():
        if n.find(template) > -1:
            params.append(m)
    return iter(params)  

def lwc_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if n.find('bound_factor') > -1 or n.find('rotate') > -1:
            params.append(m)
    return iter(params)  

def get_omni_parameters(model, use_shift=True):
    params = []
    template = "smooth" if use_shift else "smooth_scale"
    for n, m in model.named_parameters():
        if n.find('bound_factor') > -1 or n.find(template) > -1 or n.find('rotate') > -1:
            params.append(m)
    return iter(params)  

def omni_state_dict(model, destination=None, prefix='', keep_vars=False):
    if destination is None:
        destination = OrderedDict()
    for name, param in model.named_parameters():
        if name.find('smooth') > -1 or name.find('bound_factor') > -1 or name.find('rotate') > -1:
            destination[prefix + name] = param if keep_vars else param.detach()
    return destination

def register_scales_and_zeros(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight_quantizer.register_scales_and_zeros()

class TruncateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        truncated_tensor = input.clone()
        truncated_tensor[truncated_tensor.abs() < threshold] = truncated_tensor[truncated_tensor.abs() < threshold].sign() * threshold
        return truncated_tensor
        

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

     
def truncate_number(number, threshold=1e-2):
    # avoid overflow with AMP training
    return TruncateFunction.apply(number, threshold)     

# def smooth_and_quant_temporary(model, args, isllama):
#     if args.let:
#         with torch.no_grad():
#             for name, module in model.named_parameters():
#                 if "smooth_scale" in name:
#                     module.data = truncate_number(module)
#         if isllama:
#             smooth_ln_fcs_temporary(model.attetnion.output.LayerNorm,[model.attention.self.query, model.attention.self.key, model.attention.self.value],
#                                     model.qkv_smooth_scale,model.qkv_smooth_shift)
#             smooth_ln_fcs_temporary(model.output.LayerNorm,[model.intermediate.dense],
#                                     model.fc1_smooth_scale,model.fc1_smooth_shift)
#             smooth_fc_fc_temporary(model.attention.self.value,model.attention.output.dense,
#                                 model.fc2_smooth_scale, model.fc2_smooth_shift)
#             model.output.dense.temp_weight = model.output.dense.weight.detach()
#     else:
#         for name, module in model.named_modules():
#             if isinstance(module, QuantLinear):
#                 module.temp_weight = module.weight
def smooth_and_quant_temporary(model, args, isllama):
    if args.let:
        with torch.no_grad():
            for name, module in model.named_parameters():
                if "smooth_scale" in name:
                    module.data = truncate_number(module)
        if isllama:
            smooth_ln_fcs_temporary(model.input_layernorm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                    model.qkv_smooth_scale,model.qkv_smooth_shift)
            smooth_ln_fcs_temporary(model.post_attention_layernorm,[model.mlp.up_proj,model.mlp.gate_proj],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_fc_fc_temporary(model.self_attn.v_proj,model.self_attn.o_proj,
                                model.out_smooth_scale, model.out_smooth_shift)
            smooth_q_k_temporary(model.self_attn.q_proj, model.self_attn.k_proj,
                                model.qkt_smooth_scale)
            model.mlp.down_proj.temp_weight = model.mlp.down_proj.weight
        else:
            smooth_ln_fcs_temporary(model.self_attn_layer_norm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                    model.qkv_smooth_scale,model.qkv_smooth_shift)
            smooth_ln_fcs_temporary(model.final_layer_norm,[model.fc1],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_ln_fcs_temporary(model.self_attn.v_proj,model.self_attn.out_proj,
                                model.out_smooth_scale, model.out_smooth_shift)
            smooth_q_k_temporary(model.self_attn.q_proj, model.self_attn.k_proj,
                                model.qkt_smooth_scale)
            model.fc2.temp_weight = model.fc2.weight
    else:
        for name, module in model.named_modules():
            if isinstance(module, QuantLinear):
                module.temp_weight = module.weight
    # quant
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            if hasattr(module, "temp_weight"):
                name_tmp = name.replace(".","_")
                if hasattr(model, f"{name_tmp}_rotate_1"):
                    r1 = getattr(model, f"{name_tmp}_rotate_1")
                    r2 = getattr(model, f"{name_tmp}_rotate_2")
                    roatate = r1@r2
                    module.temp_weight = module.weight_quantizer(module.temp_weight, rotate=roatate)
                else:
                    module.temp_weight = module.weight_quantizer(module.temp_weight)
            else:
                name_tmp = name.replace(".","_")
                if hasattr(model, f"{name_tmp}_rotate"):
                    r1 = getattr(model, f"{name_tmp}_rotate_1")
                    r2 = getattr(model, f"{name_tmp}_rotate_2")
                    roatate = r1@r2
                    module.weight = module.weight_quantizer(module.weight, rotate=roatate)
                else:
                    
                    module.weight = module.weight_quantizer(module.weight)
            if not hasattr(module, "temp_bias"):
                module.temp_bias = module.bias
            module.use_temporary_parameter=True
            
def clear_temp_variable(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            if hasattr(module, "temp_weight"):
                del module.temp_weight
            if hasattr(module, "temp_bias"):
                del module.temp_bias

@torch.no_grad()   
def smooth_and_quant_inplace(model, args, isllama):
    if args.let:
        for name, module in model.named_parameters():
            if "smooth_scale" in name:
                module.data = truncate_number(module)
        if isllama:
            smooth_ln_fcs_inplace(model.input_layernorm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                    model.qkv_smooth_scale,model.qkv_smooth_shift)
            smooth_ln_fcs_inplace(model.post_attention_layernorm,[model.mlp.up_proj,model.mlp.gate_proj],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_fc_fc_inplace(model.self_attn.v_proj,model.self_attn.o_proj,
                                model.out_smooth_scale, model.out_smooth_shift)
        else: # opt
            smooth_ln_fcs_inplace(model.self_attn_layer_norm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                    model.qkv_smooth_scale,model.qkv_smooth_shift)
            smooth_ln_fcs_inplace(model.final_layer_norm,[model.fc1],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_fc_fc_inplace(model.self_attn.v_proj,model.self_attn.out_proj,
                                model.out_smooth_scale, model.out_smooth_shift)
        smooth_q_k_inplace(model.self_attn.q_proj, model.self_attn.k_proj,
                            model.qkt_smooth_scale)

    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            name_tmp = name.replace(".","_")
            if hasattr(model, f"{name_tmp}_rotate"):
                r1 = getattr(model, f"{name_tmp}_rotate_1")
                r2 = getattr(model, f"{name_tmp}_rotate_2")
                roatate = r1@r2
                module.weight = module.weight_quantizer(module.weight, rotate=roatate)
            else:
                print('----error error-----')
                module.weight = module.weight_quantizer(module.weight)
            module.use_temporary_parameter=False   

@torch.no_grad()   
def smooth_and_quant_inplace_visual(model, args, isllama):

    if args.let:
        for name, module in model.named_parameters():
            if "smooth_scale" in name:
                module.data = truncate_number(module)

        smooth_ln_fcs_inplace(model.ln_1,[model.mlp.c_fc],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)

    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight = module.weight_quantizer(module.weight)
            module.use_temporary_parameter=False 

def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
    # setting weight quantization here does not affect actual forward pass
    self.use_weight_quant = weight_quant
    self.use_act_quant = act_quant
    for m in self.modules():
        if isinstance(m, (QuantLinear)):
            m.set_quant_state(weight_quant, act_quant)
