import time
import torch.nn.functional as F
import torch
import torch.nn as nn
import scipy.stats
from scipy.stats import kurtosis
import copy
import math
DEV = torch.device('cuda:0')

def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM, AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model


@torch.no_grad()
def llama_sequential(model, dev):
    print("Starting...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    print("Ready.")

    block_kurtosis_value= {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        layer.float() 
        kurtosis_value = 0.0

        for name, module in layer.named_modules():
            if isinstance(module,torch.nn.Linear):
                kurt = scipy.stats.kurtosis(module.weight.data.detach().cpu().flatten().numpy())
                block_kurtosis_value[kurt] = str(i) + name
                # print(kurtosis_value)

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    print(f"block kurtosis value: {block_kurtosis_value}")
    return block_kurtosis_value


if __name__ == "__main__":
    import argparse
    from datautils_block import *

    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str, help="LlaMA model to load")
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )

    args = parser.parse_args()


    model = get_llama(args.model)
    model.eval()

    tick = time.time()
    block_kurtosis_value = llama_sequential(model, DEV)
    print('==============================')
    # print(block_kurtosis_value)
    # print(block_kurtosis_value.keys())
    print(model)
    print('==============================')
    sorted_kurtosis = dict(sorted(block_kurtosis_value.items()))
    # a = 4/3 c 
    print('------------------------------')
    print(sorted_kurtosis)
    print(sorted_kurtosis.keys())
    print('------------------------------')
    robust_ratio = math.floor(len(sorted_kurtosis) * 0.2)
    while robust_ratio % 3 !=0:
        robust_ratio  = robust_ratio - 1
    sensitive_ration = robust_ratio
    moderate_ration = len(sorted_kurtosis) - robust_ratio - sensitive_ration

    keys = list(sorted_kurtosis.keys())

    # import pdb
    # pdb.set_trace()
    robust = keys[:robust_ratio]
    moderate = keys[robust_ratio:math.floor(len(sorted_kurtosis)-sensitive_ration)]
    sensitive = keys[math.floor(len(sorted_kurtosis)-sensitive_ration):]
    robust_dict = [sorted_kurtosis.get(key) for key in robust]
    moderate_dict = [sorted_kurtosis.get(key) for key in moderate]
    sensitive_dict = [sorted_kurtosis.get(key) for key in sensitive]
    print("quantization robust group(256):")
    print(robust_dict)
    print("quantization moderate group(128): ")
    print(moderate_dict)
    print("quantization sensitive group(32): ")
    print(sensitive_dict)

    print(time.time() - tick)