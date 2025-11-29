import torch
import torch.nn as nn
from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from quantize.int_linear import QuantLinear
# import auto_gptq.nn_modules.qlinear.qlinear_cuda as qlinear_cuda
from contextlib import nullcontext
import copy
import math
import utils
import os
import pdb
import gc
from scipy.stats import kurtosis
import scipy.stats
from quantize.utils import let_parameters, lwc_parameters, get_omni_parameters,\
                            omni_state_dict, register_scales_and_zeros,smooth_and_quant_temporary,\
                            smooth_and_quant_inplace,clear_temp_variable,set_quant_state, get_rlq_parameters,rlq_state_dict

def add_new_module(name, original_module, added_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = original_module
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], added_module)
    else:
        setattr(original_module, name, added_module) 


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, QuantLinear)}


def LRQuant(
    lm,
    args,
    dataloader,
    act_scales,
    act_shifts,
    logger=None,
):
    logger.info("Starting ...")
    print(lm.model)
    # move embedding layer and first layer to target device
    model = lm.model
    dev = lm.device
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = False
    if "llama" in args.net.lower() or "vila" in args.net.lower():
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        DecoderLayer = QuantLlamaDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "o_proj":"out",
            "up_proj":"fc1"
        }
        layer_name_prefix = "model.layers"
    elif "opt" in args.net.lower():
        layers = model.model.decoder.layers
        print(layers)
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        DecoderLayer = QuantOPTDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "out_proj":"out",
            "fc1":"fc1"
        }
        layer_name_prefix = "model.decoder.layers"
    else:
        raise ValueError("Only support for opt/llama/Llama-2 now")
    
    
    layers[0] = layers[0].to(dev)
    if args.deactive_amp and args.epochs>0:
        dtype = torch.float
        traincast = nullcontext
    else:
        dtype = torch.float16
        traincast = torch.cuda.amp.autocast
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )

    cache = {"i": 0}
    # catch the first layer input 
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            if self.is_llama:
                cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama

    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
    
    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "llama" in args.net.lower() or "vila" in args.net.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    elif "opt" in args.net.lower():
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    else:
        raise ValueError("Only support for opt/llama/Llama-2 now")
    torch.cuda.empty_cache()

    
    # same input of first layer for fp model and quant model
    quant_inps = inps
    fp_inps = copy.deepcopy(inps)   # take output of fp model as input
    quant_inps_fp = copy.deepcopy(inps)
    

    attention_mask = cache["attention_mask"]
    # attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1) if args.deactive_amp else attention_mask.repeat(args.batch_size,1,1,1).float()
    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1).float()
    else:
        logger.info(
            "No attention mask caught from the first layer."
            " Seems that model's attention works without a mask."
        )
        attention_mask_batch = None
    loss_func = torch.nn.MSELoss()
    # loss_func = torch.nn.L1Loss()
    if is_llama:
        position_ids = cache["position_ids"]
    else:
        position_ids = None
    cossim = nn.CosineSimilarity(dim=2)

    if args.resume:
        rlq_parameters = torch.load(args.resume)
    else:
        rlq_parameters = {}
    
    for i in range(len(layers)):
        logger.info(f"=== Start quantize layer {i} ===")
        print(len(layers))
        
        layer = layers[i].to(dev)
        # w3-mix
        # if  i in [27, 25, 26, 28, 29, 24]:
        # if  i in [25, 27, 23, 28, 18, 15]:
        #     args.weight_quant_params['n_bits'] = 2
        #     print(args.weight_quant_params)
        #     args.epochs=40
        # # elif i in [0, 31, 1, 6, 4, 7]:
        # elif i in [ 2, 22, 30, 31, 1, 0]:
        #     args.weight_quant_params['n_bits'] = 4
        #     print(args.weight_quant_params)
        #     args.epochs=20
        # else:
        #     args.weight_quant_params['n_bits'] = 3
        #     args.epochs=20
        # if i in [ 2, 22, 30, 31, 1, 0]:
        #     args.weight_quant_params['n_bits'] = 4
        #     print(args.weight_quant_params)
        #     args.epochs=20
        # else:
        #     args.weight_quant_params['n_bits'] = 3
        #     args.epochs=20
        # logger.info(args.weight_quant_params)
        if False:
            # for llava, we only leverage lwc, which can be achieve by simply replace Linear with QuantLinear
            qlayer = copy.deepcopy(layer)
            for name, module in qlayer.named_modules():
                if isinstance(module,torch.nn.Linear):
                    kurt = scipy.stats.kurtosis(module.weight.data.detach().cpu().flatten().numpy())
                    # if kurt < 0.14:
                    if kurt < 0.077:
                        args.weight_quant_params['n_bits'] = 3
                        logger.info(f'{name} has weight bits 3')
                    # elif kurt > 1.83:
                    elif kurt > 2.11:
                        args.weight_quant_params['n_bits'] = 5
                        logger.info(f'{name} has weight bits 5')
                    else:
                        args.weight_quant_params['n_bits'] = 4
                        logger.info(f'{name} has weight bits 4')
                    quantlinear = QuantLinear(module, args.weight_quant_params, args.act_quant_params)
                    add_new_module(name, qlayer, quantlinear)  
        else:
            qlayer = DecoderLayer(lm.model.config, layer, args) 
        # qlayer = DecoderLayer(lm.model.config, layer, args)
        qlayer = qlayer.to(dev)

        
        # obtain output of full-precision model
        set_quant_state(qlayer, weight_quant=False, act_quant=False)
        if args.epochs > 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    for j in range(args.nsamples):
                        fp_inps[j] = qlayer(fp_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
                        quant_inps_fp[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]

        # init smooth parameters
        set_quant_state(qlayer, weight_quant=False, act_quant=True)  # weight will be manually quantized before forward
        qlayer.let = args.let
        use_shift = True 
        
        if args.let:
            # init channel-wise scaling and shift
            qlayer.register_parameter("qkt_smooth_scale",torch.nn.Parameter(torch.ones(layer.self_attn.q_proj.out_features,device=dev, dtype=dtype)))
            for name,module in qlayer.named_modules():
                if isinstance(module, QuantLinear):
                    for key in pairs.keys():
                        if key in name:
                            act = act_scales[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype).clamp(min=1e-5) #4096
                            scale = (act/torch.log2(2+act)).clamp(min=1e-5) #weight
                            if use_shift and not is_llama:
                                shift = act_shifts[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype)
                            else:
                                shift = torch.zeros_like(act)  
                            qlayer.register_parameter(f"{pairs[key]}_smooth_shift",torch.nn.Parameter(shift))
                            qlayer.register_parameter(f"{pairs[key]}_smooth_scale",torch.nn.Parameter(scale))
                    if args.lr_plus:
                        r1 = torch.ones(module.weight.shape[0], args.rotate_rank).to(dev)
                        r2 = torch.ones(args.rotate_rank, module.weight.shape[1]).to(dev)
                        name_tmp = name.replace(".","_")
                        # print('----rotate register-----')
                        qlayer.register_parameter(f"{name_tmp}_rotate_1",torch.nn.Parameter(r1,requires_grad=True))
                        qlayer.register_parameter(f"{name_tmp}_rotate_2",torch.nn.Parameter(r2,requires_grad=True))  
        
        if args.resume:
            qlayer.load_state_dict(rlq_parameters[i], strict=False)
        
        if args.epochs > 0:
            with torch.no_grad():
                qlayer.float()      # required for AMP training
            # create optimizer
            optimizer = torch.optim.AdamW(
                [{"params":let_parameters(qlayer, use_shift),"lr":args.let_lr}, {"params":lwc_parameters(qlayer),"lr":args.lwc_lr}],weight_decay=args.wd)
            loss_scaler = utils.NativeScalerWithGradNormCount()

            ema_loss1 = 1.0
            ema_loss2 = 1.0
            momentum = 0.95           
                   
            for epochs in range(args.epochs):
                loss_list = []
                norm_list = []
                for j in range(args.nsamples//args.batch_size):    
                    index = j * args.batch_size
                    # obtain output of quantization model
                    with traincast():
                        smooth_and_quant_temporary(qlayer, args, True)
                        quant_out = qlayer(quant_inps[index:index+args.batch_size,], attention_mask=attention_mask_batch,position_ids=position_ids)[0]
                        loss1 =  loss_func(fp_inps[index:index+args.batch_size,], quant_out)

                        if args.lr_plus:
                            loss1 += loss_func(quant_inps_fp[index:index+args.batch_size,], quant_out)
                            cos2 = cossim(quant_inps_fp[index:index+args.batch_size,],quant_out).mean().abs()
                            loss2 -= torch.log(cos2)
                        
                            ema_loss1 = momentum * ema_loss1 + (1 - momentum) * loss1.item()
                            ema_loss2 = momentum * ema_loss2 + (1 - momentum) * loss2.item()
                            norm_loss1 = loss1 / (ema_loss1 + 1e-6)
                            norm_loss2 = loss2 / (ema_loss2 + 1e-6)
                            loss = norm_loss1 + norm_loss2   #dynamic
                        else:
                            cos1 = cossim(quant_out,fp_inps[index:index+args.batch_size,]).mean().abs()
                            loss2 = -torch.log(cos1)
                            loss = loss1 + loss2

                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        pdb.set_trace()
                        
                    loss_list.append(loss.data)
                    optimizer.zero_grad()
                    norm = loss_scaler(loss, optimizer,parameters=get_rlq_parameters(qlayer, use_shift))
                    norm_list.append(norm.data)

                loss_mean = torch.stack(loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(f"layer {i} iter {epochs} loss:{loss_mean} norm:{norm_mean} max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2} ")
            clear_temp_variable(qlayer)
            del optimizer
        
        # real smooth and quantization
        smooth_and_quant_inplace(qlayer, args, True)     
        if args.epochs>=0:
            # update input of quantization model
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    for j in range(args.nsamples):
                        quant_inps[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
            register_scales_and_zeros(qlayer)
            qlayer.half()
            layers[i] = qlayer.to("cpu")
            rlq_parameters[i] = rlq_state_dict(qlayer)
        else:
            register_scales_and_zeros(qlayer)
            qlayer.half()
            layers[i] = qlayer.to("cpu")          
        
        del layer
        torch.cuda.empty_cache()
        

    del inps
    del quant_inps
    del fp_inps
    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model