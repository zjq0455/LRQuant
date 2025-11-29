# RLQLLM
## Usage
**We provide full script to run RLQuant. We use llama-7b as an example here**:
1. Obtain the channel-wise scales and shifts required for initialization:

```
python generate_act_scale_shift.py --model /PATH/TO/llama/llama-7b
```
### LRQuant
2. Weight-activation quantization
```
# W4A4 ppl
CUDA_VISIBLE_DEVICES=0 python main.py \
--model /PATH/TO/llama/llama-7b  \
--epochs 20 --output_dir ./log/llama-7b-w4a4 \
--eval_ppl --wbits 4 --abits 4 --lwc --let 

# W4A4 zero-shot
CUDA_VISIBLE_DEVICES=0 python main.py \
--model /PATH/TO/llama/llama-7b  \
--epochs 20 --output_dir ./log/llama-7b-w4a4 \
--wbits 4 --abits 4 --lwc --let \
--tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande
```

#### LRQuant+
2. Weight-activation quantization
```
# W4A4 ppl
CUDA_VISIBLE_DEVICES=0 python main.py \
--model /PATH/TO/llama/llama-7b  \
--epochs 20 --output_dir ./log/llama-7b-w4a4 \
--eval_ppl --wbits 4 --abits 4 --lwc --let --lr_plus

# W4A4 zero-shot
CUDA_VISIBLE_DEVICES=0 python main.py \
--model /PATH/TO/llama/llama-7b  \
--epochs 20 --output_dir ./log/llama-7b-w4a4 \
--wbits 4 --abits 4 --lwc --let --lr_plus \
--tasks piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande
```

## Related Project
[SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://github.com/mit-han-lab/smoothquant)

[OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models](https://github.com/OpenGVLab/OmniQuant.git)