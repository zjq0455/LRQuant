# LRQuant(+) for LLMs and MLLMs
## Usage
**We provide the detailed command to run LRQuant and LRQuant+ for LLMs and MLLMs. We use llama-7b as an example here**:
1. Obtain the channel-wise scales and shifts required for initialization:

```
cd LRLLM
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
3. Weight-only quantization
```
# W4A16 ppl
CUDA_VISIBLE_DEVICES=0 python main.py \
--model /PATH/TO/llama/llama-7b  \
--epochs 20 --output_dir ./log/llama-7b-w4a16 \
--eval_ppl --wbits 4 --abits 4 --lwc
```

#### LRQuant+
4. Weight-activation quantization
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

5. Weight-only quantization
```
# W4A16 ppl
CUDA_VISIBLE_DEVICES=0 python main.py \
--model /PATH/TO/llama/llama-7b  \
--epochs 20 --output_dir ./log/llama-7b-w4a16 \
--eval_ppl --wbits 4 --abits 4 --lwc --lr_plus
```

More detailed and optional arguments:
- `--model`: the local model path or huggingface format.
- `--wbits`: weight quantization bits.
- `--abits`: activation quantization bits.
- `--lwc`: activate the Learnable Weight Clipping (LWC).
- `--let`: activate the Learnable Equivalent Transformation (LET).
- `--lwc_lr`: learning rate of LWC parameters, 1e-2 as default.
- `--let_lr`: learning rate of LET parameters, 5e-3 as default.
- `--epochs`: training epochs. You can set it as 0 to evaluate pre-trained MSQuant checkpoints.
- `--nsamples`: number of calibration samples, 128 as default.
- `--save_dir`: saving the quantization model for further exploration.