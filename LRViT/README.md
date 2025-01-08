# LRQuant(+) for ViTs

## Usage
**We provide the detailed command to run LRQuant and LRQuant+ for ViTs. We use vit-base-patch16-224 as an example here**:
1. Obtain the channel-wise scales and shifts required for initialization:

```
cd LRViT
python generate_act_scale_shift.py --model /PATH/TO/vit-base-patch16-224
```

### LRQuant
2. Weight-only quantization
```
# W4A16
CUDA_VISIBLE_DEVICES=0 python main.py \
--model /PATH/TO/vit-base-patch16-224  \
--epochs 20 --output_dir ./log/vit-base-patch16-224-w4a16 \
--wbits 4 --abits 16 --lwc
```

3. Weight-activation quantization

```
# W4A4
CUDA_VISIBLE_DEVICES=0 python main.py \
--model /PATH/TO/vit-base-patch16-224  \
--epochs 20 --output_dir ./log/vit-base-patch16-224-w4a4 \
--wbits 4 --abits 4 --lwc --let \
--tasks ImageNet
```

### LRQuant+
4. Weight-only quantization
```
# W4A16
CUDA_VISIBLE_DEVICES=0 python main.py \
--model /PATH/TO/vit-base-patch16-224  \
--epochs 20 --output_dir ./log/vit-base-patch16-224-w4a16 \
--wbits 4 --abits 16 --lwc --lr_plus
```

5. weight-activation quantization

```
# W4A4
CUDA_VISIBLE_DEVICES=0 python main.py \
--model /PATH/TO/vit-base-patch16-224  \
--epochs 20 --output_dir ./log/vit-base-patch16-224-w4a4 \
--wbits 4 --abits 4 --lwc --let --lr_plus \
--tasks ImageNet
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
