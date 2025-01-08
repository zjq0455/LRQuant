# LRQuant: A Unified and Learnable Framework to Post-training Quantization for Transformer-based Large Foundation Models


Post-training quantization (PTQ) for transformer-based large foundation models (LFMs) significantly accelerates model inference and relieves memory constraints, without incurring model training. However, existing methods face three main issues: 1) The scaling factors, which are commonly used in scale reparameterization based weight-activation quantization for mitigating the quantization errors, are mostly hand-crafted defined which may lead to suboptimal results; 2) The formulation of current quantization error defined by L2-norm ignores the directional shifts after quantization;3) Most methods are devised tailored for single scenario, i.e., only evaluated on LLMs or only designed for weight-only quantization, which lacks of a comprehensive evaluation on diverse benchmarks and a broad application scope. To address these challenges, this paper introduces a unified Learnable and Robust post-training Quantization framework for transformer based LFMs and various quantization scenarios, called LRQuant. Firstly, we consider an efficient block-wise learnable paradigm to find optimal scaling factors which are initialized by logarithmic activation equivalent and get suitable clipping range of quantization steps. In addition, we empirically find that only relying on MSE loss could hardly lead to optimal quantization results, so we reformulate the quantization error and then propose a novel loss function based on the negative logarithm of cosine similarity (NLC loss) between outputs of full-precision and quantized block. To fully investigate the potentiality of our learnable paradigm, we propose a more superior version LRQuant+. Specifically, we devise learnable rotation vectors to further directly reduce directional gaps. In addition, we improve the block-wise optimization framework into a novel two-branch nature which jointly considers the error propagation and homologous reconstruction error. Extensive experiments demonstrate the superiority of our LRQuant and LRQuant+, as well as their unified effectiveness across various LFMs for both weight-activation and weight-only quantization, especially under challenging quantization scenarios, i.e., W4A4 and W2A16 on LLMs, ViTS, and MLLMs.

## Usage

LRQuant can well implement weight-only and weight-activation quantization on LLM, MLLM and ViT. You can quantize LLM and MLLM models using [LRLLM](./LRLLM), and use [LRViT](./LRViT) to quantize ViT and DeiT models.

## Related Project
[SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://github.com/mit-han-lab/smoothquant)

[OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models](https://github.com/OpenGVLab/OmniQuant.git)

[AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://github.com/mit-han-lab/llm-awq)

[GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://github.com/IST-DASLab/gptq)

[PTQ4ViT: Post-training quantization for vision transformers with twin uniform quantization](https://github.com/hahnyuan/PTQ4ViT)

[FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer](https://github.com/megvii-research/FQ-ViT)

[RepQ-ViT: Scale Reparameterization for Post-Training Quantization of Vision Transformers](https://github.com/zkkli/RepQ-ViT)




