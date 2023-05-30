# Awesome-LLM-Compression [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
Awesome LLM compression research papers and tools. 

# Contents

- [Papers](#papers)
  - [Quantization](#general)
  - [Pruning](#architecture)
  - [Sparsity](#quantization)
  - [Distillation](#binarization)
- [Tools](#tools)

## Papers

### Quantization

- ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers <br> NeurIPS 2022 [[Paper]](https://arxiv.org/abs/2206.01861) [[Code]](https://github.com/microsoft/DeepSpeed)

- LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale <br> NeurIPS 2022 [[Paper]](https://arxiv.org/abs/2208.07339) [[Code]](https://github.com/TimDettmers/bitsandbytes)

- LUT-GEMM: Quantized Matrix Multiplication based on LUTs for Efficient Inference in Large-Scale Generative Language Models <br> Arxiv 2022 [[Paper]](https://arxiv.org/abs/2206.09557) 

- Outlier Suppression+: Accurate quantization of large language models by equivalent and optimal shifting and scaling <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2304.09145)

- Quantized Distributed Training of Large Models with Convergence Guarantees <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2302.02390)

- SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models <br> ICML 2023 [[Paper]](https://arxiv.org/abs/2211.10438) [[Code]](https://github.com/mit-han-lab/smoothquant)

- GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers <br> ICLR 2023 [[Paper]](https://arxiv.org/abs/2210.17323) [[Code]](https://github.com/IST-DASLab/gptq)

- RPTQ: Reorder-based Post-training Quantization for Large Language Models <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2304.01089) [[Code]](https://github.com/hahnyuan/RPTQ4LLM)

- ZeroQuant-V2: Exploring Post-training Quantization in LLMs from Comprehensive Study to Low Rank Compensation <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2303.08302) [[Code]](https://github.com/microsoft/DeepSpeed)

- QLoRA: Efficient Finetuning of Quantized LLMs <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2305.14314) [[Code]](https://github.com/artidoro/qlora)

- Integer or Floating Point? New Outlooks for Low-Bit Quantization on Large Language Models <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2305.12356)

- Memory-Efficient Fine-Tuning of Compressed Large Language Models via sub-4-bit Integer Quantization <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2305.14152)


### Pruning
- Structured Pruning of Large Language Models <br> EMNLP 2020 [[Paper]](https://arxiv.org/abs/1910.04732) [[Code]](https://github.com/asappresearch/flop)

- The Optimal BERT Surgeon: Scalable and Accurate Second-Order Pruning for Large Language Models <br> EMNLP 2022 [[Paper]](https://arxiv.org/abs/2203.07259) [[Code]](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT)

- ZipLM: Hardware-Aware Structured Pruning of Language Models <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2302.04089) 

- SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2301.00774) [[Code]](https://github.com/IST-DASLab/sparsegpt)

- LLM-Pruner: On the Structural Pruning of Large Language Models <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2305.11627) [[Code]](https://github.com/horseee/LLM-Pruner)

### Sparsity

### Distillation

- Lifting the Curse of Capacity Gap in Distilling Language Models <br> ACL 2023 [[Paper]](https://arxiv.org/abs/2305.12129) [[Code]](https://github.com/GeneZC/MiniMoE)

- Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes <br> ACL 2023 [[Paper]](https://arxiv.org/abs/2305.02301) 

- Large Language Model Distillation Doesn't Need a Teacher <br> [[Paper]](https://arxiv.org/abs/2305.14864) [[Code]](https://github.com/ananyahjha93/llm-distill)

## Tools
  


