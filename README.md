# Awesome-LLM-Compression [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
Awesome LLM compression research papers and tools to accelerate the LLM training and inference. 

# Contents

- [Papers](#papers)
  - [Quantization](#quantization)
  - [Pruning and Sparsity](#pruning-and-sparsity)
  - [Distillation](#distillation)
  - [Efficient Prompting](#efficient-prompting)
  - [Other](#other)
- [Tools](#tools)

## Papers

### Quantization

- ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers <br> NeurIPS 2022 [[Paper]](https://arxiv.org/abs/2206.01861) [[Code]](https://github.com/microsoft/DeepSpeed)

- LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale <br> NeurIPS 2022 [[Paper]](https://arxiv.org/abs/2208.07339) [[Code]](https://github.com/TimDettmers/bitsandbytes)

- LUT-GEMM: Quantized Matrix Multiplication based on LUTs for Efficient Inference in Large-Scale Generative Language Models <br> Arxiv 2022 [[Paper]](https://arxiv.org/abs/2206.09557) 

- SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models <br> ICML 2023 [[Paper]](https://arxiv.org/abs/2211.10438) [[Code]](https://github.com/mit-han-lab/smoothquant)

- FlexRound: Learnable Rounding based on Element-wise Division for Post-Training Quantization <br> ICML 2023 [[Paper]](https://arxiv.org/abs/2301.12017) [[Code(DeepSpeed)]](https://github.com/microsoft/DeepSpeed)

- Understanding INT4 Quantization for Transformer Models: Latency Speedup, Composability, and Failure Cases <br> ICML 2023 [[Paper]](https://arxiv.org/abs/2306.00317) [[Code]](https://openreview.net/attachment?id=-tYCaP0phY_&name=supplementary_material)

- GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers <br> ICLR 2023 [[Paper]](https://arxiv.org/abs/2210.17323) [[Code]](https://github.com/IST-DASLab/gptq)

- RPTQ: Reorder-based Post-training Quantization for Large Language Models <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2304.01089) [[Code]](https://github.com/hahnyuan/RPTQ4LLM)

- Outlier Suppression+: Accurate quantization of large language models by equivalent and optimal shifting and scaling <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2304.09145)

- Quantized Distributed Training of Large Models with Convergence Guarantees <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2302.02390)

- ZeroQuant-V2: Exploring Post-training Quantization in LLMs from Comprehensive Study to Low Rank Compensation <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2303.08302) [[Code]](https://github.com/microsoft/DeepSpeed)

- QLoRA: Efficient Finetuning of Quantized LLMs <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2305.14314) [[Code]](https://github.com/artidoro/qlora)

- Integer or Floating Point? New Outlooks for Low-Bit Quantization on Large Language Models <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2305.12356)

- The Quantization Model of Neural Scaling <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2303.13506)

- Memory-Efficient Fine-Tuning of Compressed Large Language Models via sub-4-bit Integer Quantization <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2305.14152)

- Compress, Then Prompt: Improving Accuracy-Efficiency Trade-off of LLM Inference with Transferable Prompt <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2305.11186)

- AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2306.00978) [[Code]](https://github.com/mit-han-lab/llm-awq)

- LLM-QAT: Data-Free Quantization Aware Training for Large Language Models <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2305.17888) [[Code]](https://github.com/facebookresearch/LLM-QAT)

- SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2306.03078) [[Code]](https://github.com/Vahe1994/SpQR)

- OWQ: Lessons learned from activation outliers for weight quantization in large language models <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2306.02272)

- SqueezeLLM: Dense-and-Sparse Quantization <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2306.07629)  [[Code]](https://github.com/SqueezeAILab/SqueezeLLM)

- INT2.1: Towards Fine-Tunable Quantized Large Language Models with Error Correction through Low-Rank Adaptation <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2306.08162)

- INT-FP-QSim: Mixed Precision and Formats For Large Language Models and Vision Transformers <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2307.03712) [[Code]](https://github.com/lightmatter-ai/INT-FP-QSim)

- QIGen: Generating Efficient Kernels for Quantized Inference on Large Language Models <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2307.03738) [[Code]](https://github.com/IST-DASLab/QIGen)

- Do Emergent Abilities Exist in Quantized Large Language Models: An Empirical Study <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2307.08072)

- ZeroQuant-FP: A Leap Forward in LLMs Post-Training W4A8 Quantization Using Floating-Point Formats <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2307.09782) [[Code (DeepSpeed)]](https://github.com/microsoft/DeepSpeed)

- OliVe: Accelerating Large Language Models via Hardware-friendly Outlier-Victim Pair Quantization <br> ISCA 2023 [[Paper]](https://arxiv.org/abs/2304.07493)

- QuIP: 2-Bit Quantization of Large Language Models With Guarantees <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2307.13304) [[Code]](https://github.com/jerry-chee/QuIP)

- Floating-Point Quantized Transformers <br> Arxiv 2023 [[Paper]](https://openreview.net/forum?id=eaupB5ToNR)

### Pruning and Sparsity

- The Lazy Neuron Phenomenon: On Emergence of Activation Sparsity in Transformers <br> ICLR 2023 [[Paper]](https://openreview.net/forum?id=TJ2nxciYCk-)

- Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time <br> ICML 2023 [[Paper]](https://proceedings.mlr.press/v202/liu23am.html)  [[Code]](https://github.com/FMInference/DejaVu)

- LoSparse: Structured Compression of Large Language Models based on Low-Rank and Sparse Approximation <br> ICML 2023 [[Paper]](https://arxiv.org/abs/2306.11222)  [[Code]](https://github.com/yxli2123/LoSparse)

- SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2301.00774) [[Code]](https://github.com/IST-DASLab/sparsegpt)

- LLM-Pruner: On the Structural Pruning of Large Language Models <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2305.11627) [[Code]](https://github.com/horseee/LLM-Pruner)

- Prune and Tune: Improving Efficient Pruning Techniques for Massive Language Models <br> ICLR 2023 TinyPapers [[Paper]](https://openreview.net/pdf?id=cKlgcx7nSZ)

- Unlocking Context Constraints of LLMs: Enhancing Context Efficiency of LLMs with Self-Information-Based Content Filtering <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2304.12102) [[Code]](https://github.com/liyucheng09/Selective_Context)

- Rethinking the Role of Scale for In-Context Learning: An Interpretability-based Case Study at 66 Billion Scale <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2212.09095) [[Code]](https://github.com/amazon-science/llm-interpret)

- A Simple and Effective Pruning Approach for Large Language Models <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2306.11695) [[Code]](https://github.com/locuslab/wanda)


### Distillation

- Lifting the Curse of Capacity Gap in Distilling Language Models <br> ACL 2023 [[Paper]](https://arxiv.org/abs/2305.12129) [[Code]](https://github.com/GeneZC/MiniMoE)

- Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes <br> ACL 2023 [[Paper]](https://arxiv.org/abs/2305.02301) 

- LaMini-LM: A Diverse Herd of Distilled Models from Large-Scale Instructions <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2304.14402) [[Code]](https://github.com/mbzuai-nlp/LaMini-LM)

- Large Language Model Distillation Doesn't Need a Teacher <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2305.14864) [[Code]](https://github.com/ananyahjha93/llm-distill)

- The False Promise of Imitating Proprietary LLMs <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2305.15717)

- GPT4All: Training an Assistant-style Chatbot with Large Scale Data Distillation from GPT-3.5-Turbo <br> Arxiv 2023 [[Paper]](https://s3.amazonaws.com/static.nomic.ai/gpt4all/2023_GPT4All_Technical_Report.pdf) [[Code]](https://github.com/nomic-ai/gpt4all)

- PaD: Program-aided Distillation Specializes Large Models in Reasoning <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2305.13888) 

- Knowledge Distillation of Large Language Models <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2306.08543) [[Code]](https://github.com/microsoft/LMOps/tree/main/minillm)

- GKD: Generalized Knowledge Distillation for Auto-regressive Sequence Models <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2306.13649)

- Chain-of-Thought Prompt Distillation for Multimodal Named Entity and Multimodal Relation Extraction <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2306.14122)

- Task-agnostic Distillation of Encoder-Decoder Language Models <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2305.12330)

### Efficient Prompting

- Did You Read the Instructions? Rethinking the Effectiveness of Task Definitions in Instruction Learning <br> ACL 2023 [[Paper]](https://arxiv.org/abs/2306.01150) [[Code]](https://github.com/fanyin3639/Rethinking-instruction-effectiveness)

- Efficient Prompting via Dynamic In-Context Learning <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2305.11170)

- Learning to Compress Prompts with Gist Tokens <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2304.08467) [[Code]](https://github.com/jayelm/gisting)

- Batch Prompting: Efficient Inference with Large Language Model APIs <br> Arxiv 2023 [[Paper]](https://github.com/HKUNLP/batch-prompting) [[Code]](https://arxiv.org/abs/2301.08721)

- Adapting Language Models to Compress Contexts <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2305.14788) [[Code]](https://github.com/princeton-nlp/AutoCompressors)

- In-context Autoencoder for Context Compression in a Large Language Model <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2307.06945) 

### Other

- TensorGPT: Efficient Compression of the Embedding Layer in LLMs based on the Tensor-Train Decomposition <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2307.00526)

- Dynamic Context Pruning for Efficient and Interpretable Autoregressive Transformers <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2305.15805)

- SkipDecode: Autoregressive Skip Decoding with Batching and Caching for Efficient LLM Inference <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2307.02628)

- Scaling In-Context Demonstrations with Structured Attention <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2307.02690)

- Response Length Perception and Sequence Scheduling: An LLM-Empowered LLM Inference Pipeline <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2305.13144) [[Code]](https://github.com/zhengzangw/Sequence-Scheduling)

- Text Alignment Is An Efficient Unified Model for Massive NLP Tasks <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2307.02729) [[Code]](https://github.com/yuh-zha/Align)

- CPET: Effective Parameter-Efficient Tuning for Compressed Large Language Models <br> Arxiv 2023 [[Paper]](https://arxiv.org/abs/2307.07705)


## Tools

- BMCook: Model Compression for Big Models [[Code]](https://github.com/OpenBMB/BMCook)
  
- llama.cpp: Inference of LLaMA model in pure C/C++ [[Code]](https://github.com/ggerganov/llama.cpp)

- LangChain: Building applications with LLMs through composability [[Code]](https://github.com/hwchase17/langchain)

- GPTQ-for-LLaMA: 4 bits quantization of LLaMA using GPTQ [[Code]](https://github.com/qwopqwop200/GPTQ-for-LLaMa)

- Alpaca-CoT: An Instruction Fine-Tuning Platform with Instruction Data Collection and Unified Large Language Models Interface [[Code]](https://github.com/PhoebusSi/Alpaca-CoT)

- vllm: A high-throughput and memory-efficient inference and serving engine for LLMs [[Code]](https://github.com/vllm-project/vllm)

- LLaMA Efficient Tuning: Fine-tuning LLaMA with PEFT (PT+SFT+RLHF with QLoRA) [[Code]](https://github.com/hiyouga/LLaMA-Efficient-Tuning)

- Efficient-Tuning-LLMs: (Efficient Finetuning of QLoRA LLMs). QLoRA, LLama, bloom, baichuan-7B, GLM [[Code]](https://github.com/jianzhnie/Efficient-Tuning-LLMs)

- bitsandbytes: 8-bit CUDA functions for PyTorch [[Code]](https://github.com/TimDettmers/bitsandbytes)

- ExLlama: A more memory-efficient rewrite of the HF transformers implementation of Llama for use with quantized weights. [[Code]](https://github.com/turboderp/exllama)

- Lit-LLaMA: Implementation of the LLaMA language model based on nanoGPT. Supports flash attention, Int8 and GPTQ 4bit quantization, LoRA and LLaMA-Adapter fine-tuning, pre-training. [[Code]](https://github.com/Lightning-AI/lit-llama)

- lama.onnx: LLaMa/RWKV onnx models, quantization and testcase [[Code]](https://github.com/tpoisonooo/llama.onnx)

