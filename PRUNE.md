# Awesome-LLM-Prune

<div align='center'>
  <img src=https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg >
  <img src=https://img.shields.io/github/stars/pprp/Awesome-LLM-Prune.svg?style=social >
  <img src=https://img.shields.io/github/watchers/pprp/Awesome-LLM-Prune.svg?style=social >
  <img src=https://img.shields.io/badge/Release-v0.1-brightgreen.svg >
  <img src=https://img.shields.io/badge/License-GPLv3.0-turquoise.svg >
 </div>   
This repository is dedicated to the pruning of large language models (LLMs). It aims to serve as a comprehensive resource for researchers and practitioners interested in the efficient reduction of model size while maintaining or enhancing performance.

We encourage the community to engage with this resource; please leave comments on issues related to papers you’re interested in or corrections where misunderstandings may occur. For further inquiries or to contribute to this project, feel free to submit a pull request or open an issue.

| Taxonomy | Unstructured                                                 | Structured                                                  | Semi-Structured                                              | Benchmark                                                    |
| -------- | ------------------------------------------------------------ | ----------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Label    | <img src=https://img.shields.io/badge/unstructured-turquoise.svg > | <img src=https://img.shields.io/badge/structured-blue.svg > | <img src=https://img.shields.io/badge/semi_structured-brightgreen.svg > | <img src=https://img.shields.io/badge/benchmark-purple.svg > |



- SparseGPT: Massive Language Models Can be Accurately Pruned in One-shot 
    - Label: <img src=https://img.shields.io/badge/unstructured-turquoise.svg ><img src=https://img.shields.io/badge/semi_structured-brightgreen.svg >
    - Author: Elias Frantar, Dan Alistarh
    - Link: https://arxiv.org/pdf/2301.00774.pdf 
    - Code: https://github.com/IST-DASLab/sparsegpt 
    - Pub: ICML 2023
    - Summary: First to prune GPT with at least 50% sparsity without any training. SparseGPT is entirely local, which only focus on weight updates without any global gradient information. 

- Wanda: A Simple and Effective Pruning Approach For Large Language Models
    - Label: <img src=https://img.shields.io/badge/unstructured-turquoise.svg ><img src=https://img.shields.io/badge/structured-blue.svg >
    - Author: Mingjie Sun, Zhuang Liu, Anna Bair, etc.
    - Link: https://arxiv.org/pdf/2306.11695.pdf 
    - Code: https://github.com/locuslab/wanda
    - Pub: ICML23 workshop 
    - Summary: Wanda simplify the SparseGPT with approximation thus just rely on weight and activation to compute the pruning metric. Wanda can be seen as a simplified version of SparseGPT, as it simplify the Hessian approximation, reducing computation greatly. 

- Pruner-Zero: Evolving Symbolic Pruning Metric 

    - Label: <img src=https://img.shields.io/badge/unstructured-turquoise.svg ><img src=https://img.shields.io/badge/structured-blue.svg >
    - Author: Peijie Dong, Lujun Li, Zhenheng Tang, Xiang Liu, Xinglin Pan, Qiang Wang, Xiaowen Chu
    - Link: [arxiv.org/pdf/2406.02924v1](https://arxiv.org/pdf/2406.02924v1) 
    - Code: [pprp/Pruner-Zero: Evolving Symbolic Pruning Metric from scratch (github.com)](https://github.com/pprp/Pruner-Zero)
    - Pub: ICML24 
    - Summary: Pruner-Zero formulates the pruning metric as a symbolic discovery problem. They develop an automatic framework for searching symbolic pruning metrics using genetic programming. They model the pruning metric as tree-based symbols and employ genetic programming to automatically identify the optimal candidate symbolic pruning metric. Experiments on LLaMA, LLaMA-2, OPT demonstrate the superiority of Pruner-Zero. 

- LLM-Kick: Compressing LLMs: The Truth Is Rarely Pure and Never Simple
    - Label: <img src=https://img.shields.io/badge/benchmark-purple.svg >
    - Author: Ajay Jaiswal, Zhe Gan, etc
    - Link: https://arxiv.org/pdf/2310.01382.pdf
    - Code: https://github.com/VITA-Group/llm-kick
    - Pub: ICLR 2024
    - Summary: Re-define the evaluation protocol for compressed LLMs; Observation: SoTA Pruning methods suffer significant performance degradation, despite negligible changes in perplexity. SoTA Pruning do not work well for N:M structured pruning. Quantization methods are more successful.
    - Comment: This paper question the performance of LLM after pruning, which provide us a new perspective besides pure perplexity. This paper is worth reading because its evaluation is comprehensive. 

- RIA: Plug-and-Play: An Efficient Post-Training Pruning Method for Large Language Models
    - Label: <img src=https://img.shields.io/badge/unstructured-turquoise.svg ><img src=https://img.shields.io/badge/semi_structured-brightgreen.svg >
    - Author: Yingtao Zhang, Haoli Bai, Haokun Lin, Jialin Zhao, Lu Hou, Carlo Vittorio Cannistraci
    - Link: https://openreview.net/pdf?id=Tr0lPx9woF
    - Code: https://github.com/biomedical-cybernetics/Relative-importance-and-activation-pruning
    - Pub: ICLR 2024
    - Summary: For post-training pruning method, this paper proposed two innovative and plug-and-play components, which is Relative Importance and Activations (RIA) and Channel Permutation (CP). (1) RIA re-evaluate the importance of each weight element based on all connections that originate from input and output. (2) CP aims to preserve important weights under N:M sparsity, which yields better N:M structures by permuting the input channels of weight.
    - Comment: I have thoroughly reviewed the source code and can affirm its effectiveness. The code is indeed of superior quality, demonstrating excellent standards in development. 

- Pruning Large Language Models via Accuracy Predictor
    - Label: <img src=https://img.shields.io/badge/unstructured-turquoise.svg >
    - Author: Yupeng Ji, Yibo Cao, Jiucai Liu 
    - Link: https://arxiv.org/pdf/2309.09507.pdf 
    - Code: Not available 
    - Pub: Arxiv 
    - Summary: Formulate the pruning LLM as NAS problem. The search space is the prunining ratio, layer type, etc. By utilizing GBDT accuracy predictor, this paper take the layer-wise importance as input and predict the PPL. 
    - Comment: With 525 architecture-accuracy pair, this paper train the GBDT with 7:3 ratio.

- LLM-Pruner: On the Strucutal Pruning of Large Language Models 
    - Label: <img src=https://img.shields.io/badge/structured-blue.svg >
    - Author: Xinyin Ma, Gongfan Fang, Xinchao Wang 
    - Link: https://arxiv.org/pdf/2305.11627.pdf 
    - Code: https://github.com/horseee/LLM-Pruner
    - Pub: NeurIPS 2023 
    - Summary: This paper endeavor find the copuled structures (Dependency Graph) in LLaMA and proposed Groupded Importance Estimation like Vector-wise, Element-wise, and Group Importance. 
    - Comment: Impressive work. This work is similar to MMRazor, which can handle CNN-based model. 

- The Emergence of Essential Sparsity in Large Pre-trained Models: The Weights that Matter
    - Label: <img src=https://img.shields.io/badge/benchmark-purple.svg >
    - Author: Ajay Jaiswal, Shiwei Liu, Tianlong Chen, Zhangyang Wang
    - Link: https://arxiv.org/pdf/2306.03805.pdf
    - Code: https://github.com/VITA-Group/essential_sparsity
    - Pub: NeurIPS 2023 
    - Summary: This paper proposes the existence of – “essential sparsity” defined with a sharp dropping point beyond which the performance declines much faster w.r.t the rise of sparsity level, when we directly remove weights with the smallest magnitudes in one-shot.



- Compresso: Structured Pruning with Collaborative Prompting Learns Compact Large Language Models
    - Label: <img src=https://img.shields.io/badge/structured-blue.svg >
    - Author: Song Guo, Jiahang Xu, Li Lyna Zhang, Mao Yang 
    - Link: https://arxiv.org/pdf/2310.05015.pdf 
    - Code: https://github.com/microsoft/Moonlit/tree/main/Compresso
    - Pub: Under Review 
    - Summary: Combing instruction tuning with training-based Pruning. LoRA is incorporated to achieve memory-efficient. Collaborative pruning prompt encourage LLMs to better align with the pruning algorithm. 
    - Comment: The prompt is really interesting, which is "Attention! LLM".

- The Truth is in There: Improving Reasoning in Language Models with Layer-Selective Rank Reduction
  - Label: <img src=https://img.shields.io/badge/structured-blue.svg >
  - Author: Pratyusha Sharma, Jordan T. Ash, Dipendra Misra 
  - Link: https://arxiv.org/pdf/2312.13558.pdf 
  - Code: Not available 
  - Pub: ICLR Under review 
  - Summary: This paper is not related to Pruning but to Low-rank decomposition. They find that removing higher-order component of weight matrics in MLP and attention can significantly improve the performance of LLMs.

- Outlier Weighed Layerwise Sparsity (OWL): A Missing Secret Sauce for Pruning LLMs to High Sparsity
  - Label: <img src=https://img.shields.io/badge/unstructured-turquoise.svg >
  - Author:Lu Yin, You Wu, Zhenyu Zhang, Cheng-Yu Hsieh, Yaqing Wang, Yiling Jia, Mykola Pechenizkiy, Yi Liang, Zhangyang Wang, Shiwei Liu
  - Link:https://arxiv.org/abs/2310.05175
  - Code:https://github.com/luuyin/OWL 
  - Pub: ICML 2024
  - Summary: OWL challenges the assumption of uniform layer-wise assumption and tries to assign different layers with different pruning ratio by proposed OWL metric.



- The LLM Surgeon
	
	- Label: <img src=https://img.shields.io/badge/unstructured-turquoise.svg >
	- Author:Tycho F.A. van der Ouderaa, Markus Nagel, Mart van Baalen, Yuki M. Asano, Tijmen Blankevoort
	- Link:https://arxiv.org/pdf/2312.17244
	- Pub: ICLR24
	- Summary: This paper scales Kronecker-factored curvature approximations of the target loss landscape to large language models. The metric for this paper is Fisher information matrix. 
- Shortened LLaMA: A Simple Depth Pruning for Large Language Models

  - Label: <img src=https://img.shields.io/badge/structured-blue.svg >
  - Authors: Bo-Kyeong Kim, Geonmin Kim, Tae-Ho Kim, Thibault Castells, Shinkook Choi, Junho Shin, Hyoung-Kyu Song
  - Link: https://arxiv.org/abs/2402.02834 
  - Pub: ICLR24 Workshop (ME-FoMo) 
  - Summary: This paper delves into the naive implementation of structured pruning, specifically Depth Pruning, for Large Language Models (LLMs). Through experiments on zero-shot tasks, it is revealed that its performance is on par with width pruning techniques. However, the pruning ratio remains constrained to less than 35% (20%, 27%, 35%), and the performance on wikitext-2 (PPL) is somewhat less favorable compared to wanda. Nonetheless, this study demonstrates the feasibility of pruning by eliminating layers with lower block-level importance scores. Moreover, performance enhancement is observed after one-shot pruning via LoRA fine-tuning.
- SliceGPT: Compress Large Language Models by Deleting Rows and Columns 

  - Label: <img src=https://img.shields.io/badge/structured-blue.svg >
  - Author: Saleh Ashkboos, Maximilian L. Croci, Marcelo Gennari do Nascimento, Torsten Hoefler, James Hensman
  - Link: https://arxiv.org/abs/2401.15024 
  - Pub: ICLR24
  - Summary: This paper focuses on structured pruning by removing rows and columns of a matrix to reduce parameters. However, this idea is similar to LLM-Pruner but weaker. The organization of this paper is somewhat peculiar: it dedicates two and a half pages to related works (too long). Additionally, in Table 1, SliceGPT (<30% sparsity) mainly compares its performance with SparseGPT under 2:4 structure pruning settings (50% sparsity), which is not quite fair. Please correct me if I am wrong.
- PERP: Rethinking the Prune-Retrain Paradigm in the Era of LLMs 

  - Label: <img src=https://img.shields.io/badge/unstructured-turquoise.svg >
  - Author: Max Zimmer, Megi Andoni, Christoph Spiegel, Sebastian Pokutta
  - Link: https://arxiv.org/pdf/2312.15230.pdf
  - Pub: Arxiv 
  - Summary: In the era of Large Language Models (LLMs), retraining becomes impractical due to memory and compute constraints. This paper proposes the use of Low-Rank Adaption to mitigate the expense of the retraining process. They explore four approaches, including BN-Recalibration, Biases, BN-Parameters, and Linear Probing. However, it's worth noting that most LLMs do not utilize Batch Normalization (BN). Indeed, this paper only conducts a few experiments on OPT and primarily focuses on works such as ResNet50 pruning. Furthermore, LoRA + Pruning is actually a component of SparseGPT (published in January 2023), so the novelty of this paper is somewhat limited.
- Structural pruning of large language models via neural architecture search

  - Label: <img src=https://img.shields.io/badge/structured-blue.svg >
  - Author:Aaron Klein, Jacek Golebiowski, Xingchen Ma, Valerio Perrone, Cedric Archambeau 
  - Link: https://openreview.net/pdf?id=SHlZcInS6C
  - Pub: AutoML 
  - Summary: This paper discuss the relationship between NAS and structural pruning and employ multi-objective NAS to compress LLM. They view the pre-trained network as a super-network and search for the best sub-network that optimally balance between downstream tasks and parameter count. For training weight-sharing NAS, they employ sandwich rule to train sub-networks. After training, local search is utilized for finding the best sub-network.
- Not all Layers of LLMs are Necessary during Inference 

  - Label: <img src=https://img.shields.io/badge/structured-blue.svg >
  - Author: Siqi Fan, Xin JIang, Xiang Li, Xuying Meng, Peng Han, Shuo Shang, Aixin Sun, Yequan Wang, Zhongyuan Wang
  - Link: https://arxiv.org/pdf/2403.02181.pdf 
  - Pub: CoRR 
  - Summary: This paper analyse the activated layers across tasks and propose AdaInfer to determine the inference termination moments based on the input instance. Thus, they can use shallow layers for easy instance and deep layers for hard ones. In general, this technique can be treated as an early stopping strategy. The early stop signal is generated by two components: Feature Selection Module that crafts feature vector for current input instance; Classifier that utilize SVM or CRF to access the strength of stopping signal.
- ShortGPT: Layers in Large Language Models are More Redundant Than You Expect

  - Label: <img src=https://img.shields.io/badge/structured-blue.svg >
  - Author: Xin Men, Mingyu Xu, Qingyu Zhang, Bingning Wang, Hongyu Lin, Yaojie Lu, Xianpei Han, Weipeng Chen
  - Link: https://arxiv.org/abs/2403.03853 
  - Pub: CoRR 
  - Summary: They discovered that the layers of LLMs exhibit high similarity and some layers are negligible. To remove the unimportant layers, they define a metric called Block Influence (BI) to gauge the significance of each layers in LLMs. Specifically, the BI score is actually the cosine similarity of two successive blocks. The experiments are limited as they didn't provide the results of ppl and there are various one-shot pruning for LLMs like SparseGPT and Wanda etc.
- LaCo: Large Language Model Pruning via Layer Collapse

  - Label: <img src=https://img.shields.io/badge/structured-blue.svg >
  - Author:Yifei Yang, Zouying Cao, Hai Zhao
  - Link: https://arxiv.org/pdf/2402.11187.pdf
  - Pub: CoRR 
  - Summary: These paper compute the different among layers (call it Reserving-Difference) and merge them (Call it Seeking-Common). Specifically, they merge m consecutive layers into one by using sum of parameter difference. Also, they employ trial-and-error by evaluating each merged  model with Cosine Similarity and make adjustment of the merge.
  - Comments: There is a lack of explanation of equation-1. Why it worked?
- Shortened LLaMA: A Simple Depth Pruning for Large Language Models 

  - Label: <img src=https://img.shields.io/badge/structured-blue.svg >
  - Author: Bo-Kyeong Kim, Geonmin Kim, Tae-Ho Kim, Thibault Castells, Shinkook Choi, Junho Shin, Hyoung-Kyu Song 
  - Link:https://arxiv.org/abs/2402.02834 
  - Pub: CoRR
  - Summary: This paper focus on depth pruning and analysis the influence of depth and width pruning on LLM inference efficiency. They explore various design factors including the choice of prunable units, the criteria and retraining frequency. (1) Prunable Units: width and depth; (2) Criteria: Magnitude, Taylor, Mag+ and Talyor+, PPL; (3)retrain: LoRA. Finally, they choose PPL as criteria and target Depth Pruning. They claim that depth pruning approach can compte with recent width pruning methods  on Zero-shot tasks performance.
- FLAP: Fluctuation-based adaptive structured pruning for large language models

  - Label: <img src=https://img.shields.io/badge/structured-blue.svg >
  - Author: Yongqi An, Xu Zhao, Tao Yu, Ming Tang, Jinqiao Wang
  - Link: https://arxiv.org/abs/2312.11983 
  - Code: https://github.com/CASIA-IVA-Lab/FLAP 
  - Pub: AAAI 24
  - Summary: They proposed a retraining-free structured pruning framework for LLMs. (1) Structured Importance Metric: Fluctuation Pruning Metric (2) Adaptively Search Global Compress Ratio: (3) Compensation Mechanism: add additional biases to recover the output feature maps. Specifically, they observe that certain channels of hidden state features exhibits a low variation across different samples, indicating that if their corresponding input feature channels are pruned, the resulted change can be counterbalanced by the baseline value. Compared with Wanda, FLAP compute the sample variance of each input feature and weight it with the squared norm of the corresponding column of the weight matrics. 
  - Comment: This paper is well-written and the framework is clear. However, I have a question: they claim FLAP is a retraining-free framework but it still require retraining the biases.
- Bonsai: Everybody Prune Now: Structured Pruning of LLMs with only Forward Passes

  - Label: <img src=https://img.shields.io/badge/structured-blue.svg >
  - Author: Lucio Dery, Steven Kolawole, Jean-François Kagy, Virginia Smith, Graham Neubig, Ameet Talwalkar
  - Link: [arxiv.org/pdf/2402.05406.pdf](https://arxiv.org/pdf/2402.05406.pdf)
  - Code: https://github.com/ldery/Bonsai 
  - Summary: This work devoted to structured pruning of LLMs using only forward passes (gradient-free way). Bonsai can outperform gradient-based structured pruning methods and twice as fast as semi-structured pruning methods. Specifically, Bonsai measures the performance of each module's performance by generating sub-models, which require multiple forwards. Also, Bonsai use informative priors (Deep compression, a.k.a other unstructured pruning method) to drop modules. Bonsai adopts iterative pruning method. In each iteration, it will assess the prior of unpruned module and utilize them to select new sub-model.
- The Unreasonable Ineffectiveness of the Deeper Layers 

  - Label: <img src=https://img.shields.io/badge/structured-blue.svg >
  - Author: Andrey Gromov, Kushal Tirumala, Hassan Shapourian, Paolo Glorioso, Daniel A. Roberts
  - Link: https://arxiv.org/pdf/2403.17887v1.pdf
  - Pub: Arxiv 
  - Summary: This paper aims to layer-pruning (structured pruning) by identify the optimal block of layers to prune by considering the similarity across layers. To recover performance, QLoRA is employed to make all experiments can be conducted on a A100. This paper claims that the shallow layers plays a more critical role than deeper layers of network. 
  - Comment: good reference for studying the depth-dependence of neural networks. 
- SLEB: Streamlining LLMs through Redundancy Verification and Elimination of Transformer Blocks

    - Label: <img src=https://img.shields.io/badge/structured-blue.svg >
    - Author: Jiwon Song, Kyungseok Oh, Taesu Kim, Hyungjun Kim, Yulhwa Kim, Jae-Joon Kim
    - Link: https://arxiv.org/pdf/2402.09025.pdf
    - Code: [https://github.com/leapingjagg-dev/SLEB](https://github.com/leapingjagg-dev/SLEB?tab=readme-ov-file)
    - Pub: Arxiv
    - Summary: This paper streamlines LLMs by identifying and removing redundant blocks. Specifically, cosine similarity is utilized to analyze the redundancy. Another metric3 is proposed for removing blocks.
    - Comment: There should be more methods for comparison.
- Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning

  - Label: <img src=https://img.shields.io/badge/structured-blue.svg >
  - Tag: Structured Pruning
  - Author: Mengzhou Xia, Tianyu Gao, Zhiyuan Zeng, Danqi Chen;
  - Link: https://arxiv.org/abs/2310.06694
  - Code: https://github.com/princeton-nlp/LLM-Shearing
  - Pub: ICLR 2024 
  - Summary: To prune larger pre-trained model, this paper proposed (1) Targeted structured pruning: prune a LLM to specified target shape by removing layers, heads, and intermediate and hidden dimensions in an end-to-end manner; (2) Dynamic Batch Loading: update the composition of sampled data in each training batch based on varying losses across different domains.
- Gradient-Free Adaptive Global Pruning for Pre-trained Language Models

    - Label: <img src=https://img.shields.io/badge/unstructured-turquoise.svg >
    - Author: Guangji Bai, Yijiang Li, Chen Ling, Kibaek Kim, Liang Zhao
    - Link: https://arxiv.org/pdf/2402.17946v1.pdf
    - Pub: Arxiv
    - Code: https://github.com/BaiTheBest/AdaGP
    - Summary: Due to the size of LLM, global pruning becomes impractical. However, local pruning often leads to suboptimal solutions. To address this issue, this paper propose Adaptive Global Pruning (AdaGP) to redefine the global pruning process into manageable, coordinated subproblems, allowing for resource-efficient optimization with global optimality.
- NutePrune: Efficient Progressive Pruning with Numerous Teachers for Large Language Models

    - Label: <img src=https://img.shields.io/badge/unstructured-turquoise.svg >
    - Author: Shengrui Li, Xueting Han, Jing Bai
    - Link: https://arxiv.org/pdf/2402.09773.pdf
    - Pub: Arxiv
    - Code: Not available
    - This work = structure pruning + progressive knowledge distillation; However, due to the memory constraints, knowledge distillation is hard in the context of LLM. To mitigate the memory cost, this paper propose to switch teacher and student by apply different sparsity ratio using various masks and LoRA modules.
- BESA: Pruning Large Language Models with Blockwise Parameter-Efficient Sparity Allocation

    - Label: <img src=https://img.shields.io/badge/unstructured-turquoise.svg >
    - Author: Peng Xu, Wenqi Shao, Mengzhao Chen, Shitao Tang, Kaipeng Zhang, Peng Gao, Fengwei An, Yu Qiao, Ping Luo.
    - Link: https://openreview.net/pdf?id=gC6JTEU3jl
    - Code: https://github.com/LinkAnonymous/BESA
    - Pub: ICLR 2024
    - Summary: Existing pruning methods for LLM adopted a layer-wise approach but resulted in significant perturbation to the model’s output and required meticulous hyperparameter tuning(Pruning Ratio). This paper proposes BESA to handle it with block-wise adaptation. (1) Instead of pruning each Linear layer, BESA targets the overall pruning error w.r.t. one transformer block (2) it allocates layer-specific sparsity in a differentiable manner.

- Fast and Optimal Weight Update for Pruned Large Language Models
    - Label: <img src=https://img.shields.io/badge/unstructured-turquoise.svg ><img src=https://img.shields.io/badge/semi_structured-brightgreen.svg >
    - Authors: **Vladim ́ır Bozˇa**
    - Link: https://arxiv.org/pdf/2401.02938.pdf
    - Code: Not available
    - Code: https://github.com/fmfi-compbio/admm-pruning
    - Summary:  This paper focuses on the recovery process, which was first proposed in SparseGPT. This paper proposed an Alternating Direction Method of Multipliers (ADMM), with a simple iterative pruning mask selection.

- COPAL: Continual Pruning in Large Language Generative Models
    - Label: <img src=https://img.shields.io/badge/unstructured-turquoise.svg ><img src=https://img.shields.io/badge/semi_structured-brightgreen.svg >
    - Authors: Srikanth Malla, Joon Hee Choi, Chiho Choi
    - Link: https://arxiv.org/pdf/2405.02347v1.pdf
    - Code: Not Available
    - Summary: This paper introduces COPAL, an algorithm for continual pruning of large language models under a model adaptation setting. The approach utilizes sensitivity analysis to guide the pruning process, enhancing model adaptability and computational efficiency without the need for retraining. The empirical evaluation demonstrates COPAL's effectiveness in maintaining performance across various datasets and model sizes.

- DaSS: Dependency-Aware Semi-Structured Sparsity of GLU Variants in Large Language Models
    - Label: <img src=https://img.shields.io/badge/semi_structured-brightgreen.svg >
    - Authors: Zhiyu Guo, Hidetaka Kamigaito, Taro Wanatnabe
    - Link: https://arxiv.org/pdf/2405.01943v1.pdf
    - Code: Not available
    - Summary: This paper introduces Dependency-aware Semi-structured Sparsity (DaSS), a novel pruning method for SwiGLU-based Large Language Models (LLMs). DaSS integrates structural dependency into weight magnitude-based pruning, using an MLP-specific pruning metric that evaluates the importance of each weight by considering both its magnitude and the corresponding MLP intermediate activation norms. The method offers a balance between unstructured pruning flexibility and structured pruning consistency, achieving hardware-friendly N:M sparsity patterns. Empirical results show DaSS outperforms SparseGPT and Wanda in various tasks while maintaining computational efficiency.

- Structural Pruning of Pre-trained Language Models via Neural Architecture Search
    - Label: <img src=https://img.shields.io/badge/structured-blue.svg >
    - Authors: Aaron Klein, Jacek Golebiowski, Xingchen Ma, Valerio Perrone, Cedric Archambeau
    - Link: https://arxiv.org/pdf/2405.02267v1.pdf
    - Code: Not available
    - Summary: This paper explores the use of Neural Architecture Search (NAS) for structural pruning of pre-trained language models to address the challenges of high GPU memory requirements and inference latency. The authors propose a multi-objective approach that identifies the Pareto optimal set of sub-networks, enabling a flexible compression process without the need for retraining. The method leverages weight-sharing NAS techniques to accelerate the search for efficient sub-networks. Empirical evaluations demonstrate that their approach outperforms baseline models in terms of efficiency and adaptability, offering a promising strategy for deploying large language models in real-world applications.
    - Note: All experiments are conducted on BERT not LLAMA. This NAS procedure requires massive computation when applying to LLaMA.


- Pruning Small Pre-Trained Weights Irreversibly and Monotonically Impairs "Difficult" Downstream Tasks in LLMs
    - Label: <img src=https://img.shields.io/badge/benchmark-purple.svg >
    - Authors: Lu Yin, Ajay Jaiswal, Shiwei Liu, Souvik Kundu, Zhangyang Wang
    - Link: https://arxiv.org/pdf/2310.02277v2.pdf
    - Code: https://github.com/VITA-Group/Junk_DNA_Hypothesis.git
    - Summary: The paper presents the "Junk DNA Hypothesis," which challenges the notion that **small-magnitude weights in large language models (LLMs) are redundant and can be pruned without performance loss**. Contrary to common beliefs, the study argues that these weights encode essential knowledge for difficult downstream tasks. The authors demonstrate a monotonic relationship between the performance drop of downstream tasks and the magnitude of pruned weights, indicating that pruning can cause irreversible knowledge loss, even with continued training. The paper also contrasts pruning with quantization, showing that the latter does not exhibit the same monotonic effect on task difficulty. The findings suggest that small-magnitude weights are crucial for complex tasks and cannot be simply discarded. The study provides insights into the role of these weights and implications for LLM compression techniques.
