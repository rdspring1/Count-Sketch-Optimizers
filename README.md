# Count-Sketch Optimizers
[Compressing Gradient Optimizers via Count-Sketches](http://proceedings.mlr.press/v97/spring19a/spring19a.pdf)

An ICML 2019 paper by Ryan Spring, Anastasios Kyrillidis, Vijai Mohan, Anshumali Shrivastava

# BERT-Large Training Results
Trained with Activation Checkpointing and Mixed Precision Training (FP16) on Nvidia V100 DGX-1 servers

| BERT-Large           | Adam           | Count-Min Sketch (CMS) - RMSprop |
| -------------------- | -------------- | -------------------------------- |
| Time (Days)          | **5.32**       | 5.52                             |
| Size (MB)            | 7,097          | **5,133**                        |
| Test Perplexity      | **4.04**       | 4.18                             |

![Convergence Rate - Adam, CMS-RMSprop](/paper/BERT_Large_Convergence.png)
![Faster convergence rate with larger batch size - CMS-RMSprop](/paper/BERT_Large_Batch_Size.png)

# Instructions
1. Install Requirements
2. Add optimizers folder to $PYTHONPATH

# Requirements
1. torch
2. torchvision
3. cupy
4. pynvrtc

# Examples
1. ImageNet - ResNet-18
2. LM1B - Transformer / LSTM
3. Wikitext-2 - LSTM

# Dense Layer Support
We support compressing the dense layers of the neural network without update sparsity. During training, we update the auxiliary variables and perform the gradient update for each parameter in a single fused CUDA kernel. The dense kernel is equivalent to the sparse kernel. The main difference is that we explicitly avoid generating the auxiliary variables for the dense layers in global memory. Instead, we access them inside the shared memory of the GPU Streaming Multiprocessor. Without this key feature, our approach would not save any GPU memory for the dense layers. In the sparse case, we assume that the non-zero gradient updates is significantly smaller than the auxiliary variable. (See dense\_exp\_cms.py for more details)

# References
1. [Transformer Architecture - Nvidia Megatron Language Model](https://github.com/NVIDIA/Megatron-LM)
2. [Compressing Gradient Optimizers via Count-Sketches (ICML 2019)](http://proceedings.mlr.press/v97/spring19a.html)
