# Count-Sketch Optimization
Memory-Constrained Optimization via Count-Sketches

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
[Transformer Architecture - Nvidia Sentiment Discovery](https://github.com/NVIDIA/sentiment-discovery)
