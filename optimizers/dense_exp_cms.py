import torch
from cupy_kernel import cupyKernel
import numpy as np
import math

kernel = '''
extern "C"
__inline__ __device__
int hash(int value, int range, int a, int b)
{
	int h = a * value + b;
	h ^= h >> 16;
	h *= 0x85ebca6b;
	h ^= h >> 13;
	h *= 0xc2b2ae35;
	h ^= h >> 16;
	return h % range;
}

extern "C"
__global__
void dense_cms_update(float* parameter,
	float* gradient,
	float* mem,
	const float* lr_ptr,
	const float* beta_ptr,
	const int N,
	const int D,
	const int W)
{
        const float lr = *lr_ptr;
        const float beta = *beta_ptr;
	const int offset = blockIdx.x * D;
	const int a = 994443;
	const int b = 609478;

	// Read auxiliary variables
	extern __shared__ float shared[];
	float* aux = (float*) &shared[0];
	float* acc = (float*) &shared[W];

	for(int index = threadIdx.x; index < W; index += blockDim.x)
	{
		const int global_index = blockIdx.x * W + index;
		aux[index] = mem[global_index];
		acc[index] = 0.0f;
	}
	__syncthreads();

	for(int index = threadIdx.x; index < D; index += blockDim.x)
	{
		// Read chunk from parameters, gradient
		float p = parameter[offset + index];
		float g = gradient[offset + index];
		float value = powf(g, 2);

		// Calculate auxiliary variable approximation
		const int hash_idx = hash(offset + index, W, a, b);
		float v = beta * aux[hash_idx] + (1. - beta) * value;

		// Perform parameter update
		float update = lr * g * rsqrtf(v + 1e-10);
		atomicAdd(&parameter[offset + index], update);

		// Update Accumulate Register
		acc[hash_idx] += value;
		__syncthreads();
	}

	// Update Auxiliary variables
	for(int index = threadIdx.x; index < W; index += blockDim.x)
	{
		const float global_update = (1. - beta) * (acc[index] - aux[index]);
		const int global_index = blockIdx.x * W + index;
		atomicAdd(&mem[global_index], global_update);
	}
}
'''

class DenseCMS:
    def __init__(self, N, D, sketch_size=0.20):
        self.N = N
        self.D = D
        self.blk_size = 32
        self.range = max(int(D*sketch_size), 1)
        self.kernel = cupyKernel(kernel, "dense_cms_update")
        self.cms = torch.zeros(N, self.range).float().cuda()
        print(N, "Dense CMS", self.cms.size())

    def update(self, p, g, lr, beta):
        lr = torch.FloatTensor([lr]).cuda()
        beta = torch.FloatTensor([beta]).cuda()

        # shared memory - #copies x #elements x sizeof(float)
        self.kernel(grid=(self.N,1,1),
                block=(self.blk_size,1,1),
                args=[p.data_ptr(),
                     g.data_ptr(),
                     self.cms.data_ptr(),
                     lr.data_ptr(),
                     beta.data_ptr(),
                     self.N,
                     self.D,
                     self.range],
                strm=torch.cuda.current_stream().cuda_stream,
                smem=int(8*self.range))

    def clean(self, alpha):
        self.cms.mul_(alpha)
