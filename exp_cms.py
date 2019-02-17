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
__inline__ __device__
float minimum(float a, float b, float c)
{
	return fminf(fminf(a,b),c);
}

extern "C"
__inline__ __device__
float update_retrieve(float* mem,
	float* result,
	const float beta,
	const int N,
	const int D,
	const long index,
	const float value)
{
	int a = 994443;
	int b = 609478;
	const int hash_idx = hash(index, N, a, b) * D + threadIdx.x;
	float old_value = mem[hash_idx];
	float update = (1. - beta) * (value - old_value);
	atomicAdd(&mem[hash_idx], update);
	return old_value + update;
}

extern "C"
__inline__ __device__
float cms_update_retrieve(float* mem,
	float* result,
	const float beta,
	const int N,
	const int W,
	const int D,
	const long index,
	const float value)
{
	float r[3];
	int a[3] = {994443, 4113759, 9171025};
	int b[3] = {609478, 2949676, 2171464};
	for(int idx = 0; idx < 3; ++idx)
	{
		const int hash_idx = idx*W + hash(index, N, a[idx], b[idx]) * D + threadIdx.x;
		float old_value = mem[hash_idx];
		float update = (1. - beta) * (value - old_value);
		atomicAdd(&mem[hash_idx], update);
		r[idx] = old_value + update;
	}
	return minimum(r[0], r[1], r[2]);
}

extern "C"
__global__
void cms_hash_update_retrieve(const long* indices,
	const float* values,
	const float* beta,
	float* mem,
	float* result,
	const int N,
	const int W,
	const int D)
{
	if(threadIdx.x < D)
	{
		const int idx = blockIdx.x * D + threadIdx.x;
		const float value = values[idx];
		const long index = indices[blockIdx.x];
		result[idx] = cms_update_retrieve(mem, result, *beta, N, W, D, index, value);
	}
}
'''

class CountMinSketch:
    def __init__(self, N, D, sketch_size=0.20):
        self.N = N
        self.D = D
        self.blk_size = math.ceil(D // 32) * 32
        self.range = int(N*sketch_size/3.)
        self.width = self.range * D
        self.kernel = cupyKernel(kernel, "cms_hash_update_retrieve")
        self.cms = torch.zeros(3, self.range, D).float().cuda()
        print(N, "CMS", self.cms.size())

    def update(self, indices, values, size, beta):
        M, D = values.size()
        result = torch.zeros(values.size()).float().cuda()
        beta = torch.FloatTensor([beta]).cuda()
        self.kernel(grid=(M,1,1),
                block=(self.blk_size,1,1),
                args=[indices.data_ptr(),
                     values.data_ptr(),
                     beta.data_ptr(),
                     self.cms.data_ptr(),
                     result.data_ptr(),
                     self.range,
                     self.width,
                     self.D],
                strm=torch.cuda.current_stream().cuda_stream)
        return torch.cuda.sparse.FloatTensor(indices, result, size)

    def clean(self, alpha):
        self.cms.mul_(alpha)
