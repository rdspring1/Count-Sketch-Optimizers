import torch
from cupy_kernel import cupyKernel
import math
import numpy as np

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
float fh_update_retrieve(float* mem,
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
	float new_value = beta * mem[hash_idx] + value;
	mem[hash_idx] = new_value;
	return new_value;
}

extern "C"
__inline__ __device__
float median(float a, float b, float c)
{
	return fmaxf(fminf(a,b), fminf(fmaxf(a,b),c));
}

extern "C"
__inline__ __device__
float cms(float* mem,
	const float beta,
	const int N,
	const int D,
	const long index,
	const float value,
	const int a,
	const int b,
	const int offset)
{
	const int hash_idx = offset + hash(index, N, a, b) * D + threadIdx.x;
	const bool sign_bit = hash(index, N, a+2, b+3) & 0x1;
	const float sign = (sign_bit) ? 1.0 : -1.0;

	float old_value = sign * mem[hash_idx];
	float update = (beta - 1.0f) * old_value + value;
	atomicAdd(&mem[hash_idx], sign * update);
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
		r[idx] = cms(mem, beta, N, D, index, value, a[idx], b[idx], idx*W);
	}
	return median(r[0], r[1], r[2]);
}

extern "C"
__global__
void hash_update_retrieve(const long* indices,
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


class CountSketch:
    def __init__(self, N, D, sketch_size=0.20):
        self.N = N
        self.D = D
        self.blk_size = math.ceil(D // 32) * 32
        self.range = int(N*sketch_size/3.)
        self.width = self.range * D
        self.kernel = cupyKernel(kernel, "hash_update_retrieve")
        self.sketch = torch.zeros(3, self.range, D).float().cuda()
        print(N, "Count Sketch", self.sketch.size())

    def schedule(self, rate=425000, maximum=0.990):
        value = 1. + math.log(math.floor(self.t/rate)+1, 2)
        current = 1. - pow(2.0, -value)
        self.t += 1
        if self.t % rate == 0:
            print("Momentum:", current)
        return min(current, maximum)

    def update(self, indices, values, size, beta):
        M, D = values.size()
        result = torch.zeros(values.size()).float().cuda()
        beta = torch.FloatTensor([beta]).cuda()
        self.kernel(grid=(M,1,1),
                block=(self.blk_size,1,1),
                args=[indices.data_ptr(),
                    values.data_ptr(),
                    beta.data_ptr(),
                    self.sketch.data_ptr(),
                    result.data_ptr(),
                    self.range,
                    self.width,
                    self.D],
                strm=torch.cuda.current_stream().cuda_stream)
        return torch.cuda.sparse.FloatTensor(indices, result, size)
