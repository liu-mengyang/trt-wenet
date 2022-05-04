#include "LayerNormPlugin.h"

using namespace nvinfer1;

PluginFieldCollection    LayerNormPluginCreator::fc_ {};
std::vector<PluginField> LayerNormPluginCreator::attr_;

template<typename T>
__inline__ __device__ T Div(T a, T b);

template<>
__inline__ __device__ float Div<float>(float a, float b) {
  return a / b;
}

/* https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
# For a new value newValue, compute the new count, new mean, the new M2.
# mean accumulates the mean of the entire dataset
# M2 aggregates the squared distance from the mean
# count aggregates the number of samples seen so far
def update(existingAggregate, newValue):
    (count, mean, M2) = existingAggregate
    count += 1
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2
    return (count, mean, M2)

# Retrieve the mean, variance and sample variance from an aggregate
def finalize(existingAggregate):
    (count, mean, M2) = existingAggregate
    if count < 2:
        return float("nan")
    else:
        (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
        return (mean, variance, sampleVariance)
*/
template<typename T> // 迭代式Welford
inline __device__ void WelfordCombine(T val, T* mean, T* m2, T* count) {
  // Use Welford Online algorithem to compute mean and variance
  // For more details you can refer to:
  // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
  *count += 1;
  T delta1 = val - *mean;
  *mean += Div(delta1, *count);
  T delta2 = val - *mean;
  *m2 += delta1 * delta2;
}

/* https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
def parallel_variance(n_a, avg_a, M2_a, n_b, avg_b, M2_b):
    n = n_a + n_b
    delta = avg_b - avg_a
    M2 = M2_a + M2_b + delta ** 2 * n_a * n_b / n
    var_ab = M2 / (n - 1)
    return var_ab
*/
template<typename T> // 并行式Welford
inline __device__ void WelfordCombine(T b_mean, T b_m2, T b_count, T* mean, T* m2, T* count) {
  if (b_count == 0) { return; }
  T new_count = *count + b_count;
  T delta = b_mean - *mean;
  T nb_over_n = Div(b_count, new_count);
  *mean += delta * nb_over_n;
  *m2 += b_m2 + delta * delta * (*count) * nb_over_n;
  *count = new_count;
}

template<typename T> // 并行Reduce式Welford
__inline__ __device__ void WelfordWarpReduce(T thread_mean, T thread_m2, T thread_count, T* mean, T* m2, T* count) {
  *mean = thread_mean;
  *m2 = thread_m2;
  *count = thread_count;
  for (int delta = 32 / 2; delta > 0; delta /= 2) { // 一次次Reduce
    // 获取高位Thread中的mean m2 count
    // 应该还是有很多“空转”，这里明显第一次计算是前16个Thread获取后16个Thread的数据计算，而后16个Thread中的数据就没用了
    T b_mean = __shfl_down_sync(0xffffffff, *mean, delta, 32);
    T b_m2 = __shfl_down_sync(0xffffffff, *m2, delta, 32);
    T b_count = __shfl_down_sync(0xffffffff, *count, delta, 32);
    // 执行计算
    WelfordCombine(b_mean, b_m2, b_count, mean, m2, count);
  }
}

template<typename T> // 并行Reduce式Welford，算完取数据
__inline__ __device__ void WelfordWarpAllReduce(T thread_mean, T thread_m2, T thread_count, T* mean, T* m2, T* count) {
  WelfordWarpReduce<T>(thread_mean, thread_m2, thread_count, mean, m2, count);
  // 最后就是从Warp的第一个Thread里面取数据
  *mean = __shfl_sync(0xffffffff, *mean, 0, 32);
  *m2 = __shfl_sync(0xffffffff, *m2, 0, 32);
  *count = __shfl_sync(0xffffffff, *count, 0, 32);
}

template<typename T>
__global__ void layerNormKernel(T *pInput, T *pOutput, float epsilon, const int N)
{
    const int base_index = blockIdx.x * N;

    T mean = 0, m2 = 0, count = 0;
    for (int i=threadIdx.x; i < N; i += blockDim.x) {
        T v = pInput[base_index + i];
        WelfordCombine<T>(v, &mean, &m2, &count); // 先1024个Thread并行执行顺序式的Welford，算出1024个结果
    }
    __syncthreads();

    WelfordWarpAllReduce<T>(mean, m2, count, &mean, &m2, &count); // 对这1024个Thread执行Reduce式的Welford

    __shared__ T s_mean[32], s_m2[32], s_count[32]; // 1024个进程，每32为一Warp开算一轮Reduce，共产生1024/32=32个结果

    if (threadIdx.x % 32 == 0) { // 这32个结果分布在第0,32,64,...号Thread上
        uint i = threadIdx.x / 32; // 将这32个结果收集起来
        s_mean[i] = mean;
        s_m2[i] = m2;
        s_count[i] = count;
    }
    __syncthreads();

    if (threadIdx.x < 32) { // 然后再用一个Warp开算一轮Reduce
        mean = s_mean[threadIdx.x];
        m2 = s_m2[threadIdx.x];
        count = s_count[threadIdx.x];
        WelfordWarpAllReduce<T>(mean, m2, count, &mean, &m2, &count);
    }
    __syncthreads();

    __shared__ T mean_shared, var_shared;
    if (threadIdx.x == 0) { // 那么此时，0号Thread上的就是最终结果了
        mean_shared = mean;
        var_shared = Div(m2, count);
    }
    __syncthreads();
    mean = mean_shared;
    T var = var_shared;
    
    for (int i=threadIdx.x; i < N; i += blockDim.x) {
        T v = pInput[base_index + i];
        pOutput[base_index + i] = (v - mean) * (T)rsqrtf(var + (T)epsilon);
    }
}

int32_t LayerNormPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I();
    int nBlock = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1], N = 1;
    for (int i = 2; i < inputDesc[0].dims.nbDims; ++i)
    {
        N *= inputDesc[0].dims.d[i];
    }
    if (inputDesc[0].type == DataType::kFLOAT)
    {
        layerNormKernel<float><<<nBlock, 1024, 0, stream>>>((float*)inputs[0], (float*)outputs[0], epsilon_, N);
    }
    else if (inputDesc[0].type == DataType::kHALF)
    {
        layerNormKernel<__half><<<nBlock, 1024, 0, stream>>>((__half*)inputs[0], (__half*)outputs[0], epsilon_, N);
    }
    return 0;
}

REGISTER_TENSORRT_PLUGIN(LayerNormPluginCreator);