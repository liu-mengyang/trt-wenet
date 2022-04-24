#include "LayerNormPlugin.h"

using namespace nvinfer1;

PluginFieldCollection    LayerNormPluginCreator::fc_ {};
std::vector<PluginField> LayerNormPluginCreator::attr_;

template<typename T>
__global__ void layerNormKernel(T *pInput, T *pOutput, float epsilon, const int N)
{
    const int base_index = blockIdx.x * N;

    __shared__ T mean_shared, var_shared;

    typedef cub::BlockReduce<T, 1024>            BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    T _sum = 0;
    for (int tid=threadIdx.x; tid < N; tid += blockDim.x) {
        T v = pInput[base_index + tid];
        _sum += v;
    }
    T &ref0 = _sum;
    T sum = BlockReduce(temp).Sum(ref0);

    if (threadIdx.x == 0)
        mean_shared = sum / (T)N;
    __syncthreads();

    T _var_sum = 0;
    for (int tid=threadIdx.x; tid < N; tid += blockDim.x) {
        T v = pInput[base_index + tid];
        T moment = v - mean_shared, moment2 = moment * moment;
        _var_sum += moment2;
    }
    T &ref1 = _var_sum;
    T  var  = BlockReduce(temp).Sum(ref1);

    if (threadIdx.x == 0)
        var_shared = var / (T)N;
    __syncthreads();

    for (int tid=threadIdx.x; tid < N; tid += blockDim.x) {
        T v = pInput[base_index + tid];
        T moment = v - mean_shared;
        pOutput[base_index + tid] = moment * (T)rsqrtf(var_shared + (T)epsilon);
    }
}

int32_t LayerNormPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I();
    int nBlock = inputDesc[0].dims.d[0], N = 1;
    for (int i = 1; i < inputDesc[0].dims.nbDims; ++i)
    {
        N *= inputDesc[0].dims.d[i];
    }
    int threadsPerBlock = 1024;
    if (inputDesc[0].type == DataType::kFLOAT)
    {
        layerNormKernel<float><<<nBlock, threadsPerBlock, 0, stream>>>((float*)inputs[0], (float*)outputs[0], epsilon_, N);
    }
    else
    {
        layerNormKernel<float><<<nBlock, threadsPerBlock, 0, stream>>>((float*)inputs[0], (float*)outputs[0], epsilon_, N);
    }
    return 0;
}

REGISTER_TENSORRT_PLUGIN(LayerNormPluginCreator);