#include "LayerNormPlugin.h"

using namespace nvinfer1;

PluginFieldCollection    LayerNormPluginCreator::fc_ {};
std::vector<PluginField> LayerNormPluginCreator::attr_;

template<typename T>
__global__ void layerNormKernel(T *pInput, T *pOutput, float epsilon, const int N)
{
    const int tx = threadIdx.x;
    const int base_index = blockIdx.x * N;

    __shared__ T mean_shared, var_shared;

    // typedef cub::BlockReduce<T, 1024>            BlockReduce;
    // __shared__ typename BlockReduce::TempStorage temp;
    __shared__ T BlockReduce[1024];
    __shared__ uint BlockReduceN[1024];
    
    T _sum = 0;
    uint _ReduceN = 0;
    BlockReduceN[tx] = 0;
    for (int tid=threadIdx.x; tid < N; tid += blockDim.x) {
        T v = pInput[base_index + tid];
        _sum += v;
        _ReduceN++;
    }
    // T &ref0 = _sum;
    // T sum = BlockReduce(temp).Sum(ref0);
    BlockReduce[tx] = _sum / (T)_ReduceN;
    BlockReduceN[tx] = _ReduceN;
    __syncthreads();
    
    for (int stride = 512; stride >= 1; stride /= 2)
    {
        if (tx < stride)
        {
            uint rN = BlockReduceN[tx], lN = BlockReduceN[tx + stride];
            uint total = rN + lN;
            T rScale = rN > 0 ? (T)rN / (T)total : (T)0;
            T lScale = lN > 0 ? (T)lN / (T)total : (T)0;
            T r = rScale * BlockReduce[tx];
            T l = lScale * BlockReduce[tx + stride];
            BlockReduce[tx] = l + r;
            BlockReduceN[tx] = total;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
        mean_shared = BlockReduce[0];
    __syncthreads();

    T _var_sum = 0;
    for (int tid=threadIdx.x; tid < N; tid += blockDim.x) {
        T v = pInput[base_index + tid];
        T moment = v - mean_shared, moment2 = moment * moment;
        _var_sum += moment2;
    }
    // T &ref1 = _var_sum;
    // T  var  = BlockReduce(temp).Sum(ref1);
    BlockReduce[tx] = _var_sum;
    __syncthreads();
    
    for (int stride = 512; stride >= 1; stride /= 2)
    {
        if (tx < stride)
        {
            BlockReduce[tx] += BlockReduce[tx + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
        var_shared = BlockReduce[0] / (T)N;
    __syncthreads();

    for (int tid=threadIdx.x; tid < N; tid += blockDim.x) {
        T v = pInput[base_index + tid];
        T moment = v - mean_shared;
        pOutput[base_index + tid] = moment * (T)rsqrtf(var_shared + (T)epsilon);
        // pOutput[base_index + tid] = (T)rsqrtf((T)epsilon); // 316.2278
        // pOutput[base_index + tid] = (T)epsilon; // 0
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