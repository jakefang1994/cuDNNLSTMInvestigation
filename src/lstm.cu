#include <stdio.h>
#include <cublas_v2.h>
#include <curand.h>

// define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
   if (stat != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
   }
}

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
   if (stat != CURAND_STATUS_SUCCESS) {
      fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
   }
}

// define device functions
__device__ float sigmoid(float in) {
    return 1.f / (1.f + expf(-in));
}

// define pointwise functions for unfused elementwise LSTM unit
__global__ void pw_biasAdd(float *y, float *bias, int n, int nBias) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < n) y[i] += bias[i % nBias];
}

__global__ void pw_vecAdd(float *y, float *a,  float *b, int n) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < n) y[i] = a[i] + b[i];
}

__global__ void pw_vecMul(float *y, float *a,  float *b, int n) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < n) y[i] = a[i] * b[i];
}

__global__ void pw_tanh(float *y, float *a, int n) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < n) y[i] = tanh(a[i]);
}

__global__ void pw_sigmoid(float *y, float *a, int n) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < n) y[i] = sigmoid(a[i]);
}

int LSTM_unit_unfused(int hiddenSize,
                             int miniBatch,
                             float * __restrict__ h_in, // h(t-1) * R
                             float * __restrict__ x_in, // x(t) * W
                             float * __restrict__ bias,
                            //  float * __restrict__ linearGates,
                             float * __restrict__ h_out,// h(t)
                             float * __restrict__ c_in, // c(t-1)
                             float * __restrict__ c_out,// c(t)
                             bool training,
                             cudaStream_t stream) {
    dim3 blockDim;
    dim3 gridDim;

    int numElements = hiddenSize * miniBatch;

    blockDim.x = 128;
    gridDim.x = (numElements + blockDim.x - 1) / blockDim.x;

    // element wise calculations
    // x(t) = x(t) * W + h(t-1) * R + bias, as input to this unit

    // x(t) goes through 4 gates' activation

    // assign location to 4 gates
    float *f_t = 
    float *i_t = 
    float *g_t = 
    float *o_t = 

    // f(t) = f(t) + c(t-1)

    // i(t) = i(t) + g(t)

    // i(t) = i(t) + f(t)

    // c(t) = i(t), output cell state

    // i(t) = tanh(i(t)), i(t) === c(t) here, but we must not modify c(t)

    // h(t) = i(t) * o(t)

    return 0;
}

float LSTMTest(int hiddenSize, int miniBatch, int seqLength, int numLayers) {
    int numElements = hiddenSize * miniBatch;
    // alloc device memory
    float *d_x, *d_h, *d_c;
    cudaErrCheck(cudaMalloc((void**)&d_x, (seqLength) * (numLayers + 1) * numElements * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&d_h, (seqLength + 1) * (numLayers) * numElements * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&d_c, (seqLength + 1) * (numLayers) * numElements * sizeof(float)));

    float *d_bias;
    cudaErrCheck(cudaMalloc((void**)&d_bias, numLayers * hiddenSize * 8 * sizeof(float)));

    float *d_h_in, *d_x_in;
    cudaErrCheck(cudaMalloc((void**)&d_h_in, numLayers * numElements * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&d_x_in, seqLength * numElements * sizeof(float)));

    // do not need to alloc streams for now, use default NULL streams
    cudaStream_t *stream_x, *stream_h;
    stream_x = (cudaStream_t*)malloc(numLayers * sizeof(cudaStream_t));
    stream_h = (cudaStream_t*)malloc(numLayers * sizeof(cudaStream_t));

    for(int i = 0; i < numLayers; i++) {
        stream_x[i] = NULL;
        stream_h[i] = NULL;
    }

    // alloc events
    cudaEvent_t **events_x, **events_h;
    events_x = (cudaEvent_t**)malloc(numLayers * sizeof(cudaEvent_t*));
    events_h = (cudaEvent_t**)malloc(numLayers * sizeof(cudaEvent_t*));
    for(int i = 0; i < numLayers; i++) {
        events_x[i] = (cudaEvent_t*)malloc(seqLength * sizeof(cudaEvent_t));
        events_h[i] = (cudaEvent_t*)malloc(seqLength * sizeof(cudaEvent_t));
    }

    // initiate random inputs
    curandGenerator_t gen;
    curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1782ULL));
    curandErrCheck(curandGenerateUniform(gen, d_x, (seqLength) * (numLayers + 1) * numElements));
    curandErrCheck(curandGenerateUniform(gen, d_h, (seqLength + 1) * (numLayers) * numElements));
    curandErrCheck(curandGenerateUniform(gen, d_c, (seqLength + 1) * (numLayers) * numElements));
    curandErrCheck(curandGenerateUniform(gen, d_bias, numLayers * hiddenSize * 8));
    curandErrCheck(curandDestroyGenerator(gen));

    cudaErrCheck(cudaDeviceSynchronize());
    printf("--Done alloc mem\n");

    //start timing
    float elapsedTime;
    cudaEvent_t start, stop;
    cudaErrCheck(cudaEventCreate(&start));
    cudaErrCheck(cudaEventCreate(&stop));
    cudaErrCheck(cudaEventRecord(start));

    // LSTM

    // stop timing
    cudaErrCheck(cudaEventRecord(stop));
    cudaErrCheck(cudaEventSynchronize(stop));
    cudaErrCheck(cudaEventElapsedTime(&elapsedTime, start, stop));
    cudaErrCheck(cudaDeviceSynchronize());

    cudaErrCheck(cudaDeviceSynchronize());
    printf("--Done LSTM\n");

    // free everything
    cudaErrCheck(cudaFree(d_x));
    cudaErrCheck(cudaFree(d_h));
    cudaErrCheck(cudaFree(d_c));
    cudaErrCheck(cudaFree(d_bias));
    cudaErrCheck(cudaFree(d_x_in));
    cudaErrCheck(cudaFree(d_h_in));
    
    for(int i = 0; i < numLayers; i++) {
        if(stream_x[i] != NULL) cudaErrCheck(cudaStreamDestroy(stream_x[i]));
        if(stream_h[i] != NULL) cudaErrCheck(cudaStreamDestroy(stream_h[i]));
    }
    free(stream_x);
    free(stream_h);

    for(int i = 0; i < numLayers; i++) {
        free(events_x[i]);
        free(events_h[i]);
    }
    free(events_x);
    free(events_h);
    printf("--Done free mem\n");

    return elapsedTime;
}

int main(int argc, char* argv[]) {
    int seqLength;
    int numLayers;
    int hiddenSize;
    int miniBatch; 
    
    if (argc == 5) {
        seqLength = atoi(argv[1]);
        numLayers =  atoi(argv[2]);
        hiddenSize =  atoi(argv[3]);
        miniBatch =  atoi(argv[4]);   
    }
    else if (argc == 1) {
        printf("Running with default settings\n");
        seqLength = 100;
        numLayers = 4;
        hiddenSize = 512;
        miniBatch = 64;
    }
    else {
        printf("Usage: ./LSTM <seqLength> <numLayers> <hiddenSize> <miniBatch>\n");
        return 1;      
    }

    printf("seqLength %d, numLayers %d, hiddenSize %d, miniBatch %d\n", seqLength, numLayers, hiddenSize, miniBatch);

    int numRuns = 1;
   
    float totalTime = 0.f;
    for (int run = 0; run < numRuns; run++) {
        totalTime += LSTMTest(hiddenSize, miniBatch, seqLength, numLayers);
    }
    
    printf("Runtime %fms\n", totalTime / numRuns);
    
    return time < 0;
}