#include <stdio.h>
#include <cublas_v2.h>
#include <curand.h>

#define TRAINING (false)

#define GROUP_GEMM 0
#define USE_STREAMS 0
#define FUSE_PW 0
#define PRE_TRANSPOSE 0
#define RECUR_BATCH_SIZE 1

// Define some error checking macros.
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


// Device functions
__forceinline__ __device__ float sigmoidf(float in) {
    return 1.f / (1.f + expf(-in));  
}

// Pointwise functions
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
    if (i < n) y[i] = sigmoidf(a[i]);
}

// Unfused LSTM (calling many pointwise kernels).
int LSTM_unit_unfused(int hiddenSize, 
                            int miniBatch,
                            float * __restrict__ h_in, // h(t-1) * R
                            float * __restrict__ x_in, // x(t) * W
                            float * __restrict__ bias,
                            float * __restrict__ h_out,// h(t)
                            float * __restrict__ c_in, // c(t-1)
                            float * __restrict__ c_out,// c(t)
                            cudaStream_t stream) {
    dim3 blockDim;
    dim3 gridDim;
    
    int numElements = hiddenSize * miniBatch;
    
    blockDim.x = 128;
    gridDim.x = (numElements + blockDim.x - 1) / blockDim.x;

    // element wise calculations
    // x(t) = x(t) * W + h(t-1) * R + bias, as input to this unit
    for (int i = 0; i < 4; i++) {
        pw_vecAdd <<< gridDim, blockDim, 0, stream >>> (x_in + i * numElements, 
                                                        x_in + i * numElements, 
                                                        h_in + i * numElements,
                                                        numElements);
        cudaErrCheck(cudaGetLastError());

        pw_biasAdd <<< gridDim, blockDim, 0, stream >>> (x_in + i * numElements, 
                                                         bias + i * hiddenSize, 
                                                         numElements, 
                                                         hiddenSize);
        cudaErrCheck(cudaGetLastError());
        
        pw_biasAdd <<< gridDim, blockDim, 0, stream >>> (x_in + i * numElements, 
                                                         bias + (i + 4) * hiddenSize, 
                                                         numElements, 
                                                         hiddenSize);
        cudaErrCheck(cudaGetLastError());
    }    
    
    // x(t) goes through 4 gates' activation
    pw_sigmoid <<< gridDim, blockDim, 0, stream >>> (x_in + 0 * numElements, x_in + 0 * numElements, numElements);
    cudaErrCheck(cudaGetLastError());
    
    pw_sigmoid <<< gridDim, blockDim, 0, stream >>> (x_in + 1 * numElements, x_in + 1 * numElements, numElements);
    cudaErrCheck(cudaGetLastError());
    
    pw_tanh <<< gridDim, blockDim, 0, stream >>> (x_in + 2 * numElements, x_in + 2 * numElements, numElements);
    cudaErrCheck(cudaGetLastError());
    
    pw_sigmoid <<< gridDim, blockDim, 0, stream >>> (x_in + 3 * numElements, x_in + 3 * numElements, numElements);
    cudaErrCheck(cudaGetLastError());
    
    // assign location to 4 gates
    float *in_gate      = x_in + 0 * numElements;
    float *forget_gate = x_in + 1 * numElements;
    float *in_gate2     = x_in + 2 * numElements;
    float *out_gate     = x_in + 3 * numElements;
    
    // f(t) *= c(t-1)
    pw_vecMul <<< gridDim, blockDim, 0, stream >>> (forget_gate, forget_gate, c_in, numElements);
    cudaErrCheck(cudaGetLastError());

    // i(t) *= g(t)
    pw_vecMul <<< gridDim, blockDim, 0, stream >>> (in_gate, in_gate, in_gate2, numElements);
    cudaErrCheck(cudaGetLastError());

    // i(t) += f(t)  
    pw_vecAdd <<< gridDim, blockDim, 0, stream >>> (in_gate, in_gate, forget_gate, numElements);
    cudaErrCheck(cudaGetLastError());

    // c(t) = i(t), output cell state
    cudaErrCheck(cudaMemcpyAsync(c_out, in_gate, numElements * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    
    // i(t) = tanh(i(t)), i(t) === c(t) here, but we must not modify c(t)
    pw_tanh <<< gridDim, blockDim, 0, stream >>> (in_gate, in_gate, numElements);
    cudaErrCheck(cudaGetLastError());
    
     // h(t) = i(t) * o(t)
    pw_vecMul <<< gridDim, blockDim, 0, stream >>> (h_out, out_gate, in_gate, numElements);
    cudaErrCheck(cudaGetLastError());
    
    return 0;
}


float LSTMTest(int hiddenSize, int miniBatch, int seqLength, int numLayers, bool checkF) {
    int numElements = hiddenSize * miniBatch;

    // alloc device memory
    float *h_data, *i_data, *c_data;
    cudaErrCheck(cudaMalloc((void**)&h_data, (seqLength + 1) * (numLayers) * numElements * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&i_data, (seqLength) * (numLayers + 1) * numElements * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&c_data, (seqLength + 1) * (numLayers) * numElements * sizeof(float)));
    
    float *weight, *weight_T;
    cudaErrCheck(cudaMalloc((void**)&weight, numLayers * hiddenSize * hiddenSize * 8 * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&weight_T, numLayers * hiddenSize * hiddenSize * 8 * sizeof(float)));
    
    float *bias;
    cudaErrCheck(cudaMalloc((void**)&bias, numLayers * hiddenSize * 8 * sizeof(float)));
    
    float *h_in, *x_in;
    cudaErrCheck(cudaMalloc((void**)&h_in, 4 * numLayers * numElements * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&x_in, 4 * seqLength * numElements * sizeof(float)));

    float *linearGates;
    // Activations
    if (TRAINING) {
        cudaErrCheck(cudaMalloc((void**)&linearGates, 4 * seqLength * numLayers * numElements * sizeof(float)));
    }
    
    // all use default NULL stream for now
    cudaStream_t *stream_x, *stream_h;
    stream_x = (cudaStream_t*)malloc(numLayers * sizeof(cudaStream_t));
    stream_h = (cudaStream_t*)malloc(numLayers * sizeof(cudaStream_t));

    for (int i = 0; i < numLayers; i++) {
        if (USE_STREAMS) {
          // optimization 2 uses different streams for x and h
          // optimization 6 uses different streams for various layers
        }
        else {
            stream_x[i] = NULL;  
            stream_h[i] = NULL;  
        }
    }
    
    // alloc events
    cudaEvent_t **events_x, **events_h;
    events_x = (cudaEvent_t**)malloc(numLayers * sizeof(cudaEvent_t*));
    events_h = (cudaEvent_t**)malloc(numLayers * sizeof(cudaEvent_t*));
    for (int i = 0; i < numLayers; i++) {
        events_x[i] = (cudaEvent_t*)malloc(seqLength * sizeof(cudaEvent_t));
        events_h[i] = (cudaEvent_t*)malloc(seqLength * sizeof(cudaEvent_t));
    }

    // initiate random inputs
    curandGenerator_t gen;
    curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1782ULL));
    curandErrCheck(curandGenerateUniform(gen, h_data, (seqLength + 1) * (numLayers) * numElements));
    curandErrCheck(curandGenerateUniform(gen, c_data, (seqLength + 1) * (numLayers) * numElements));
    curandErrCheck(curandGenerateUniform(gen, i_data, (seqLength) * (numLayers + 1) * numElements));
    curandErrCheck(curandGenerateUniform(gen, weight, numLayers * hiddenSize * hiddenSize * 8));
    curandErrCheck(curandGenerateUniform(gen, bias, numLayers * hiddenSize * 8));
    curandErrCheck(curandDestroyGenerator(gen));

    // create cuBLAS handle.
    cublasHandle_t handle;
    cublasErrCheck(cublasCreate(&handle));
    
    cudaErrCheck(cudaDeviceSynchronize());
  
    // start timing
    float elapsedTime;
    cudaEvent_t start, stop;
    cudaErrCheck(cudaEventCreate(&start));
    cudaErrCheck(cudaEventCreate(&stop));
    cudaErrCheck(cudaEventRecord(start));
  
    // LSTM

    const cublasOperation_t a_trans = CUBLAS_OP_T; // no pre-transpose for now, do optimization 4 here later
    const cublasOperation_t b_trans = CUBLAS_OP_N; // always N
    
    if (a_trans == CUBLAS_OP_N) {        
        // do optimization 4 here, transpose A     
    }
    else {
        weight_T = weight;
    }

    // cublasSgemm(): C = alpha * (A + B) + beta * C 
    float alpha = 1.f;
    float beta  = 0.f;        
    
    int lStart = 0; // layer starts from
    int lEnd = 0;   // layer ends at
    int tStart = 0; // timestep starts from 
    int tEnd = 0;   // timestep ends at
    int recurBatchSize = RECUR_BATCH_SIZE; // optimization 5 will make it 2
    
    while (true) {
        // Many layer "scheduling".
        if (lEnd == 0) {
            lStart = 0;
            lEnd = 1;
            tStart = 0;
        }
        else {
            // Move "up" and "left"
            lStart++;
            lEnd++;
            
            tStart -= recurBatchSize;
            
            // Over the top or off the left, reset to layer 0
            if (lEnd > numLayers || tStart < 0) {
                tStart += (lStart + 1) * recurBatchSize;

                lStart = 0;
                lEnd = 1;
            }
            
            // Off the right, step up
            while (tStart >= seqLength && lEnd <= numLayers) {
                lStart++;
                lEnd++;
                
                tStart -= recurBatchSize;
            }
            
            
            // Over the top or off the left, done!
            if (lEnd > numLayers || tStart < 0) {
                break;
            }
        }
        
        tEnd = tStart + recurBatchSize;
        if (tEnd > seqLength) tEnd = seqLength;
        
        // lStart, lEnd always differ 1
        for (int layer = lStart; layer < lEnd; layer++) {            
            cublasErrCheck(cublasSetStream(handle, stream_x[layer]));
            
            // tStart, tEnd differ recurBatchSize
            for (int i = tStart; i < tEnd; i++) {
                if (layer > 0) {
                    cudaErrCheck(cudaStreamWaitEvent(stream_x[layer], events_h[layer - 1][i], 0));
                    cudaErrCheck(cudaEventDestroy(events_h[layer - 1][i]));
                }
            }

            // x(t) *= [W_weight]
            if (GROUP_GEMM) {
                // do optimization 1 here
            }
            else {
                for (int igemm =0; igemm < 4; igemm++) {
                    cublasErrCheck(cublasSgemm(handle,
                                    a_trans, b_trans,
                                    hiddenSize, // #rows of A and C
                                    miniBatch * (tEnd - tStart), // #cols of B and C
                                    hiddenSize, // #cols of A and B
                                    &alpha,
                                    &weight_T[layer * 8 * hiddenSize * hiddenSize + igemm * hiddenSize], // A
                                    a_trans == CUBLAS_OP_N ? 4 * hiddenSize : hiddenSize, // leading dimension of A, where we can try different data layout
                                    i_data + tStart * numElements + layer * seqLength * numElements, // B
                                    hiddenSize, // leading dimension of B, where we can try different data layout
                                    &beta,
                                    x_in + 4 * tStart * numElements + igemm * hiddenSize, // C
                                    4 * hiddenSize // leading dimension of C
                                    )); 
                }
            }
            
            for (int i = tStart; i < tEnd; i++) {
                cudaErrCheck(cudaEventCreate(&events_x[layer][i], cudaEventDisableTiming));
                cudaErrCheck(cudaEventRecord(events_x[layer][i], stream_x[layer]));  
            }                
            
            for (int i = tStart; i < tEnd; i++) {
                cublasErrCheck(cublasSetStream(handle, stream_h[layer]));

                // h(t-1) *= [R_weight]
                if (GROUP_GEMM) {
                     // do optimization 1 here
                }
                else {
                    for (int igemm =0; igemm < 4; igemm++) {
                        cublasErrCheck(cublasSgemm(handle,
                                        a_trans, b_trans,
                                        hiddenSize, miniBatch, hiddenSize,
                                        &alpha,
                                        &weight_T[4 * hiddenSize * hiddenSize + layer * 8 * hiddenSize * hiddenSize + igemm * hiddenSize], 
                                        a_trans == CUBLAS_OP_N ? 4 * hiddenSize : hiddenSize,
                                        h_data + i * numElements + layer * (seqLength + 1) * numElements,
                                        hiddenSize,
                                        &beta,
                                        h_in + 4 * layer * numElements + igemm * hiddenSize, 
                                        4 * hiddenSize));
                    }
                }

                cudaErrCheck(cudaStreamWaitEvent(stream_h[layer], events_x[layer][i], 0));
                cudaErrCheck(cudaEventDestroy(events_x[layer][i]));

                if (FUSE_PW) {
                    // optimization 3 here
                }
                else {
                    LSTM_unit_unfused(hiddenSize, miniBatch,
                              h_in + 4 * layer * numElements, 
                              x_in + 4 * i * numElements, 
                              bias + 8 * layer * hiddenSize,
                              h_data + (i + 1) * numElements + layer * (seqLength + 1) * numElements,
                              c_data + i * numElements + layer * (seqLength + 1) * numElements,
                              c_data + (i + 1) * numElements + layer * (seqLength + 1) * numElements,
                              stream_h[layer]);
                }
                if (layer != numLayers - 1) {
                    cudaErrCheck(cudaEventCreate(&events_h[layer][i], cudaEventDisableTiming));
                    cudaErrCheck(cudaEventRecord(events_h[layer][i], stream_h[layer]));  
                }
            }
        }
    } 

    // stop timing
    cudaErrCheck(cudaEventRecord(stop));
    cudaErrCheck(cudaEventSynchronize(stop));
    cudaErrCheck(cudaEventElapsedTime(&elapsedTime, start, stop));
    cudaErrCheck(cudaDeviceSynchronize());

    // free everything
    cudaErrCheck(cudaFree(h_data));
    cudaErrCheck(cudaFree(i_data));  
    cudaErrCheck(cudaFree(c_data));  

    if (weight != weight_T) cudaErrCheck(cudaFree(weight)); 
    cudaErrCheck(cudaFree(weight_T));
    
    cudaErrCheck(cudaFree(bias));
    
    cudaErrCheck(cudaFree(h_in));
    cudaErrCheck(cudaFree(x_in));
    if (TRAINING) cudaErrCheck(cudaFree(linearGates));

    
    for (int i = 0; i < numLayers; i++) {
        if (stream_x[i] != NULL) cudaErrCheck(cudaStreamDestroy(stream_x[i]));
        if (stream_h[i] != NULL) cudaErrCheck(cudaStreamDestroy(stream_h[i]));
    }

    free(stream_x);
    free(stream_h);
    
    for (int i = 0; i < numLayers; i++) {
        free(events_x[i]);
        free(events_h[i]);
    }
    free(events_x);
    free(events_h);
    
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
        totalTime += LSTMTest(hiddenSize, miniBatch, seqLength, numLayers, true);
    }
    
    printf("Runtime %fms\n", totalTime / numRuns);
    
    return time < 0;
}