#include <stdio.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cudnn.h>
#include <curand_kernel.h>

#define TRAINING (false)

#define GROUP_GEMM 0
#define USE_STREAMS 0
#define FUSE_PW 0
#define PRE_TRANSPOSE 0
#define RECUR_BATCH_SIZE 1

#define BW 128

// Define some error checking macros.
void cudaErrCheck(cudaError_t stat, const char *file, int line) {
    if (stat != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
    }
}

void cublasErrCheck(cublasStatus_t stat, const char *file, int line) {
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
    }
}

void curandErrCheck(curandStatus_t stat, const char *file, int line) {
    if (stat != CURAND_STATUS_SUCCESS) {
        fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
    }
}

void cudnnErrCheck(cudnnStatus_t stat, const char *file, int line) {
    if (stat != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "cuDNN Error: %d %s %d\n", stat, file, line);
        cudaDeviceReset();
		exit(1);
    }
}

int RoundUp(int nominator, int denominator)
{
	return (nominator + denominator - 1) / denominator;
}

__global__ void FillVecKernel(float* d_vec, float value, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx <= size) return;
	d_vec[idx] = value;
}

void FillVec(int size, float* vec_d, float value)
{
	FillVecKernel <<<RoundUp(size, BW), BW >>> (vec_d, value, size);
	/*float* data = new float[size];
	for (int i = 0; i < size; i++) {
	data[i] = value;
	}
	CheckError(cudaMemcpy(vec_d, data, size*sizeof(float), cudaMemcpyHostToDevice), __FILE__, __LINE__);*/
}

__global__ void InitDataDistributedKernel(float min, float max, float* d_vec, int size) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= size) return;

	curandState state;
	curand_init((unsigned long long)clock() + idx, 0, 0, &state);

	float value = curand_uniform(&state);
	float oldRange = (1.f - 0.f);
	float newRange = (max - (min));
	float newValue = (((value - 0.f) * newRange) / oldRange) + (min);

	d_vec[idx] = newValue;
}

void initDataDistributed(float min, float max, float* d_vec, int size) {
	InitDataDistributedKernel <<< RoundUp(size, BW), BW >>> (min, max, d_vec, size);
}


float LSTMTest(int hiddenSize, int miniBatch, int seqLength, int numLayers, bool checkF) {
    // Initialization
	int m_gpuid = 0;
    cudaErrCheck(cudaSetDevice(m_gpuid), __FILE__, __LINE__);
    size_t version = cudnnGetVersion();
    printf("cuDNN version: %zd\n", version);
    cudnnHandle_t m_handle;
    cudnnErrCheck(cudnnCreate(&m_handle), __FILE__, __LINE__);

    // Declare
    int m_batchSize = miniBatch;
	int m_layerNumber = numLayers;
	int m_hiddenSize = hiddenSize;
	float m_dropout = 0;
	int m_seqLength = seqLength;
	int m_inputSize = hiddenSize;

    cudnnTensorFormat_t m_tensorFormat = CUDNN_TENSOR_NCHW;
	size_t m_workspaceSize;
	size_t m_trainingSize;
	size_t m_weightsSize;
	size_t m_workSize;
	size_t m_reserveSize;
	size_t m_dataTypeSize = sizeof(float);

    cudnnRNNDescriptor_t m_rnnDesc;
    cudnnFilterDescriptor_t m_weightsDesc;
    // cudnnFilterDescriptor_t m_weightsGradientDesc;
	cudnnDropoutDescriptor_t m_dropoutDesc;
    cudnnTensorDescriptor_t *m_srcDataDesc;
    cudnnTensorDescriptor_t *m_dstDataDesc; 
    // cudnnTensorDescriptor_t *m_gradientInputDesc;
    // cudnnTensorDescriptor_t *m_gradientOutputDesc;
    cudnnTensorDescriptor_t m_hiddenInputDesc;
    cudnnTensorDescriptor_t m_cellInputDesc;
    cudnnTensorDescriptor_t m_hiddenOutputDesc;
    cudnnTensorDescriptor_t m_cellOutputDesc;
    // cudnnTensorDescriptor_t m_gradientHiddenInputDesc;
    // cudnnTensorDescriptor_t m_gradientCellInputDesc;
    // cudnnTensorDescriptor_t m_gradHiddenOutputDesc; 
    // cudnnTensorDescriptor_t m_gradCellOutputDesc;

    float *m_d_hiddenInput = NULL;
    float *m_d_cellInput = NULL;
    // Add input
    float *m_d_input = NULL;

	// float *m_d_gradientInput;
	// float *m_d_gradientHiddenInput = NULL;
	// float *m_d_gradientCellInput = NULL;

	float *m_d_dstData;
	float *m_d_hiddenOutput = NULL;
	float *m_d_cellOutput = NULL;

	// float *m_d_gradientOutput;
	// float *m_d_gradHiddenOutput = NULL;
	// float *m_d_gradCellOutput = NULL;

	float *m_d_weights;
	float *m_d_weightsGradient;

	float *m_d_workspace;
	float *m_d_reserveSpace;

    // Allocating
    cudaErrCheck(cudaMalloc((void**)&m_d_hiddenInput, m_layerNumber * m_hiddenSize * m_batchSize * m_dataTypeSize), __FILE__, __LINE__);
    cudaErrCheck(cudaMalloc((void**)&m_d_cellInput, m_layerNumber * m_hiddenSize * m_batchSize * m_dataTypeSize), __FILE__, __LINE__);
    // Add input
    cudaErrCheck(cudaMalloc((void**)&m_d_input, m_layerNumber * m_hiddenSize * m_batchSize * m_dataTypeSize), __FILE__, __LINE__);

	// cudaErrCheck(cudaMalloc((void**)&m_d_gradientInput, m_seqLength * m_inputSize * m_batchSize * m_dataTypeSize), __FILE__, __LINE__);
	// cudaErrCheck(cudaMalloc((void**)&m_d_gradientHiddenInput, m_layerNumber * m_hiddenSize * m_batchSize  * m_dataTypeSize), __FILE__, __LINE__);
	// cudaErrCheck(cudaMalloc((void**)&m_d_gradientCellInput, m_layerNumber * m_hiddenSize * m_batchSize  * m_dataTypeSize), __FILE__, __LINE__);

	cudaErrCheck(cudaMalloc((void**)&m_d_dstData, m_seqLength * m_hiddenSize * m_batchSize  * m_dataTypeSize), __FILE__, __LINE__);
	cudaErrCheck(cudaMalloc((void**)&m_d_hiddenOutput, m_layerNumber * m_hiddenSize * m_batchSize  * m_dataTypeSize), __FILE__, __LINE__);
	cudaErrCheck(cudaMalloc((void**)&m_d_cellOutput, m_layerNumber * m_hiddenSize * m_batchSize  * m_dataTypeSize), __FILE__, __LINE__);

	// cudaErrCheck(cudaMalloc((void**)&m_d_gradientOutput, m_seqLength * m_hiddenSize * m_batchSize  * m_dataTypeSize), __FILE__, __LINE__);
	// cudaErrCheck(cudaMalloc((void**)&m_d_gradHiddenOutput, m_layerNumber * m_hiddenSize * m_batchSize  * m_dataTypeSize), __FILE__, __LINE__);
	// cudaErrCheck(cudaMalloc((void**)&m_d_gradCellOutput, m_layerNumber * m_hiddenSize * m_batchSize  * m_dataTypeSize), __FILE__, __LINE__);

    m_srcDataDesc = (cudnnTensorDescriptor_t*)malloc(m_seqLength * sizeof(cudnnTensorDescriptor_t));
	m_dstDataDesc = (cudnnTensorDescriptor_t*)malloc(m_seqLength * sizeof(cudnnTensorDescriptor_t));
	// m_gradientInputDesc = (cudnnTensorDescriptor_t*)malloc(m_seqLength * sizeof(cudnnTensorDescriptor_t));
	// m_gradientOutputDesc = (cudnnTensorDescriptor_t*)malloc(m_seqLength * sizeof(cudnnTensorDescriptor_t));

    // Create Descriptor
    for (int i = 0; i < m_seqLength; i++) {
		cudnnErrCheck(cudnnCreateTensorDescriptor(&m_srcDataDesc[i]), __FILE__, __LINE__);
		cudnnErrCheck(cudnnCreateTensorDescriptor(&m_dstDataDesc[i]), __FILE__, __LINE__);
		// cudnnErrCheck(cudnnCreateTensorDescriptor(&m_gradientInputDesc[i]), __FILE__, __LINE__);
		// cudnnErrCheck(cudnnCreateTensorDescriptor(&m_gradientOutputDesc[i]), __FILE__, __LINE__);
	}

	cudnnErrCheck(cudnnCreateTensorDescriptor(&m_hiddenInputDesc), __FILE__, __LINE__);
	cudnnErrCheck(cudnnCreateTensorDescriptor(&m_cellInputDesc), __FILE__, __LINE__);
	cudnnErrCheck(cudnnCreateTensorDescriptor(&m_hiddenOutputDesc), __FILE__, __LINE__);
	cudnnErrCheck(cudnnCreateTensorDescriptor(&m_cellOutputDesc), __FILE__, __LINE__);
	// cudnnErrCheck(cudnnCreateTensorDescriptor(&m_gradientHiddenInputDesc), __FILE__, __LINE__);
	// cudnnErrCheck(cudnnCreateTensorDescriptor(&m_gradientCellInputDesc), __FILE__, __LINE__);
	// cudnnErrCheck(cudnnCreateTensorDescriptor(&m_gradHiddenOutputDesc), __FILE__, __LINE__);
	// cudnnErrCheck(cudnnCreateTensorDescriptor(&m_gradCellOutputDesc), __FILE__, __LINE__);

	cudnnErrCheck(cudnnCreateDropoutDescriptor(&m_dropoutDesc), __FILE__, __LINE__);

	cudnnErrCheck(cudnnCreateRNNDescriptor(&m_rnnDesc), __FILE__, __LINE__);

    // Setting up TensorDescriptor
    int dimA[3];
	int strideA[3];

    for (int i = 0; i < m_seqLength; i++) {
		dimA[0] = m_batchSize;
		dimA[1] = m_inputSize;
		dimA[2] = 1;

		strideA[0] = dimA[2] * dimA[1];
		strideA[1] = dimA[2];
		strideA[2] = 1;

		cudnnErrCheck(cudnnSetTensorNdDescriptor(m_srcDataDesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA), __FILE__, __LINE__);
		// CheckError(cudnnSetTensorNdDescriptor(m_gradientInputDesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA), __FILE__, __LINE__);

		dimA[0] = m_batchSize;
		dimA[1] = m_hiddenSize;
		dimA[2] = 1;

		strideA[0] = dimA[2] * dimA[1];
		strideA[1] = dimA[2];
		strideA[2] = 1;

		cudnnErrCheck(cudnnSetTensorNdDescriptor(m_dstDataDesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA), __FILE__, __LINE__);
		// CheckError(cudnnSetTensorNdDescriptor(m_gradientOutputDesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA), __FILE__, __LINE__);
	}

    dimA[0] = m_layerNumber;
	dimA[1] = m_batchSize;
	dimA[2] = m_hiddenSize;

	strideA[0] = dimA[2] * dimA[1];
	strideA[1] = dimA[2];
	strideA[2] = 1;

	cudnnErrCheck(cudnnSetTensorNdDescriptor(m_hiddenInputDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA), __FILE__, __LINE__);
	cudnnErrCheck(cudnnSetTensorNdDescriptor(m_cellInputDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA), __FILE__, __LINE__);
	cudnnErrCheck(cudnnSetTensorNdDescriptor(m_hiddenOutputDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA), __FILE__, __LINE__);
	cudnnErrCheck(cudnnSetTensorNdDescriptor(m_cellOutputDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA), __FILE__, __LINE__);
	// cudnnErrCheck(cudnnSetTensorNdDescriptor(m_gradientHiddenInputDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA), __FILE__, __LINE__);
	// cudnnErrCheck(cudnnSetTensorNdDescriptor(m_gradientCellInputDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA), __FILE__, __LINE__);
	// cudnnErrCheck(cudnnSetTensorNdDescriptor(m_gradHiddenOutputDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA), __FILE__, __LINE__);
	// cudnnErrCheck(cudnnSetTensorNdDescriptor(m_gradCellOutputDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA), __FILE__, __LINE__);

    size_t stateSize;
	void *states;
	unsigned long long seed = 1782ull; // Pick a seed.

	cudnnErrCheck(cudnnDropoutGetStatesSize(m_handle, &stateSize), __FILE__, __LINE__);

	cudaErrCheck(cudaMalloc(&states, stateSize), __FILE__, __LINE__);

	cudnnErrCheck(cudnnSetDropoutDescriptor(m_dropoutDesc,
		m_handle,
		m_dropout,
		states,
		stateSize,
		seed), __FILE__, __LINE__);

    cudnnErrCheck(cudnnSetRNNDescriptor(m_handle,
        m_rnnDesc,
		m_hiddenSize,
		m_layerNumber,
		m_dropoutDesc,
		CUDNN_LINEAR_INPUT, // We can also skip the input matrix transformation
		CUDNN_UNIDIRECTIONAL,
        CUDNN_LSTM,
        CUDNN_RNN_ALGO_STANDARD,
		CUDNN_DATA_FLOAT), __FILE__, __LINE__);

    // Set up parameters

    cudnnErrCheck(cudnnCreateFilterDescriptor(&m_weightsDesc), __FILE__, __LINE__);
	// cudnnErrCheck(cudnnCreateFilterDescriptor(&m_weightsGradientDesc), __FILE__, __LINE__);

	cudnnErrCheck(cudnnGetRNNParamsSize(m_handle, m_rnnDesc, m_srcDataDesc[0], &m_weightsSize, CUDNN_DATA_FLOAT), __FILE__, __LINE__);

    printf("Number of params: %f", (m_weightsSize / m_dataTypeSize));

	int dimW[3];
	dimW[0] = m_weightsSize / m_dataTypeSize;
	dimW[1] = 1;
	dimW[2] = 1;

	cudnnErrCheck(cudnnSetFilterNdDescriptor(m_weightsDesc, CUDNN_DATA_FLOAT, m_tensorFormat, 3, dimW), __FILE__, __LINE__);
	// cudnnErrCheck(cudnnSetFilterNdDescriptor(m_weightsGradientDesc, CUDNN_DATA_FLOAT, m_tensorFormat, 3, dimW), __FILE__, __LINE__);

	cudaErrCheck(cudaMalloc((void**)&m_d_weights, m_weightsSize), __FILE__, __LINE__);
	// cudnnErrCheck(cudaMalloc((void**)&m_d_weightsGradient, m_weightsSize), __FILE__, __LINE__);

	// Set up work space and reserved memory  
	// Need for every pass
	cudnnErrCheck(cudnnGetRNNWorkspaceSize(m_handle, m_rnnDesc, m_seqLength, m_srcDataDesc, &m_workSize), __FILE__, __LINE__);
	// Only needed in training, shouldn't be touched between passes.
	cudnnErrCheck(cudnnGetRNNTrainingReserveSize(m_handle, m_rnnDesc, m_seqLength, m_srcDataDesc, &m_reserveSize), __FILE__, __LINE__);

	cudaErrCheck(cudaMalloc((void**)&m_d_workspace, m_workSize), __FILE__, __LINE__);
	cudaErrCheck(cudaMalloc((void**)&m_d_reserveSpace, m_reserveSize), __FILE__, __LINE__);

	cudaErrCheck(cudaDeviceSynchronize(), __FILE__, __LINE__);

    // Initialize weights
    int numLinearLayers = 8; // 2 for RELU/TANH, 8 for LSTM and 6 for GRU
	int totalNbParams = 0;

	for (int layer = 0; layer < m_layerNumber; layer++) {
		int nbParams = 0;
		for (int linLayerID = 0; linLayerID < numLinearLayers; linLayerID++) {
			cudnnFilterDescriptor_t linLayerMatDesc;
			cudnnErrCheck(cudnnCreateFilterDescriptor(&linLayerMatDesc), __FILE__, __LINE__);
			float *d_linLayerMat;

			cudnnErrCheck(cudnnGetRNNLinLayerMatrixParams(m_handle,
				m_rnnDesc,
				layer,
				m_srcDataDesc[0],
				m_weightsDesc,
				m_d_weights,
				linLayerID,
				linLayerMatDesc,
				(void**)&d_linLayerMat), __FILE__, __LINE__);

			cudnnDataType_t dataType;
			cudnnTensorFormat_t format;
			int nbDims;
			int filterDimA[3];
			cudnnErrCheck(cudnnGetFilterNdDescriptor(linLayerMatDesc,
				3,
				&dataType,
				&format,
				&nbDims,
				filterDimA), __FILE__, __LINE__);

			initDataDistributed(-0.08f, 0.08f, d_linLayerMat, filterDimA[0] * filterDimA[1] * filterDimA[2]);

			nbParams += filterDimA[0] * filterDimA[1] * filterDimA[2];

			cudnnErrCheck(cudnnDestroyFilterDescriptor(linLayerMatDesc), __FILE__, __LINE__);

			cudnnFilterDescriptor_t linLayerBiasDesc;
			cudnnErrCheck(cudnnCreateFilterDescriptor(&linLayerBiasDesc), __FILE__, __LINE__);
			float *d_linLayerBias;

			cudnnErrCheck(cudnnGetRNNLinLayerBiasParams(m_handle,
				m_rnnDesc,
				layer,
				m_srcDataDesc[0],
				m_weightsDesc,
				m_d_weights,
				linLayerID,
				linLayerBiasDesc,
				(void**)&d_linLayerBias), __FILE__, __LINE__);

            cudnnErrCheck(cudnnGetFilterNdDescriptor(linLayerBiasDesc,
				3,
				&dataType,
				&format,
				&nbDims,
				filterDimA), __FILE__, __LINE__);

			FillVec(filterDimA[0] * filterDimA[1] * filterDimA[2], d_linLayerBias, 1.f);

			nbParams += filterDimA[0] * filterDimA[1] * filterDimA[2];
			cudnnErrCheck(cudnnDestroyFilterDescriptor(linLayerBiasDesc), __FILE__, __LINE__);
		}
		totalNbParams += nbParams;
    }

    // initiate random inputs
    curandGenerator_t gen;
    curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT), __FILE__, __LINE__);
    curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1782ULL), __FILE__, __LINE__);
    curandErrCheck(curandGenerateUniform(gen, m_d_hiddenInput, m_layerNumber * m_hiddenSize * m_batchSize * m_dataTypeSize), __FILE__, __LINE__);
    curandErrCheck(curandGenerateUniform(gen, m_d_cellInput, m_layerNumber * m_hiddenSize * m_batchSize * m_dataTypeSize), __FILE__, __LINE__);
    curandErrCheck(curandGenerateUniform(gen, m_d_input, m_layerNumber * m_hiddenSize * m_batchSize * m_dataTypeSize), __FILE__, __LINE__);
    // curandErrCheck(curandGenerateUniform(gen, weight, numLayers * hiddenSize * hiddenSize * 8), __FILE__, __LINE__);
    // curandErrCheck(curandGenerateUniform(gen, bias, numLayers * hiddenSize * 8), __FILE__, __LINE__);
    curandErrCheck(curandDestroyGenerator(gen), __FILE__, __LINE__);

    // // create cuBLAS handle.
    // cublasHandle_t handle;
    // cublasErrCheck(cublasCreate(&handle), __FILE__, __LINE__);
    
    // cudaErrCheck(cudaDeviceSynchronize(), __FILE__, __LINE__);
    
    // start timing
    float elapsedTime;
    cudaEvent_t start, stop;
    cudaErrCheck(cudaEventCreate(&start), __FILE__, __LINE__);
    cudaErrCheck(cudaEventCreate(&stop), __FILE__, __LINE__);
    cudaErrCheck(cudaEventRecord(start), __FILE__, __LINE__);

    // Inference
    cudnnErrCheck(cudnnRNNForwardInference(m_handle,
        m_rnnDesc,
        m_seqLength,
        m_srcDataDesc,
        m_d_input,
        m_hiddenInputDesc,
        m_d_hiddenInput,
        m_cellInputDesc,
        m_d_cellInput,
        m_weightsDesc,
        m_d_weights,
        m_dstDataDesc,
        m_d_dstData,
        m_hiddenOutputDesc,
        m_d_hiddenOutput,
        m_cellOutputDesc,
        m_d_cellOutput,
        m_d_workspace,
        m_workSize), __FILE__, __LINE__);
    
    // stop timing
    cudaErrCheck(cudaEventRecord(stop), __FILE__, __LINE__);
    cudaErrCheck(cudaEventSynchronize(stop), __FILE__, __LINE__);
    cudaErrCheck(cudaEventElapsedTime(&elapsedTime, start, stop), __FILE__, __LINE__);
    cudaErrCheck(cudaDeviceSynchronize(), __FILE__, __LINE__);

    // free everything
    cudaErrCheck(cudaFree(m_d_hiddenInput), __FILE__, __LINE__);
	cudaErrCheck(cudaFree(m_d_cellInput), __FILE__, __LINE__);
	cudaErrCheck(cudaFree(m_d_dstData), __FILE__, __LINE__);
	cudaErrCheck(cudaFree(m_d_hiddenOutput), __FILE__, __LINE__);
	cudaErrCheck(cudaFree(m_d_cellOutput), __FILE__, __LINE__);
	// cudaErrCheck(cudaFree(m_d_gradientInput), __FILE__, __LINE__);
	// cudaErrCheck(cudaFree(m_d_gradientHiddenInput), __FILE__, __LINE__);
	// cudaErrCheck(cudaFree(m_d_gradientCellInput), __FILE__, __LINE__);
	// cudaErrCheck(cudaFree(m_d_gradientOutput), __FILE__, __LINE__);
	// cudaErrCheck(cudaFree(m_d_gradHiddenOutput), __FILE__, __LINE__);
	// cudaErrCheck(cudaFree(m_d_gradCellOutput), __FILE__, __LINE__);
	cudaErrCheck(cudaFree(m_d_workspace), __FILE__, __LINE__);
	cudaErrCheck(cudaFree(m_d_reserveSpace), __FILE__, __LINE__);
	cudaErrCheck(cudaFree(m_d_weights), __FILE__, __LINE__);
	// cudaErrCheck(cudaFree(m_d_weightsGradient), __FILE__, __LINE__);

	cudnnErrCheck(cudnnDestroyRNNDescriptor(m_rnnDesc), __FILE__, __LINE__);
	cudnnErrCheck(cudnnDestroyFilterDescriptor(m_weightsDesc), __FILE__, __LINE__);
	// cudnnErrCheck(cudnnDestroyFilterDescriptor(m_weightsGradientDesc), __FILE__, __LINE__);
	cudnnErrCheck(cudnnDestroyDropoutDescriptor(m_dropoutDesc), __FILE__, __LINE__);
	cudnnErrCheck(cudnnDestroyTensorDescriptor(m_hiddenInputDesc), __FILE__, __LINE__);
	cudnnErrCheck(cudnnDestroyTensorDescriptor(m_cellInputDesc), __FILE__, __LINE__);
	cudnnErrCheck(cudnnDestroyTensorDescriptor(m_hiddenOutputDesc), __FILE__, __LINE__);
	cudnnErrCheck(cudnnDestroyTensorDescriptor(m_cellOutputDesc), __FILE__, __LINE__);
	// cudnnErrCheck(cudnnDestroyTensorDescriptor(m_gradientHiddenInputDesc), __FILE__, __LINE__);
	// cudnnErrCheck(cudnnDestroyTensorDescriptor(m_gradientCellInputDesc), __FILE__, __LINE__);
	// cudnnErrCheck(cudnnDestroyTensorDescriptor(m_gradHiddenOutputDesc), __FILE__, __LINE__);
	// cudnnErrCheck(cudnnDestroyTensorDescriptor(m_gradCellOutputDesc), __FILE__, __LINE__);

	for (int i = 0; i < m_seqLength; i++) {
		cudnnErrCheck(cudnnDestroyTensorDescriptor(m_srcDataDesc[i]), __FILE__, __LINE__);
		cudnnErrCheck(cudnnDestroyTensorDescriptor(m_dstDataDesc[i]), __FILE__, __LINE__);
		// cudnnErrCheck(cudnnDestroyTensorDescriptor(m_gradientInputDesc[i]), __FILE__, __LINE__);
		// cudnnErrCheck(cudnnDestroyTensorDescriptor(m_gradientOutputDesc[i]), __FILE__, __LINE__);
	}

    cudnnErrCheck(cudnnDestroy(m_handle), __FILE__, __LINE__);
    cudaErrCheck(cudaDeviceReset(), __FILE__, __LINE__);

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

// To build: nvcc -l cudnn -l curand cudnn_lstm.cu  -o a.out
