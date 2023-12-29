#include "Operator.h"

// index = c*n_row + r

// A = (n, m)   B = (m, l)
//tiled matrix multiplication
__global__ void tiled_matrixMul_kernel(float *res, float *A, float *B, int n, int m, int l) {
  __shared__ float tile1[BLOCK_WIDTH * BLOCK_HEIGHT];
  __shared__ float tile2[BLOCK_WIDTH * BLOCK_HEIGHT];
  int out_row = blockDim.y * blockIdx.y + threadIdx.y;
  int out_col = blockDim.x * blockIdx.x + threadIdx.x;
  float sum = 0;
  for (int i = 0; i < (m + blockDim.x - 1) / blockDim.x; ++i) {
    int A_idx = (i * blockDim.x + threadIdx.x) * n + out_row;
    int B_idx = out_col * m + i * blockDim.y + threadIdx.y;
    if (out_row < n && i * blockDim.x + threadIdx.x < m)
      tile1[threadIdx.y * blockDim.x + threadIdx.x] = A[A_idx];
    else
      tile1[threadIdx.y * blockDim.x + threadIdx.x] = 0;
    if (i * blockDim.y + threadIdx.y < m && out_col < l)
      tile2[threadIdx.y * blockDim.x + threadIdx.x] = B[B_idx];
    else
      tile2[threadIdx.y * blockDim.x + threadIdx.x] = 0;
    //waiting until all cells in SMEM are assigned
    __syncthreads();
    for (int j = 0; j < blockDim.x; ++j) {
      sum += tile1[threadIdx.y * blockDim.x + j] * tile2[j * blockDim.x + threadIdx.x];
    }
    //waiting until all values in SMEM are processed
    __syncthreads();
  }
  if (out_row < n && out_col < l)
    res[out_col * n + out_row] = sum;
}

__global__ void matrixMul_kernel(float *res, float *A, float *B, int n, int m, int l) {
  int out_row = blockDim.y * blockIdx.y + threadIdx.y;
  int out_col = blockDim.x * blockIdx.x + threadIdx.x;
  float sum = 0;
  if (out_row < n && out_col < l) {
    for (int i = 0; i < m; ++i) {
      sum += A[i * n + out_row] * B[out_col * m + i];

    }
    res[out_col * n + out_row] = sum;
  }
}

__global__ void matrixColwiseAddVec_kernel(float *des, float *vec, int n, int m) {
  int out_row = blockIdx.y * blockDim.y + threadIdx.y;
  int out_col = blockIdx.x * blockDim.x + threadIdx.x;
  if (out_row < n && out_col < m) {
    des[out_col * n + out_row] += vec[out_row];
  }
}

__global__ void matrixRowwiseAddVec_kernel(float* des, float* vec, int n, int m) {
  int out_row = blockIdx.y * blockDim.y + threadIdx.y;
  int out_col = blockIdx.x * blockDim.x + threadIdx.x;
  if (out_row < n && out_col < m) {
    des[out_col * n + out_row] += vec[out_col];
  }
}
void dev_matrixMul(float *res, float *A, float *B, int n, int m, int l) {
  size_t A_size = sizeof(float) * n * m;
  size_t B_size = sizeof(float) * m * l;
  size_t res_size = sizeof(float) * n * l;
  //allocate dev memory
  float* d_A = nullptr;
  float* d_B = nullptr;
  float* d_res = nullptr;
  CHECK(cudaMalloc(&d_A, A_size));
  CHECK(cudaMalloc(&d_B, B_size));
  CHECK(cudaMalloc(&d_res, res_size));
  //data transfer from host to device
  CHECK(cudaMemcpy(d_A, A, A_size, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_B, B, B_size, cudaMemcpyHostToDevice));
  //call kernel
  //default block size: 32 x 32
  dim3 block_size(BLOCK_WIDTH, BLOCK_HEIGHT);
  dim3 grid_size((l + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y);
  tiled_matrixMul_kernel<<<grid_size, block_size>>>(d_res, d_A, d_B, n, m, l);
  // matrixMul_kernel<<<grid_size, block_size>>>(d_res, d_A, d_B, n, m, l);
  //data transfer from device back to host
  CHECK(cudaMemcpy(res, d_res, res_size, cudaMemcpyDeviceToHost));
  //free dev memory
  CHECK(cudaFree(d_A));
  CHECK(cudaFree(d_B));
  CHECK(cudaFree(d_res));
}

// des = (n, m) vec = (n)
void dev_matrixColwiseAddVec(float *des, float *vec, int n, int m) {
  float* d_des = nullptr;
  float* d_vec = nullptr;
  //allocate dev memory
  CHECK(cudaMalloc(&d_des, sizeof(float) * n * m));
  CHECK(cudaMalloc(&d_vec, sizeof(float) * n));
  //data transfer from host to device
  CHECK(cudaMemcpy(d_des, des, sizeof(float) * n * m, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_vec, vec, sizeof(float) * n, cudaMemcpyHostToDevice));
  //call kernel
  //default block size: 32 x 32
  dim3 block_size(BLOCK_WIDTH, BLOCK_HEIGHT);
  dim3 grid_size((m + block_size.x -1) / block_size.x, (n + block_size.y - 1) / block_size.y);
  matrixColwiseAddVec_kernel<<<grid_size, block_size>>>(d_des, d_vec, n , m);
  //data transfer from device back to host
  CHECK(cudaMemcpy(des, d_des, sizeof(float) * n * m, cudaMemcpyDeviceToHost));
  //free dev memory
  CHECK(cudaFree(d_des));
  CHECK(cudaFree(d_vec));
}

// des = (n, m) vec = (m)
void dev_matrixRowwiseAddVec(float *des, float *vec, int n, int m) {
  float* d_des = nullptr;
  float* d_vec = nullptr;
  //allocate dev memory
  CHECK(cudaMalloc(&d_des, sizeof(float) * n * m));
  CHECK(cudaMalloc(&d_vec, sizeof(float) * m));
  //data transfer from host to device
  CHECK(cudaMemcpy(d_des, des, sizeof(float) * n * m, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_vec, vec, sizeof(float) * m, cudaMemcpyHostToDevice));
  //call kernel
  //default block size: 32 x 32
  dim3 block_size(BLOCK_WIDTH, BLOCK_HEIGHT);
  dim3 grid_size((m + block_size.x -1) / block_size.x, (n + block_size.y - 1) / block_size.y);
  matrixColwiseAddVec_kernel<<<grid_size, block_size>>>(d_des, d_vec, n , m);
  //data transfer from device back to host
  CHECK(cudaMemcpy(des, d_des, sizeof(float) * n * m, cudaMemcpyDeviceToHost));
  //free dev memory
  CHECK(cudaFree(d_des));
  CHECK(cudaFree(d_vec));
}

//////////////////////////////////////////////////////////////////////// Convolution using Cuda ///////////////////////////////////////

// input size: (height_in * width_in * channel_in)
// data size: (hw_out * hw_kernel * channel_in)
__global__ void im2col(float* input, float* data, int height_in, int width_in, int channel_in, int height_kernel, int width_kernel, 
			int height_out, int width_out, int channel_out, int stride)
{	
	int i = blockIdx.y * blockDim.y + threadIdx.y;   // row: 0 - hw_out
	int j = blockIdx.x * blockDim.x + threadIdx.x;   // col: 0 - channel_out
	
	int hw_in = height_in * width_in;
	int hw_kernel = height_kernel * width_kernel;
	int hw_out = height_out * width_out;
	
	if (i < hw_out && j < channel_out)
	{
		if (threadIdx.x == 0)
		{
			for (int c = 0; c < channel_in; c++) 
			{
				int step_h = i / width_out;
				int step_w = i % width_out;
				int start_idx = step_h * width_in * stride + step_w * stride;  
				for (int k = 0; k < hw_kernel; k ++) 
				{
					int cur_col = start_idx % width_in + k % width_kernel; 
					int cur_row = start_idx / width_in + k / width_kernel;
					if (cur_col < 0 || cur_col >= width_in || cur_row < 0 || cur_row >= height_in) 
					{
						data[i * hw_kernel * channel_in + c * hw_kernel + k] = 0;
					}
					else 
					{
						int pick_idx = hw_in * c + cur_row * width_in + cur_col;
						data[i * hw_kernel * channel_in + c * hw_kernel + k] = input[pick_idx];
					}
				}	
			}
		}
	}
}

// data size (m, n) - (hw_out, hw_kernel * channel_in)
// weight size (n, k) - (hw_kernel * channel_in, channel_out)
// output size (m, k) - (hw_out, channel_out)
// bias size (k) - (channel_out)

__global__ void convolution(float* data, float* weight, float* output, float* bias, int m, int n, int k)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
    	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < m && j < k)
	{
		float s = 0;
		for (int p = 0; p < n; p++)
		{
			//s += data[i * n + p] * weight[p * k + j];
			s += data[i * n + p] * weight[j * n + p];
		}
		output[i * k + j] = s + bias[j];
	        // output[i * k + j] = s;
	}
}

__global__ void convolution_kernel2(float* data, float* weight, float* output, float* bias, int m, int n, int k)
{
	__shared__ float s_data[TILE_WIDTH][TILE_WIDTH];    //BLOCK HEIGHT, BLOCK WIDTH
	__shared__ float s_weight[TILE_WIDTH][TILE_WIDTH];  //BLOCK HEIGHT, BLOCK WIDTH
	
	int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	float s = 0;
	
	for (int t = 0; t < (n - 1) / TILE_WIDTH + 1; t++)
	{
		if (i < m && t * TILE_WIDTH + tx < n)
		{
			s_data[ty][tx] = data[i * n + t * TILE_WIDTH + tx];
		}
		else
		{
			s_data[ty][tx] = 0;
		}
		
		if (t * TILE_WIDTH + ty < n && j < k)
		{
			s_weight[ty][tx] = weight[(t * TILE_WIDTH + ty) * k + j];
		}
		else
		{
			s_weight[ty][tx] = 0;
		}
		__syncthreads();
		
		
		for (int p = 0; p < TILE_WIDTH; p++)
		{
			s+= s_data[ty][p] * s_weight[p][tx];
		}
		__syncthreads();
	}	
		
	if (i < m && j < k)
	{
		output[i * k + j] = s + bias[j];
		// output[i * k + j] = s;
	}
}


// Input in: (ch_in * h_in * w_in); index = ch_idx*(h_in * w_in) + h_idx*w_in + w_idx
// Output out: (ch_out * h_out * w_out); index = ch_idx*(h_out * w_out) + h_idx*w_out + w_idx
// Weight wei: (ch_out * ch_in * h_ker * w_ker); index = ch_out_idx*(ch_in * h_ker * w_ker) + ch_in_idx*(h_ker * w_ker) + h_ker_idx*w_ker + w_ker_idx
// Bias bias: (ch_out); index = ch_out_idx
void dev_convForward(float *out, float *in, float *wei, float *bias,
                      int h_in, int w_in, int ch_in, int h_out, int w_out, int ch_out, int h_ker, int w_ker, int stride) {
  int hw_out = h_out * w_out;
  int hw_ker = h_ker * w_ker;

  //Allocate memories
  float *d_data, *d_output, *d_input, *d_weight, *d_bias;
  size_t n_data = ((ch_in * hw_out) * (hw_ker)) * sizeof(float);
  size_t n_output = (ch_out * hw_out) * sizeof(float);
  size_t n_input = (ch_in * h_in * w_in) * sizeof(float);
  size_t n_weight = (ch_out * ch_in * hw_ker) * sizeof(float);
  size_t n_bias = (ch_out) * sizeof(float);
  CHECK(cudaMalloc(&d_data, n_data));
  CHECK(cudaMalloc(&d_output, n_output));
  CHECK(cudaMalloc(&d_input, n_input));
  CHECK(cudaMalloc(&d_weight, n_weight));
  CHECK(cudaMalloc(&d_bias, n_bias));

  //TODO: Copy data from in to d_input
  CHECK(cudaMemcpy(d_input, in, n_input, cudaMemcpyHostToDevice));
  //TODO: Copy data from weight to d_weight
  CHECK(cudaMemcpy(d_weight, wei, n_weight, cudaMemcpyHostToDevice));
  //TODO: Copy data from bias to d_bias
  CHECK(cudaMemcpy(d_bias, bias, n_bias, cudaMemcpyHostToDevice));

  //Grid size and Block size
  dim3 blockSize (32, 32); //default
  dim3 gridSize((ch_out - 1) / blockSize.x + 1,
          (hw_out - 1) / blockSize.y + 1);

  im2col<<<gridSize, blockSize>>>(d_input, d_data, h_in, w_in, ch_in, h_ker, w_ker, h_out, w_out, ch_out, stride);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaGetLastError());
  convolution<<<gridSize, blockSize>>>(d_data, d_weight, d_output, d_bias, hw_out, hw_ker * ch_in, ch_out);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaGetLastError());

  //TODO: Copy data from d_output to out
  CHECK(cudaMemcpy(out, d_output, n_output, cudaMemcpyDeviceToHost));

  // Free data
  CHECK(cudaFree(d_data));
  CHECK(cudaFree(d_output));
  CHECK(cudaFree(d_input));
  CHECK(cudaFree(d_weight));
  CHECK(cudaFree(d_bias));
}
