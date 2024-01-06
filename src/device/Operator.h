#ifndef _DeviceOperator_H_
#define _DeviceOperator_H_

#include <cuda_runtime.h>
#include "Util.h"

#define BLOCK_WIDTH 32
#define BLOCK_HEIGHT 32

#define TILE_WIDTH 32

#define MAX_BIAS_SIZE 1024
#define BIAS_SIZE 16

// A = (n, m)   B = (m, l)
void dev_matrixMulAndAddBias(float *res, float *A, float *B, float *bias, int n, int m, int l, bool isColWise);

// A = (n, m)   B = (m, l)
void dev_matrixMul(float *res, float *A, float *B, int n, int m, int l);

// des = (n, m) vec = (n)
void dev_matrixColwiseAddVec(float *des, float *vec, int n, int m);

// des = (n, m) vec = (m)
void dev_matrixRowwiseAddVec(float *des, float *vec, int n, int m);

// Input in: (ch_in * h_in * w_in); index = ch_idx*(h_in * w_in) + h_idx*w_in + w_idx
// Output out: (ch_out * h_out * w_out); index = ch_idx*(h_out * w_out) + h_idx*w_out + w_idx
// Weight wei: (ch_out * ch_in * h_ker * w_ker); index = ch_out_idx*(ch_in * h_ker * w_ker) + ch_in_idx*(h_ker * w_ker) + h_ker_idx*w_ker + w_ker_idx
// Bias bias: (ch_out); index = ch_out_idx
void dev_convForward(float *out, float *in, float *wei, float *bias,
                      int h_in, int w_in, int ch_in, int h_out, int w_out, int ch_out, int h_ker, int w_ker, int stride, bool usingOpt);

#endif