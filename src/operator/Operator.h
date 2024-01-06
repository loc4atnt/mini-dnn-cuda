#ifndef _Operator_H_
#define _Operator_H_

#include <Eigen/Core>
#include "../utils.h"
#include "../device/Operator.h"

Matrix matrixMulAndAddBiasColwise(const Matrix &A, const Matrix &B, const Vector &bias, bool isColWise = true, bool usingDevice=false, bool usingOpt=false);
void matrixColwiseAddVec(Matrix &des, const Vector &vec, bool usingDevice=false);
void matrixRowwiseAddVec(Matrix &des, const Vector &vec, bool usingDevice=false);

#endif