#ifndef _Operator_H_
#define _Operator_H_

#include <Eigen/Core>
#include "../utils.h"
#include "../device/Operator.h"

Matrix optimizedMatrixMul(const Matrix &A, const Matrix &B, const Vector &bias, bool isColWise = true);
Matrix matrixMul(const Matrix &A, const Matrix &B);
void matrixColwiseAddVec(Matrix &des, const Vector &vec);
void matrixRowwiseAddVec(Matrix &des, const Vector &vec);

#endif