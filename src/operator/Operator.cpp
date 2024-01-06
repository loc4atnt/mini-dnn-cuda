#include "Operator.h"

Matrix optimizedMatrixMul(const Matrix &A, const Matrix &B, const Vector &bias, bool isColWise) {
  Matrix result;
  result.resize(A.rows(), B.cols());
  dev_matrixMulWithBias((float*)result.data(), (float*)A.data(), (float*)B.data(), (float*)bias.data(), A.rows(), A.cols(), B.cols(), isColWise);
  return result;
}

Matrix matrixMul(const Matrix &A, const Matrix &B) {
  Matrix result;
  result.resize(A.rows(), B.cols());
  dev_matrixMul((float*)result.data(), (float*)A.data(), (float*)B.data(), A.rows(), A.cols(), B.cols());
  return result;
}

void matrixColwiseAddVec(Matrix &des, const Vector &vec) {
  dev_matrixColwiseAddVec((float*)des.data(), (float*)vec.data(), des.rows(), des.cols());
}

void matrixRowwiseAddVec(Matrix &des, const Vector &vec) {
  dev_matrixRowwiseAddVec((float*)des.data(), (float*)vec.data(), des.rows(), des.cols());
}