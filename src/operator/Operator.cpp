#include "Operator.h"

Matrix matrixMulAndAddBiasColwise(const Matrix &A, const Matrix &B, const Vector &bias, bool isColWise, bool usingDevice) {
  if (usingDevice) {
    Matrix result;
    result.resize(A.rows(), B.cols());
    dev_matrixMulAndAddBias((float*)result.data(), (float*)A.data(), (float*)B.data(), (float*)bias.data(), A.rows(), A.cols(), B.cols(), isColWise);
    return result;
  } else {
    return A * B;
  }
}

void matrixColwiseAddVec(Matrix &des, const Vector &vec, bool usingDevice) {
  if (usingDevice) {
    dev_matrixColwiseAddVec((float*)des.data(), (float*)vec.data(), des.rows(), des.cols());
  } else {
    des.colwise() += vec;
  }
}

void matrixRowwiseAddVec(Matrix &des, const Vector &vec, bool usingDevice) {
  if (usingDevice) {
    dev_matrixRowwiseAddVec((float*)des.data(), (float*)vec.data(), des.rows(), des.cols());
  } else {
    des.rowwise() += vec.transpose();
  }
}