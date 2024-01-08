# mini-dnn-cuda
**mini-dnn-cuda** is a CUDA/C C++ demo of deep neural networks base on its implementation purely in C++ and use CUDA to improve the convolution layers also fully connected layers

## Usage
Download and unzip [MNIST](http://yann.lecun.com/exdb/mnist/) dataset in `mini-dnn-cpp/data/mnist/`.

```shell
mkdir build
cd build
cmake ..
make
```

Run `./demo`.

Pull source code from [github](https://github.com/loc4atnt/mini-dnn-cuda.git) to your Colab

```shell
%cd /content
rm -rf mini-dnn-cuda
git clone https://github.com/loc4atnt/mini-dnn-cuda.git
```

Build source code

```shell
%cd /content/mini-dnn-cuda
rm -rf build
mkdir build
%cd build
cmake ..
make
```

Run train model
```shell
!./main train param 
```

Run test model
```shell
!./main test param host
```

Run test model in device with optimal/non-optimial version
```shell
!./main test param device
!./main test param device opt
```

Result: 
Parallel LeNet-like CNN can obtain `0.88` accuracy on MNIST testset.
