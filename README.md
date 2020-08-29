# gpu-mpi-pytorch

# (一/二)  GeForce MX150

> 主要是参考的这篇文章：https://codle.net/install-pytorch-from-source/
为了防止文章丢失，这里抄写一遍。

> PyTorch 是一个非常强大的神经网络的框架，为了发挥其最大的效果一般都会结合 GPU 来使用。但是随着相关显卡硬件的发展，官方对于一些老型号显卡的预编译也随之取消了。如果还在使用较老的显卡型号，同时也想使用 PyTorch 的 GPU 支持，那么就需要从源码进行安装了。

## (1/4).本机配置

首先，这里介绍一些本机的实际环境：

操作系统：Ubuntu 18.04 LTS

Python 版本：3.6

笔记本型号：联想 Y50-70

显卡型号：GTX 860m（4G，开普勒架构版本）

内存：16G

## (2/4).安装环境

安装环境主要包含 Python 环境、显卡驱动、CUDA、cuDNN。下面将分别讲述安装的过程。

### Python 环境
PyTorch 是一个 Python 下的神经网络框架，因此首先需要安装 Python 环境。对于 Python 环境，官方推荐使用 Anaconda 工具包。

首先是 Anaconda 的下载，官方下载页面：https://www.anaconda.com/download/#linux ，一般是下载 Python 3.6 版本。

下载是一个 sh 文件，使用以下命令进行安装：

```
sh ./Anaconda3-5.2.0-Linux-x86_64.sh
```

### 显卡驱动
显卡驱动安装是 GPU 支持中比较困难的一步。英伟达的显卡驱动分为两个版本，apt 上的非公版本和英伟达官网的版本。但从我自己的试验结果而言，英伟达的官方驱动很难安装成功，还是推荐使用 apt 来安装。

对于 Ubuntu 16.04 以上的版本，有一个比较简单的方法：

```
sudo ubuntu-drivers autoinstall
```

这是一个 Ubuntu 设备管理器，使用autoinstall可以安装所有设备的最新版本驱动，相当方便。

### CUDA

CUDA 从英伟达官网下载，下载地址：https://developer.nvidia.com/cuda-downloads

选择：Linux->x86_64->Ubuntu->17.10（还没有 18.04 版本，17.10版本可以正常使用）->runfile(local)

这里下载 runfile 文件，不要下载 deb 的版本，因为 deb 版本会自动安装显卡驱动，与 apt 源里面的驱动冲突，会导致安装异常。

runfile 文件可以使用bash cuda_9.2.148_396.37_linux.run来运行，中间大部分都可以选择默认，唯一需要注意的是询问是否安装显卡驱动时，一定要选否。

安装好后，编辑~/.bashrc文件，在后面增加：

```
export PATH=/usr/local/cuda-9.2/bin:${PATH:+:${PATH}}
export CUDA_HOME=/usr/local/cuda-9.2
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CUDNN_HOME}/lib64:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=${CUDA_HOME}/include:${CUDNN_HOME}/include:$CPLUS_INCLUDE_PATH
```

### cuDNN

cuDNN 的下载在https://developer.nvidia.com/cudnn ，需要一个英伟达账号，注册即可。

下载为 tar 包，首先是解压：

```
tar -xzvf cudnn-9.0-linux-x64-v7.tgz
```

然后链接库：

```
sudo cp cuda/include/*.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

```
接着在.bashrc 中添加：

```
export CUDNN_PATH="/usr/local/cuda/lib64/libcudnn.so.7"
export CUDNN_LIBRARY="/usr/local/cuda/lib64/libcudnn.so"
export CUDNN_INCLUDE_DIR="/usr/local/cuda/include/"
export CUDNN_LIB_DIR="/usr/local/cuda/lib64/"
```

## (3/4).安装 PyTorch

需要注意的一点

源码需要在本机上进行编译，因此需要电脑安装 GCC 和 G++。针对 GCC 的版本，PyTorch 要求 GCC 低于 7 版本，而 CUDA 9.2 要求高于 6 版本，因此推荐使用 6 版本。

### 安装gcc,g++：

```
sudo apt install gcc-6 g++-6 gcc-6-multilib g++-6-multilib
```

设置权重：

```
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 50
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-6 50
```

配置当前的 GCC、G++ 版本：

```
sudo update-alternatives --config gcc
sudo update-alternatives --config g++
```

根据提示框选择相应版本即可。

### 安装依赖项目

首先安装依赖：

```
export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" # [anaconda root directory]

conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing
conda install -c mingfeima mkldnn

conda install -c pytorch magma-cuda90 
```

### 下载源码

需要安装 Git 软件，没有安装使用sudo apt install git进行安装。

执行下面代码，将 PyTorch 拷贝下来：

```
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
```

### 安装

进入文件夹，安装 PyTorch。

```
cd pytorch
python setup.py install
```

大概需要等 2 个小时左右，取决于 CPU 的速度。

## (4/4).测试

离开pytorch 的源文件路径，打开ipython

![测试](https://github.com/HongdaChen/gpu-mpi-pytorch/blog/master/003.png)


## （二/二）GeForce GT                                             gcc-7.5

- (1/10)安装nvidia-driver 440 （具体安装什么版本要根据自己的显卡查）安装完成后，运行< nvidia-smi>,发现需要重启。最好在这里就重启，否则可能以后的都是徒劳。

```
sudo apt install nvidia-driver-440
```

- (2/10)安装一些必要的工具vim , git 

```
bash tools_install.sh
```

- (3/10)安装anaconda

```
bash Anaconda.........
```

- (4/10)安装一些依赖包，其中openmpi是分布式通信的backend之一，只有采用源码安装，才可以使用这个backend。

```
bash conda_install.sh
```

- (5/10)添加一个环境变量

```
export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
```


- (6/10)安装cuda

```
bash cuda_install.sh
```

运行之后，会有提示。。。不存在，进入那个文件，把第一行删除。然后执行

```
sudo apt-get -y install cuda
```
 - (7/10)安装cudnn   ( 其实就是解压，然后将文件复制到一些目录下。对于这个版本来说，要将cuda/include/*.h都复制到/usr/local/cuda/include/下，说不准会用到哪一个，这样保险)
 
 ```
bash  cudnn_cp.sh
```

- (8/10)添加环境变量

```
将 bashrc_add.txt中的内容添加到~/.bashrc的后面
source  ~/.bashrc
```

- (9/10)下载pytorch和 vision代码库，必须使用git工具

```
bash pytorch_vision_clone.sh
```

- (10/10)pytorch 和 vision 的安装都必须是源码安装，而且顺序上必须先专装pytorch。他们的安装步骤是一样的：

   - 1。查看历史版本<git tag> 
   - 2。选择合适的版本< git checkout [某个tag]> 
   - 3。<git submodule sync> 
   - 4。<git submodule update --init --recursive > 
   - 5。<python setup.py install> 
  
  在编译过程中，如果速度快，大约1个小时左右。
  
****
  
  需要注意的地方：
  
- 1。gcc，g++的版本既要和cuda兼容，也要和pytorch的特定版本兼容，发现不符合时，优先考虑更换pytorch版本。
  
- 2。nvidia-driver版本要和cuda版本兼容
  
 ![nvidia-driver版本要和cuda版本对照表](https://github.com/HongdaChen/gpu-mpi-pytorch/blog/master/001.png)
  
- 3。vision要和pytorch匹配
 
 ![vison和pytorch版本对照表](https://github.com/HongdaChen/gpu-mpi-pytorch/blog/master/002.png)

- 4。由于从github下载特别慢，所以选择的是gitee.com的镜像，很快。同时推荐gitee.com的git教程。

- 5。当编译报错终止时，在进行下一次编译之前要python setup.py clean

- 6。如果torchvision直接使用pip或者conda安装的话，之前成功安装的torch会被冲刷掉。
