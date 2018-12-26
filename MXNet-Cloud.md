# Using MXNet in Cloud Environment
Following libraries of specific version are required for MXnet 1.2.1 in GPU environment:

- CUDA 9.2
- CUDNN 7.x (Not necessary but recommend because it really helps acceleration a lot)
- Intel MKL (Optional)

Some clouds provides CUDA images so that you don't have to manually install the
libraries. But every Mxnet's pip package can only work with its corresponding
CUDA version. [Check the correct version of package here.](#mxnet-installation)

If you don't wanna use the provided images or you cannot find an adequate one,
please follow the next steps.

## CUDA Installation
[Reference](https://developer.nvidia.com/cuda-92-download-archive?target_os=Linux)
* Example Cent OS 7
```
$ wget https://developer.nvidia.com/compute/cuda/9.2/Prod2/local_installers/cuda-repo-rhel7-9-2-local-9.2.148-1.x86_64
$ sudo rpm -i cuda-repo-rhel7-9-2-local-9.2.148-1.x86_64.rpm
$ sudo yum clean all
$ sudo yum install cuda
```
NVIDIA released one patch for CUDA 9.2 that includes performance improvements to cuBLAS GEMM APIs and bug fixes for CUPTI and cuda-gdb.
Please download it and do the installation again.
```
$ wget https://developer.nvidia.com/compute/cuda/9.2/Prod2/patches/1/cuda-repo-rhel7-9-2-148-local-patch-1-1.0-1.x86_64
```

## CUDNN Installation
First please download the **cuDNN Library for Linux** from [here](https://developer.nvidia.com/cudnn). 
Then install from the tar file:
```
$ tar -xzvf cudnn-7.x-linux-x64-v7.tgz
$ sudo cp cuda/include/cudnn.h /usr/local/cuda/include
$ sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
$ sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
```

## Mxnet Installation
In the cloud environment the instance you have probably exists for a short term,
and single instance is actually in charge of only one task. Therefore, here we
don't use python virtual environment but directly install the package in
`/usr/local`
```
$ sudo pip install --upgrade pip
$ sudo pip install mxnet-cu92
```
Below lists the correct Mxnet version corresponding to the CUDA.

- CUDA 9.2:	mxnet-cu92 (Mxnet 1.2.1, 1.3.1) <br />
- CUDA 9.1:	mxnet-cu91 (Mxnet 1.1.1, 1.2.1, 1.3.1) <br />
- CUDA 9.0:	mxnet-cu90 (Mxnet 1.0.1, 1.1.1, 1.2.1, 1.3.1) <br />
- CUDA 8.0:	mxnet-cu90 (Mxnet 1.0.1, 1.1.1, 1.2.1, 1.3.1 or all legacy version) <br />

To validate whether the installation is success or not, please refer to [the validation with GPU](MXNet-HPC.md#running-the-tests).

## Configuration for distributation training in the cloud
### SSH Agent
- Generate one SSH public RSA key for all nodes when launching instances.

- Copy the key to `.ssh/` every training node.
```
$ scp your_key.pem user@host_on_cloud:~/.ssh/
```
- Add the command to each node's `.bashrc` file for automatically starting SSH agent while login the node
```
if [ -z "$SSH_AUTH_SOCK" ] ; then
  eval `ssh-agent -s`
  ssh-add ${HOME}/.ssh/*.pem
  trap "kill $SSH_AGENT_PID" 0
fi
```

