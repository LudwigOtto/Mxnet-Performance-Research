#CUDA
sudo rpm -i cuda-repo-rhel7-9-2-local-9.2.148-1.x86_64.rpm
sudo rpm -i cuda-repo-rhel7-9-2-148-local-patch-1-1.0-1.x86_64.rpm
sudo yum clean all
sudo yum install cuda
nvidia-smi
nvidia-smi -q

#CUDNN
tar -xzvf cudnn-9.2-linux-x64-v7.3.1.20.tgz
$ sudo cp cuda/include/cudnn.h /usr/local/cuda/include
$ sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
$ sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*



git checkout -b v1.2.0 --track origin/v1.2.0


cp .ykao-key-001.pem .ssh/
cp backup_bashrc ~/.bashrc

# Distributed Training for MXNet

User guide for MXNet on Seawulf

## Getting Started
Apache MXNet (incubating) is a deep learning framework compatible with both symbolic-style and imperative-style programming.
https://github.com/apache/incubator-mxnet

This note introduces:
1. Required environment confiugure for MXNet on Seawulf clusters.
2. How to train classic data set (i.e. cifar-10 and imagenet) by using MXNet example codes
3. How to use MXNet API to generate training model

### Prerequisites
###### [May 3rd, 2018] The following confugre setting is already added to .bashrc file. You can skip this part now.

To install MXNet or run MXNet training on the seawulf, the following 5 models should be loaded first:
1. anaconda/2 
2. cuda80/toolkit/8.0.44
3. cudnn/5.1
4. torque/6.0.2
5. shared
```
module load anaconda/2 cuda80 cudnn/5.1 torque/6.0.2 shared
```
After loading the modules, you can echo $LD_LIBRARY_PATH to confirm that the required modules exists now.
```
echo $LD_LIBRARY_PATH | grep cudnn; echo $LD_LIBRARY_PATH | grep cuda
```
### Installing
###### [May 3rd, 2018] Currently MXNet version on Seawulf is v1.1.0 with cuda80. Once Seawulf is ready is for cuda90 and cudnn 7, you can reinstall MXNet again. Now you can skip this part. 

Following is the complete introduction of how to install MXNet, here is a brief digest for reference.
https://mxnet.incubator.apache.org/install/index.html

As long as you load 'Anaconda/2' module, it is assumed that you should have no problem to use 'virtualenv' and 'pip'. The first step is using virtualenv to create MXNet virtual environment.
```
virtualenv --system-site-packages $dir/name_of_mxnet_env
source $dir/name_of_mxnet_env/bin/activate
```
$dir is the path that you wish to locate your virtual environment.
The second step is using pip to install MXNet package in the active environment. Here I only show GPU version of MXNet package. (MXNet of GPU version can ONLY run on the CUDA-feasible machines)
```
(name_of_mxnet_env)$ pip install --upgrade pip
(name_of_mxnet_env)$ pip install mxnet-cu90 --force-reinstall
```
To validate whether the installation is success or not, please refer to the next section about how to use MXNet on Seawulf GPU nodes.

## Using MXNet on Seawulf GPU nodes
### Activating virtual environment
###### [May 3rd, 2018] The following confugre setting is already added to .bashrc file. You can skip this part now.
Before you use MXNet, please always remember to load the required modules and to enable the virtual environment.
```
module load anaconda/2 cuda80 cudnn/5.1 torque/6.0.2 shared
source $dir/name_of_mxnet_env/bin/activate
```
### GPU on Seawulf
On Seawulf there are 8 GPU nodes and each contains 4 Nvidia Tesla K80 GPUs (2x GK210 Cores per GPU). Seawulf uses PBS queue system to dynamically distribute jobs. There are 2 GPU queues we can submit jobs to: 
```
Queue       Default run time    Max run time    Max # of nodes  Min # of nodes
gpu             1 hour          8 hours             2               n/a
gpu-long        8 hours         48 hours            2               n/a
```
You can use 'qstat -q [name_of_queue]' to check the queue status. It's highly probable that the nodes are all occupied and the queue is full of waiting jobs.
```
$ qstat -q gpu
server: master.cm.cluster

Queue            Memory CPU Time Walltime Node  Run Que Lm  State
---------------- ------ -------- -------- ----  --- --- --  -----
gpu                --      --    08:00:00   --    0   0 --   E R
                                               ----- -----
                                                   0     0
```
### Obtaining GPU node
If your training only requires single machine, you can simply submit the training job. Refer to the following link for how to draft PBS script.
https://it.stonybrook.edu/help/kb/example-pbs-job-script
For multiple nodes, please submit the interactive shell.
```
$ qsub -q gpu -I -V -l walltime=02:00:00,nodes=2:ppn=20
qsub: waiting for job 213061.sn-mgmt.cm.cluster to start
qsub: job 213061.sn-mgmt.cm.cluster ready
```
You can use the job name '213061' to check the names of all node you have.
```
$qnodes | grep -n 213061
5221:     status = rectime=1525487560,cpuclock=Fixed,varattr=,jobs=213061.sn-mgmt.cm.cluster,state=free,netload=7093456129300,gres=,loadave=0.00,ncpus=28,physmem=131939556kb,availmem=120710232kb,totmem=131939556kb,idletime=15088,nusers=0,nsessions=0,uname=Linux sn-nvda3 3.10.0-327.3.1.el7.x86_64 #1 SMP Wed Dec 9 14:09:15 UTC 2015 x86_64,opsys=linux
5295:     status = rectime=1525487559,cpuclock=Fixed,varattr=,jobs=213061.sn-mgmt.cm.cluster(cput=0,energy_used=0,mem=9852kb,vmem=473040kb,walltime=279,Error_Path=/dev/pts/0,Output_Path=/dev/pts/0,session_id=12404),state=free,netload=3485314871412,gres=,loadave=0.01,ncpus=28,physmem=131939552kb,availmem=120800296kb,totmem=131939552kb,idletime=1592383,nusers=1,nsessions=1,sessions=12404,uname=Linux sn-nvda7 3.10.0-327.3.1.el7.x86_64 #1 SMP Wed Dec 9 14:09:15 UTC 2015 x86_64,opsys=linux
```
The nodes you get in this example are sn-nvda7 and sn-nvda3.
### Running the tests
Here I use the example from the official installation guide to show how to do the validation. If all the above steps work well, you should see the output of 2-D list.
```
>>> import mxnet as mx
>>> a = mx.nd.ones((2, 3), mx.gpu())
>>> b = a * 2 + 1
>>> b.asnumpy()
array([[ 3.,  3.,  3.],
       [ 3.,  3.,  3.]], dtype=float32)
```

## Running the example of image classification
###### All the path variables and commands can be found in $HOME/ykao/run_mxnet_sample.sh
### dataset and source code path
```
MX_DIR=$HOME/ykao/Mxnet-Src/incubator-mxnet/example/image-classification
TOOL_DIR=$HOME/ykao/Mxnet-Src/incubator-mxnet/tools
IMGNET_DIR=/gpfs/scratch/yoshen/ykao/imagenet_mxnet/compas
CIFAR_DIR=/gpfs/scratch/yoshen/ykao/cifar10_mxnet
BW_DIR=$HOME/ykao/Mxnet-Src/incubator-mxnet/tools/bandwidth/
```
### Training Imagenet on single node
Following shows the command and argument detail:
```
python ${MX_DIR}/train_imagenet.py --data-train ${IMGNET_DIR}/imagenet1k-train.rec --data-val ${IMGNET_DIR}/imagenet1k-val.rec --batch-size 128 --network resnet --num-layers 50 --kv-store local --data-nthreads 8
```
```
'--network',        type=str, help='the neural network to use'
'--num-layers',     type=int, help='number of layers in the neural network, required by some networks such as resnet'
'--gpus',           type=str, help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu'
'--kv-store',       type=str, default='device', help='key-value store type'
'--num-epochs',     type=int, default=100, help='max num of epochs'
'--lr',             type=float, default=0.1, help='initial learning rate'
'--lr-factor',      type=float, default=0.1, help='the ratio to reduce lr on each step'
'--lr-step-epochs', type=str, help='the epochs to reduce the lr, e.g. 30,60'
'--initializer',    type=str, default='default', help='the initializer type'
'--optimizer',      type=str, default='sgd', help='the optimizer type'
'--mom',            type=float, default=0.9, help='momentum for sgd'
'--wd',             type=float, default=0.0001, help='weight decay for sgd'
'--batch-size',     type=int, default=128, help='the batch size'
'--disp-batches',   type=int, default=20, help='show progress for every n batches'
'--model-prefix',   type=str, help='model prefix')
'--monitor', dest='monitor', type=int, default=0, help='log network parameters every N iters if larger than 0'
'--load-epoch',     type=int, help='load the model on an epoch using the model-load-prefix'
'--top-k',          type=int, default=0, help='report the top-k accuracy. 0 means no report.'
'--test-io',        type=int, default=0, help='1 means test reading speed without training'
'--dtype',          type=str, default='float32', help='precision: float32 or float16'
'--gc-type',        type=str, default='none', help='type of gradient compression to use, takes `2bit` or `none` for now'
'--gc-threshold',   type=float, default=0.5, help='threshold for 2bit gradient compression'
    # additional parameters for large batch sgd
'--macrobatch-size', type=int, default=0, help='distributed effective batch size'
'--warmup-epochs',  type=int, default=5, help='the epochs to ramp-up lr to scaled large-batch value'
'--warmup-strategy', type=str, default='linear', help='the ramping-up strategy for large batch sgd'
```
### Reading log of imagenet training
By default, the output log is directly displayed on the STDOUT. But considering that it is quite time-consuming to get sufficient log for training imagenet, I store the log output to a file in $HOME/ykao/log/. If your training is over cifar-10 or mnist, you can ignore this part.
```
$ tail log/train_imagenet.log 
INFO:root:Epoch[0] Batch [40]	Speed: 27.98 samples/sec	accuracy=0.000781
```
### Training Imagenet on multiple nodes
Please refer to the following link for technique details.
https://mxnet.incubator.apache.org/versions/master/faq/distributed_training.html

Training on multiple devices requires a launcher to establish connection with all servers, distribute dataset per epoch, and manage the transmission of gradient. MXNet's launcher is located at $HOME/ykao/Mxnet-Src/incubator-mxnet/tools and support many interfaces like ssh (default) and mpi. You need to let launcher know all hosts' names or IPs by giving a list file (${HOME}/ykao/Hosts), and then the launcher will asynchronously create a number of processes. (corresponds to the number of worker you set) That means if you want to terminate the training, you cannot just terminate the launcher but also have to terminate the processes on each server. Please ssh to each server and manually kill the processes. Following is the example of creating a residual netwrok training for imagenet on 4 servers.

###### Please remember the host of launcher must also be the GPU node because MXNet regards it as one of the training server nodes. 
###### Only after obtaining GPU nodes will you know the host names or IPs. That's why you need to submit STDIN to PBS queue when doing distributed training. 
\
```
#python ${TOOL_DIR}/launch.py -n 4 -s 4 --hostfile ${HOME}/ykao/Hosts --sync-dst-dir ${HOME}/ykao python ${MX_DIR}/train_imagenet.py \
#--gpus 0,1,2,3,4,5,6,7 --num-epochs 1 --data-train ${IMGNET_DIR}/imagenet1k-train.rec --data-val ${IMGNET_DIR}/imagenet1k-val.rec \
#--batch-size 64 --network resnet --num-layers 152 --kv-store dist_sync --data-nthreads 8
```
```
$ cat ${HOME}/ykao/Hosts 
sn-nvda1
sn-nvda2
sn-nvda3
sn-nvda4
```
The arguments with definition shown below are required for the launcher. 
```
'-n', '--num-workers',      required=True, type=int, help = 'number of worker nodes to be launched'
'-s', '--num-servers',      type=int, help = 'number of server nodes to be launched, in default it is equal to NUM_WORKERS'
'-H', '--hostfile',         type=str, help = 'the hostfile of slave machines which will run the job. Required for ssh and mpi launcher'
'--sync-dst-dir',           type=str, help = 'if specificed, it will sync the current directory into slave machines\'s SYNC_DST_DIR if ssh launcher is used'
'--launcher',               type=str, default='ssh', choices = ['local', 'ssh', 'mpi', 'sge', 'yarn'], help = 'the launcher to use'
```
In the log you can see a set of repeated batches. They correspond to the performance on each server you call and need not follow the order in the list file of hosts. (That means you can't trace the performance for any specific node from the log unless you modify the output format) Since the dataset is equally distributed to each computing server, the total batch number per epoch in this example is one quater of the number in single machine training.
```
$ tail log/train_imagenet.log
INFO:root:Epoch[0] Batch [20]   Speed: 12.13 samples/sec        accuracy=0.000744
INFO:root:Epoch[0] Batch [20]   Speed: 12.07 samples/sec        accuracy=0.000000
INFO:root:Epoch[0] Batch [20]   Speed: 12.10 samples/sec        accuracy=0.000000
INFO:root:Epoch[0] Batch [20]   Speed: 11.85 samples/sec        accuracy=0.001488
INFO:root:Epoch[0] Batch [40]   Speed: 11.46 samples/sec        accuracy=0.001563
INFO:root:Epoch[0] Batch [40]   Speed: 11.46 samples/sec        accuracy=0.000781
INFO:root:Epoch[0] Batch [40]   Speed: 11.47 samples/sec        accuracy=0.001563
INFO:root:Epoch[0] Batch [40]   Speed: 11.48 samples/sec        accuracy=0.000781
```

## Acceleration
### Gradient Synchronization
https://mxnet.incubator.apache.org/versions/master/faq/distributed_training.html#different-modes-of-distributed-training

### Gradient Compression
https://mxnet.incubator.apache.org/versions/master/faq/distributed_training.html#gradient-compression


