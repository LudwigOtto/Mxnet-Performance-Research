# Using MXNet in HPC Environment
### Edit on Nov 30th, 2018

To install MXNet or run MXNet training on HPC, the following 5 models should be loaded first:
- shared
- anaconda/2 
- cuda91 (Seawulf)
- cudnn/7.4.5
- torque/6.0.2

```
module load shared anaconda/2 cuda91 cudnn/7.4.5 torque/6.0.2
```
After loading the modules, you can `echo $LD_LIBRARY_PATH` to confirm that the required modules exists now.
```
echo $LD_LIBRARY_PATH | grep cudnn; echo $LD_LIBRARY_PATH | grep cuda
```
## Installing

Following is the complete introduction of how to install MXNet, [or you can look here for detail](https://mxnet.incubator.apache.org/install/index.html)

Please refer [here to match the version of MXNet to your CUDA](MXNet-Cloud.md/#mxnet-installation)

As long as you load `anaconda/2` module, it is assumed that you should have no problem to use `virtualenv` and `pip`. The first step is using virtualenv to create MXNet virtual environment.
```
virtualenv --system-site-packages $dir/name_of_mxnet_env
source $dir/name_of_mxnet_env/bin/activate
```
`$dir` is the path that you wish to locate your virtual environment.
The second step is using pip to install MXNet package in the active environment. Here I only show GPU version of MXNet package. (MXNet of GPU version can ONLY run on the CUDA-feasible machines)
```
(name_of_mxnet_env)$ pip install --upgrade pip
(name_of_mxnet_env)$ pip install mxnet-cu91
```
To validate whether the installation is success or not, please refer to the next section.

## Using MXNet on HPC GPU nodes
### Activating virtual environment
Before you use MXNet, please always remember to load the required modules and to enable the virtual environment.
```
module load anaconda/2 cuda91 cudnn/7.4.5 torque/6.0.2 shared
source $dir/name_of_mxnet_env/bin/activate
```
### GPU on Seawulf
On Seawulf there are 8 GPU nodes and each contains 4 Nvidia Tesla K80 GPUs (2x GK210 Cores per GPU). Seawulf uses PBS queue system to dynamically distribute jobs. There are 2 GPU queues we can submit jobs to: 
```
Queue       Default run time    Max run time    Max # of nodes  Min # of nodes
gpu             1 hour          8 hours             2               n/a
gpu-long        8 hours         48 hours            2               n/a
```
You can use `qstat -q name_of_queue` to check the queue status. It's highly probable that the nodes are all occupied and the queue is full of waiting jobs.
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
If your training only requires single machine, you can simply submit the training job. Refer [here](https://it.stonybrook.edu/help/kb/example-pbs-job-script) for how to draft PBS script.

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

