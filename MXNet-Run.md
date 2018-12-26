# Running MXNet example of image classification
```
MX_DIR=$username/MXNetFolder/incubator-mxnet/example/image-classification
TOOL_DIR=$username/MXNetFolder/incubator-mxnet/tools
BW_DIR=$username/MXNetFolder/incubator-mxnet/tools/bandwidth/
```
## Training Imagenet on single node
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
By default, the log is directly displayed on the `STDOUT`. But of course we can
output it to a file 
```
$ tail log/train_imagenet.log 
INFO:root:Epoch[0] Batch [40]	Speed: 27.98 samples/sec	accuracy=0.000781
```
## Training Imagenet on multiple nodes
Please refer [here](https://mxnet.incubator.apache.org/faq/distributed_training.html) for technique details.

Training on multiple devices requires a launcher to establish parameter-server framework.
MXNet's launcher code is located at `MXNetFolder/incubator-mxnet/tools` and support many interfaces like ssh (default) and mpi.
Launcher has to know all hosts' names or IPs assigned in a hostfile, and will asynchronously create both the server and worker nodes which amounts are set in the execution.
If you use `ssh` and wanna cease the training job, you cannot terminate just the launcher but have the processes on each server and worker individually. 
Please ssh to each server and worker to manually kill the processes. 
Following is the example of creating a residual netwrok training for imagenet on 4 servers and 4 workers.

** Only after obtaining GPU nodes will you know the host names or IPs. 
That's why you need to submit STDIN to PBS queue when doing distributed training in HPC. **

```
$ python ${TOOL_DIR}/launch.py -n 4 -s 4 --hostfile ${HOME}/Hosts --sync-dst-dir ${HOME}/sync python ${MX_DIR}/train_imagenet.py \
--gpus 0,1,2,3,4,5,6,7 --num-epochs 1 --data-train ${IMGNET_DIR}/imagenet1k-train.rec --data-val ${IMGNET_DIR}/imagenet1k-val.rec \
--batch-size 64 --network resnet --num-layers 152 --kv-store dist_sync --data-nthreads 8
```
```
$ cat ${HOME}/Hosts 
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
In the log you can see a set of repeated batches. 
They correspond to the performance on each server you call and need not follow the order in the list file of hosts. 
Since the dataset is equally distributed to each computing server, the total batch number per epoch in this example is one quater of the number in single machine training.
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


## Authors
* **Yu-Jiun Kao** - *Initial work* - [Mail](yu-jiun.kao@stonybrook.edu)
Feel free to text me if you are interested in the work

