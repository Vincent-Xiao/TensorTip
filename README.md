
##TensorTip  
A TensorFlow scripts and templates for Training CNN and BNN  
This implementation is based on https://github.com/eladhoffer/convNet.tf and https://github.com/itayhubara/BinaryNet.tf.git

## Update
1. tensorflow dependency from 1.2.1 to 1.4.0 with related code update
2. adding logging function to implement local log recording
3. results is saved by time format directory rather than covering mode like orignial version
4. fix some bugs of orignial version to work well in tensorflow 1.4.0
5. fix incorrect cifar10 data preprocessing code
6. Using "sparse_softmax_cross_entropy_with_logits" loss function
7. supporting ImageNet Dataset based on https://github.com/tensorflow/models/tree/master/research/inception
    * ./ImageNetPreProcess dir contains download and imagenet data processing scripts;
    * ./ImageNetReading dir contatins scripts for reading imagenet dataset while training;
    * bash ./ImageNetPreProcess/download_and_preprocess_imagenet.sh (Maybe need to change some dir path params) to generate TFRecords before training  
    * python main.py --model alexnet --save alexnet --dataset imagenet  --batch_size xxx --device x --data_dir=$YourTFRecordsPath
8. supporting Residual Neural Network and Wide Residual Network(WRN) including ResNet and WRN of basic,bottleneck,pre-activation,dropout

## Dependencies
tensorflow version 1.4.0

## Training
* Train cifar10 model using gpu:
python main.py --model cifar10 --save cifar10 --dataset cifar10 --device x
* Train cifar10 model using cpu:
python main.py --model cifar10 --save cifar10 --dataset cifar10 --device x --False
* Train alexnet model using gpu:
python main.py --model alexnet --save alexnet --dataset imagenet  --batch_size xxx --device x --data_dir=$YourTFRecordsPath --decay_steps 10000  
*Resuming  
py main.py --model cifar10 --load $CheckPointDir(Eg:results/cifar10/2018-03-14-17-48-19) --resume True --dataset cifar10 --device x

## Results
Cifar10 : 90% top-1 accuracy(128 epochs)  
BNNCifar10 : 83.2% top-1 accuracy(128 epochs)  
WRN28-10:91.6% top-1 accuracy(128 epochs) for cifar10







