# SceneTextDetector
This code is modified from [EAST](https://github.com/argman/EAST). Thanks to their open source.:blush:
## Introduction
* The backbone of this method is a ResNet50-like network with SENet. It's worth mentioning that the size of output of backbone is 1/16(not 1/32 as classical ResNet50) of input, which makes full use of the information from original images.
* Multi-level feature fusion is deployed to detect multi-scale text regions.
* Label generation is the same as EAST.
* The whole network is trained from scratch. We don't use ResNet50 pretrained on ImageNet, because of the bias between classification task and detection/segementation task.
## Requirements
* TensorFlow > 1.0
* Python 3.6
* Shapely
* OpenCV
* ...
## Performance
[***ICDAR 2017 MLT***](http://rrc.cvc.uab.es/?ch=8&com=evaluation&task=1)

We use both training set(7200 images) and validation set(1800 images) to train our model from scratch. Two GPU is used and batch_size_per_gpu is set to 16. Learing rate of ADAM starts from 0.0002, decays to 94% every 5000 steps. The model is trained for about 230000 steps(about a week on two Tesla GPU).

|**Recall(%)**|**Precision(%)**|**F-Score(%)**|
|:-----------:|:-------------:|:------------:|
|60.47|75.68|67.22|

[***ICDAR 2015***](http://rrc.cvc.uab.es/?ch=2&com=evaluation&task=1)

We use the model trained on ICDAR 2017 MLT to test the performance on ICDAR 2015 directly.
(score_map_thresh=0.6, box_thresh=0.15)

|**Recall(%)**|**Precision(%)**|**F-Score(%)**|
|:-----------:|:-------------:|:------------:|
|77.80|84.17|80.86|

We use 1000 ICDAR 2015 training images to fine-tune our model trained on ICDAR 2017 MLT. The Result is coming soon.

|**Recall(%)**|**Precision(%)**|**F-Score(%)**|
|:-----------:|:-------------:|:------------:|
| | | |

## Training
```shell
vim multigpu_train.sh & change the configuration by yourself
sh multigpu_train.sh
```
### Data augmentation when training
* **random scale**: scale image size to 0.5/1/2/3 randomly.
* **random crop**: 512*512 random samples are cropped from scaled image.
* **randome rotation**: each cropped image have a 0.5 probability of a [-30, 30] degree rotation.
* **
## Testing
```shell
vim eval.sh & change the configuration by yourself
sh eval.sh
```
## Calculate the performance
If the groundtruth of test set is available, we can calculate the performance offline without logging into the public website. 
When you use this script first time, you need set up Polygon at first:
```shell
cd test_script/Polygon3-3.0.8
python setup.py install
```
Use the following command to calculate the performance:
```shell
vim script.py & change the parameters by yourself
python script.py -g= -s= -o=   //more information can be found in readme.txt in test_script
```
