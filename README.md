# SceneTextDetector
This code is modified from [EAST](https://github.com/argman/EAST). Thanks to their open source.
## Introduction
* The backbone of this method is a ResNet50-like network with SeNet. It's worth mentioning that the size of output of backbone is 1/16(not 1/32 as classical ResNet50) of input, which makes full use of the information from original images.
* Multi-level feature fusion is deployed to detect multi-scale text regions.
* Label generation is the same as EAST.
* The whole network is trained from scratch. We don't use ResNet50 pretrained on ImageNet, because of the difference between classification task and detection/segementation task.
## Requirements
* TensorFlow > 1.0
* Python 3.6
* Shapely
* OpenCV
* ...
