# Car-Damage-Detection

## Problem Statement
Given a pic of damaged car, find which part is damaged. The parts can be either of rear_bumper, front_bumper, headlamp, door, hood.

## Solution Approach
* At first, it looked like a classification task but it turned out to be more complex. I did some initial analysis of the dataset to understand the problem statement and given (lot of) annotation files (code be found [here](https://github.com/Lplenka/Car-Damage-Detection/blob/master/initial_analysis/dataset_analysis.ipynb).)

* Based on the analysis, I decided to build two image segmentation models. One model to segment the damages which returns the "damage" polygon(s). One model to segment the parts of the car which returns the "parts" polygon(s). Then I can check damage polygons lie inside which "part" polygon and can detect the damaged part. There should be some way to train a single model that does both the tasks, but it can be the second version. Much later, while writing the final inference code I decided to see how far the damage is from different parts and return the part nearest to a damage.

* I decided to use [MaskRCNN](https://github.com/matterport/Mask_RCNN) for segmentation tasks. The given data format matched the required VGG format. I completed the installation and implemented of custom [Dataset()](https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/utils.py#L239) and [Config()](https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/config.py#L17) classes for this problem.

* Unfortunately, when I started the training in Google Colab, it got stuck at epoch one. Later I found that many have faced similar issues [#1781](https://github.com/matterport/Mask_RCNN/issues/1781), [#1877](https://github.com/matterport/Mask_RCNN/issues/1877), [#2168](https://github.com/matterport/Mask_RCNN/issues/2168) with the library as it is almost obsolete. I tried to solve it for hours both locally and on Google Colab but could not make it work.

* After a lot of research I decided to use Facebook's [detectron2](https://github.com/facebookresearch/Detectron). I wrote code to convert given VGG annotations to COCO annotations (can be found [here](https://github.com/Lplenka/Car-Damage-Detection/tree/master/via2coco)) since Detectron2 does not work with VGG dataset.

* Then I tested how Detectron2 works ([here](https://github.com/Lplenka/Car-Damage-Detection/blob/master/initial_analysis/detectron_test.ipynb) and [here](https://github.com/Lplenka/Car-Damage-Detection/blob/master/initial_analysis/detectron_custom_data.ipynb)) before moving to Google Colab for training.

* After that, I trained two image segmentation models, one for damage segmentation (can be found [here](https://github.com/Lplenka/Car-Damage-Detection/blob/master/detectron_damage_train.ipynb)) and one for parts segmentation (can be found [here](https://github.com/Lplenka/Car-Damage-Detection/blob/master/detectron_multiclass_parts_train.ipynb).)

* Due to time constraint I could not spend much time in fine tuning of the models. Though with minimal tweaks the Average Precision scores were not that good but the models' performances in segmetation tasks were decent.

* In the final inference notebook (can be found [here](https://github.com/Lplenka/Car-Damage-Detection/blob/master/detectron_inference.ipynb)) I load the models and perform inference of sample images. The results were decent. The performance can be improved using data augmentation and transfer learning. I will be working on those next.


## Folder Structure
```bash
.
├── Initial Analysis
│   ├── pycocotools
│   ├── data_visualize_coco.ipynb
│   ├── dataset_analysis.ipynb
│   ├── detectron_custom_data.ipynb
│   ├── detectron_test.ipynb
├── COCO_dataset
├── dataset
├── via2coco
│       ├── convert.py
│       ├── getArea.py
│       ├── merge.py
├── detectron_damage_train.ipynb
├── detectron_inference.ipynb
├── detectron_multiclass_parts_train.ipynb
├── requirements.txt
├──.gitignore  
└── Readme.md
```
## Installation
* Install python dependencies using requirements text file.
```
pip install -r requirements.txt
```
* [Pytorch](https://github.com/pytorch/pytorch) and [Detectron2](https://github.com/facebookresearch/Detectron), you can install those separately using the following commands or by refering [here](https://pytorch.org/get-started/locally/#mac-installation) and [here](https://detectron2.readthedocs.io/tutorials/install.html), I used Google Colab for training, so following are the steps to install Pytorch and Detectron:

```
torch==1.5.1+cu101 
torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install detectron2==0.2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html
```
* Pycocotools to Visualize Coco Dataset can be installed using this instructions provided [here.](https://github.com/cocodataset/cocoapi)

## Link to datasets and Models
* [COCO_dataset](https://drive.google.com/drive/folders/1mxdGl5Ah8NTJQIAAcWsDarJxmZFS0B84?usp=sharing)
* [dataset (Original VGG dataset)](https://drive.google.com/drive/folders/1lvapqYrXS7oCt5Mzp5UO4ZVBqZyvrjz8?usp=sharing)
* [Damage Segmentation Model Weights](https://drive.google.com/file/d/1-Zc5l3jyPVIDSl9dy1jubMQRSXVLMGGV/view?usp=sharing)
* [Parts Segmentation Model Weights](https://drive.google.com/file/d/1-c8ClXB9YHwkMFY6hwuX1TGqy_Q3yE7e/view?usp=sharing)

## Conclusion
* Current parts segmentation Model confuses between front and rear bumper, probably because of the less data.
* Center of polygon was mostly different from center of bounding box. 
* Currently, I am using the center of bounding boxes in `detect_damage_part` function in inference notebook. I will try to use center of polygons instead, which can give better results.

## Next Steps
* Need data augmentation
* Need transfer learning
* Modular code
* Write docstrings for all functions
