## DeepSORT Vehicle Tracker 
#### !!! This project is created for learning purposes do not use in production directly !!!
### Introduction
This is a 2D Vehicle Tracker based on works [Simple Online and Realtime Tracking with a Deep Association Metric (Deep SORT)](https://arxiv.org/abs/1703.07402) and [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497). 

For DeepSORT, [official implementation of authors](https://github.com/nwojke/deep_sort) is used. The official repository contains deprecated modules, so [I fixed them](https://github.com/mucozcan/deep_sort/tree/fix/sklearn). The implementation is using a feature extractor(CNN) trained on pedestrian images. I trained a new Siamese Network using the vehicle dataset shared by user [@John1liu](https://github.com/John1liu) and used it as feature extractor for association.
[Dataset Link](https://drive.google.com/file/d/1lushuv4QMTmfFwURU1Ug6mezkzQAMi0S/view)

For Faster-RCNN, I used the official torch implementation with a backbone [MobileNet V3](https://arxiv.org/abs/1905.02244) trained on COCO Dataset. Tracker only tracks object with classes "car", "truck" and "bus".
### Prerequisites
* Python 3.8+
* Poetry 1.1.11([offical installation guide](https://python-poetry.org/docs/#installation))

### Installation
Clone the vehicle-tracker repo:
``` 
git clone https://github.com/mucozcan/deepsort-vehicle-tracker.git
cd deepsort-vehicle-tracker/
```
Clone the official DeepSORT implementation with the dependency fix made by me:
```
git clone -b fix/sklearn https://github.com/mucozcan/deep_sort.git
```
Install dependencies:
```
poetry install
```
Run the demo:
```
poetry run python3 tracker.py --source test.mp4
```
or
```
poetry run python3 tracker.py --source [RTSP Stream Link]
```

### TODO
 * detector trained on custom vehicle dataset
 * onnx and TVM support
 * tuning on DeepSORT parameters
