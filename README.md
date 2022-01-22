## Vehicle Tracker 
#### !!! This project is created for learning purposes do not use in production directly !!!
### Introduction

dataset: https://drive.google.com/file/d/1lushuv4QMTmfFwURU1Ug6mezkzQAMi0S/view @John1liu

### Prerequisites
* Python 3.8+
* Poetry 1.1.11([offical installation guide](https://python-poetry.org/docs/#installation))

### Installation
Clone the vehicle-tracker repo:
``` 
git clone https://github.com/mucozcan/vehicle-tracker.git
cd vehicle-tracker/
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
poetry run python3 tracker.py
```

### TODO

 * train detector on custom vehicle dataset
 * argument parser
 * onnx and TVM support
 * tuning on DeepSORT parameters
