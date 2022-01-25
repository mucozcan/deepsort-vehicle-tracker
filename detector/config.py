import os

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))


feature_extractor_path = os.path.join(
    ROOT_DIR, "siamese_network/ckpts/model_128x128.pth")

train_data_dir = os.path.join(ROOT_DIR, "data/train")
test_data_dir = os.path.join(ROOT_DIR, "data/test")

train_input_size = (360, 640)
num_classes = 5
batch_size = 16
lr = 0.01
momentum = 0.9
weight_decay = 0.0005

prediction_input_size = (360, 640)
num_classes = 5

class_dict = {
    0: "__background__",
    1: "car",
    2: "truck",
    3: "van",
    4: "bus"
}
classes_to_track = ["car", "truck", "bus", "van"]
