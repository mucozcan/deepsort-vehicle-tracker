import os

SIAMESE_ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

train_dir = os.path.join(SIAMESE_ROOT_DIR, "data/")
model_save_path = os.path.join(SIAMESE_ROOT_DIR, "ckpts/model.pth")
batch_size = 32
input_size = (256, 128)

epochs = 100
lr = 0.001
