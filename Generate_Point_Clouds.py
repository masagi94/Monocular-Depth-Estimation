import os
import keras
import tensorflow as tf
from tensorflow.keras import layers
tf.random.set_seed(69)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import time
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.ion()

import skimage.io
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import math
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--dataset_path', type=str, required=True)
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--image_size', type=int, required=False, default=256)

args = parser.parse_args()

DATASET_PATH = args.dataset_path
MODEL_PATH = args.model_path
HEIGHT,WIDTH = (args.image_size,args.image_size)

print("\n\n******************************************************")
print("CHOSEN OPTIONS\n")
print('Dataset Location:', DATASET_PATH)
print('Model Location:', MODEL_PATH)
print("******************************************************\n\n\n")

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size=6, dim=(768, 1024), n_channels=3, shuffle=True):
        """
        Initialization
        """
        self.data = data
        self.indices = self.data.index.tolist()
        self.dim = dim
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.min_depth = 0.1
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        if (index + 1) * self.batch_size > len(self.indices):
            self.batch_size = len(self.indices) - index * self.batch_size
        # Generate one batch of data
        # Generate indices of the batch
        index = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        # Find list of IDs
        batch = [self.indices[k] for k in index]
        x, y = self.data_generation(batch)

        return x, y

    def on_epoch_end(self):

        """
        Updates indexes after each epoch
        """
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def load(self, image_path, depth_map, mask):
        """Load input and target image."""

        image_ = cv2.imread(image_path)
        image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
        image_ = cv2.resize(image_, self.dim)
        image_ = tf.image.convert_image_dtype(image_, tf.float32)

        depth_map = np.load(depth_map).squeeze()
        depth_map = cv2.resize(depth_map, self.dim)
        depth_map = np.expand_dims(depth_map, axis=2)
        depth_map = tf.image.convert_image_dtype(depth_map, tf.float32)

        return image_, depth_map

    def data_generation(self, batch):

        x = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, 1))

        for i, batch_id in enumerate(batch):
            x[i,], y[i,] = self.load(
                self.data["image"][batch_id],
                self.data["depth"][batch_id],
                self.data["mask"][batch_id],
            )

        return x, y

def get_data(path, sample=1):
    filelist = []

    for root, dirs, files in os.walk(path):
        for file in files:
            filelist.append(os.path.join(root, file))

    filelist.sort()

    data = {
        "image": [x for x in filelist if x.endswith(".png")],
        "depth": [x for x in filelist if x.endswith(".npy")],
        "mask": [x for x in filelist if x.endswith(".npy")],
    }
    df = pd.DataFrame(data)

    return df.sample(frac=sample, random_state=42)

if(not os.path.exists(MODEL_PATH)):
	print("ERROR: '" + MODEL_PATH + "' DOES NOT EXIST")
	exit()

imported_model = tf.keras.models.load_model(MODEL_PATH)

df = get_data(DATASET_PATH, 1)

if not os.path.exists("Point_Clouds"):
    os.mkdir("Point_Clouds")

## Visualizing model output
max = len(df)
i = 0
while i < max:
    samples = next(iter(DataGenerator(data=df[i:], batch_size=6, dim=(HEIGHT, WIDTH))))

    print('SAMPLES',len(samples))

    ground_truth_depth_map = (samples[1][1].squeeze())  # target
    img = (samples[0][1].squeeze())
    input, target = samples
    
    pred_depth_map = imported_model.predict(input)
    pred_depth_map = pred_depth_map[0].squeeze()


    # # set up a figure twice as wide as it is tall
    fig = plt.figure(figsize=(10,30))
    ax = fig.add_subplot(3, 1, 1)

    ax.text(10, 10, 'Original Image', fontsize=8, bbox={'facecolor': 'white', 'pad': 3})
    ax.imshow(img)
    ax = fig.add_subplot(3, 2, 3)
    ax.text(10, 10, 'Ground Truth Depth Map', fontsize=8, bbox={'facecolor': 'white', 'pad': 3})
    ax.imshow(ground_truth_depth_map)
    ax = fig.add_subplot(3, 2, 4)
    ax.text(10, 10, 'Predicted Depth Map', fontsize=8, bbox={'facecolor': 'white', 'pad': 3})
    ax.imshow(pred_depth_map)
    
    pred_depth_map = np.flipud(pred_depth_map)
    img = np.flipud(img)

    ax = fig.add_subplot(3, 2, 6, projection='3d')
    STEP = 7
    for x in range(0, img.shape[0], STEP):
        for y in range(0, img.shape[1], STEP):
            ax.scatter(
                pred_depth_map[x,y], y, x,
                c=[tuple(img[x, y, :3])], s=3)      
    
    ax.view_init(15, 0)
    plt.show()


    ground_truth_depth_map = np.flipud(ground_truth_depth_map)

    ax = fig.add_subplot(3, 2, 5, projection='3d')
    STEP = 7
    for x in range(0, img.shape[0], STEP):
        for y in range(0, img.shape[1], STEP):
            ax.scatter(
                ground_truth_depth_map[x,y], y, x,
                c=[tuple(img[x, y, :3])], s=3)      
    
    ax.view_init(15, 0)
    plt.show()
    plt.savefig("Point_Clouds/"+str(i)+".png")
    plt.close()    

    i += 1