import os
import sys
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

parser.add_argument('--train_dataset', type=str, required=True)
parser.add_argument('--test_dataset', type=str, required=True)
parser.add_argument('--learning_rate', type=int, required=False, default=.01)
parser.add_argument('--image_size', type=int, required=False, default=256)
parser.add_argument('--epochs', type=int, required=False, default=50)
parser.add_argument('--batch_size', type=int, required=False, default=16)

args = parser.parse_args()

TRAIN_PATH = args.train_dataset
TEST_PATH = args.test_dataset
LEARNING_RATE = args.learning_rate
HEIGHT,WIDTH = (args.image_size,args.image_size)
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size


print("\n\n******************************************************")
print("CHOSEN OPTIONS\n")
print('Train Dataset Location:', TRAIN_PATH)
print('Validation Dataset Location:', TEST_PATH)
print('Learning Rate:', LEARNING_RATE)
print('Train image size:', (HEIGHT,WIDTH))
print('EPOCHS:', EPOCHS)
print('Batch Size:', BATCH_SIZE)
print("******************************************************\n\n\n")

# ## Building a data pipeline
# 
# 1. The pipeline takes a dataframe containing the path for the RGB images,
# as well as the depth and depth mask files.
# 2. It reads and resize the RGB images.
# 3. It reads the depth and depth mask files, process them to generate the depth map image and
# resize it.
# 4. It returns the RGB images and the depth map images for a batch.

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

    # We had to comment out the mask areas of this code as our
    # custom datset did not have depth masks. 
    def load(self, image_path, depth_map, mask):
        """Load input and target image."""

        image_ = cv2.imread(image_path)
        image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
        image_ = cv2.resize(image_, self.dim)
        image_ = tf.image.convert_image_dtype(image_, tf.float32)

        depth_map = np.load(depth_map).squeeze()

        # mask = np.load(mask)
        # mask = mask > 0

        # max_depth = min(300, np.percentile(depth_map, 99))
        # depth_map = np.clip(depth_map, self.min_depth, max_depth)
        # depth_map = np.log(depth_map, where=mask)

        # depth_map = np.ma.masked_where(~mask, depth_map)

        # depth_map = np.clip(depth_map, 0.1, np.log(max_depth))
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


# Visualize the loss over time from training the model
def visualize_loss_histogram(loss_history, LR,batch,EPOCHS,):
    name = "EPOCHS-"+str(EPOCHS)+" LR-" + str(LR) + " BATCH-" + str(batch) 
    loss = loss_history['loss']
    val_loss = loss_history['val_loss']
    
    x_tick_increments = 5* math.ceil((EPOCHS/4)/5)

    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Model Loss - LR: '+str(LR)+' Batch Size: ' +str(batch))
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.yticks(np.arange(0.0, max(max(loss),max(val_loss)), .1))
    plt.xticks(np.arange(0, len(loss)+1, x_tick_increments))
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("Results Figures/Model Loss - "+name+".png")
    plt.close()
    


# Visualize the generated depth map of the model on a sample image from the test set
def visualize_depth_map(samples,LR,batch,EPOCHS, model=None, ):
    name = "EPOCHS-"+str(EPOCHS)+" LR-" + str(LR) + " BATCH-" + str(batch) 
    input, target = samples

    cmap = plt.cm.jet
    cmap.set_bad(color="black")

    pred = model.predict(input)
    fig, ax = plt.subplots(6, 3, figsize=(50, 50))
    for i in range(6):
        ax[i, 0].imshow((input[i].squeeze()))
        ax[i, 1].imshow((target[i].squeeze()), cmap=cmap)
        ax[i, 2].imshow((pred[i].squeeze()), cmap=cmap)
        plt.pause(0.001)
    plt.tight_layout()
    plt.savefig("Results Figures/Predictions - "+name+".png")
    plt.close()

    
    

# The DownscaleBlock, UpscaleBlock, and BottleneckBlock are used for building the model
# We experimented with a couple different layers and even some dropout layers but found them
# to decrease model performance.
class DownscaleBlock(layers.Layer):
    def __init__(
        self, filters, kernel_size=(3, 3), padding="same", strides=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.convA = layers.Conv2D(filters, kernel_size, strides, padding)
        self.convB = layers.Conv2D(filters, kernel_size, strides, padding)
        self.reluA = layers.LeakyReLU(alpha=0.2)
        self.reluB = layers.LeakyReLU(alpha=0.2)
        self.bn2a = tf.keras.layers.BatchNormalization()
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.pool = layers.MaxPool2D((2, 2), (2, 2))
        self.dropout = tf.keras.layers.Dropout(.5)
    def call(self, input_tensor):
        d = self.convA(input_tensor)
        # d = self.dropout(d)
        x = self.bn2a(d)
        x = self.reluA(x)
        # x = self.dropout(x)

        x = self.convB(x)
        # x = self.dropout(x)
        x = self.bn2b(x)
        x = self.reluB(x)

        x += d
        p = self.pool(x)
        return x, p


class UpscaleBlock(layers.Layer):
    def __init__(
        self, filters, kernel_size=(3, 3), padding="same", strides=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.us = layers.UpSampling2D((2, 2))
        self.convA = layers.Conv2D(filters, kernel_size, strides, padding)
        self.convB = layers.Conv2D(filters, kernel_size, strides, padding)
        self.reluA = layers.LeakyReLU(alpha=0.2)
        self.reluB = layers.LeakyReLU(alpha=0.2)
        self.bn2a = tf.keras.layers.BatchNormalization()
        self.bn2b = tf.keras.layers.BatchNormalization()
        self.conc = layers.Concatenate()
        self.dropout = tf.keras.layers.Dropout(.5)

    def call(self, x, skip):
        x = self.us(x)
        
        concat = self.conc([x, skip])
        x = self.convA(concat)
        # x = self.dropout(x)
        x = self.bn2a(x)
        x = self.reluA(x)
        # x = self.dropout(x)

        x = self.convB(x)
        # x = self.dropout(x)
        x = self.bn2b(x)
        x = self.reluB(x)

        return x


class BottleNeckBlock(layers.Layer):
    def __init__(
        self, filters, kernel_size=(3, 3), padding="same", strides=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.convA = layers.Conv2D(filters, kernel_size, strides, padding)
        self.convB = layers.Conv2D(filters, kernel_size, strides, padding)
        self.reluA = layers.LeakyReLU(alpha=0.2)
        self.reluB = layers.LeakyReLU(alpha=0.2)
        self.dropout = tf.keras.layers.Dropout(.5)
    def call(self, x):
        x = self.convA(x)
        # x = self.dropout(x)
        x = self.reluA(x)
        x = self.convB(x)
        # x = self.dropout(x)
        x = self.reluB(x)
        return x


# We use SSIM, L1, Edge, Huber, and MSE for the loss function.
# The aggregate of these losses along with their weights is what is considered when
# training the model.
class DepthEstimationModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.ssim_loss_weight = 0.85
        self.l1_loss_weight = 0.1
        self.edge_loss_weight = 0.9
        self.huber_loss_weight = .3
        self.mse_loss_weight = .3
        self.loss_metric = tf.keras.metrics.Mean(name="loss")
        f = [16, 32, 64, 128, 256, 512]
        self.downscale_blocks = [
            DownscaleBlock(f[0]),
            DownscaleBlock(f[1]),
            DownscaleBlock(f[2]),
            DownscaleBlock(f[3]),
            DownscaleBlock(f[4]),
        ]
        self.bottle_neck_block = BottleNeckBlock(f[5])
        self.upscale_blocks = [
            UpscaleBlock(f[4]),
            UpscaleBlock(f[3]),
            UpscaleBlock(f[2]),
            UpscaleBlock(f[1]),
            UpscaleBlock(f[0]),
        ]
        self.conv_layer = layers.Conv2D(1, (1, 1), padding="same", activation="tanh")

    def calculate_loss(self, target, pred):
        # Edges
        dy_true, dx_true = tf.image.image_gradients(target)
        dy_pred, dx_pred = tf.image.image_gradients(pred)
        weights_x = tf.exp(tf.reduce_mean(tf.abs(dx_true)))
        weights_y = tf.exp(tf.reduce_mean(tf.abs(dy_true)))

        # Depth smoothness
        smoothness_x = dx_pred * weights_x
        smoothness_y = dy_pred * weights_y

        depth_smoothness_loss = tf.reduce_mean(abs(smoothness_x)) + tf.reduce_mean(abs(smoothness_y))

        # Structural similarity (SSIM) index
        ssim_loss = tf.reduce_mean(1 - tf.image.ssim(target, pred, max_val=WIDTH, filter_size=7, k1=0.01 ** 2, k2=0.03 ** 2))
        # Point-wise depth
        l1_loss = tf.reduce_mean(tf.abs(target - pred))

        mse_loss = tf.reduce_mean(tf.square(target - pred))

        huber_loss = tf.keras.losses.Huber(delta=1.0)(target, pred)

        loss = (
            (self.ssim_loss_weight * ssim_loss)
            + (self.l1_loss_weight * l1_loss)
            + (self.edge_loss_weight * depth_smoothness_loss)
            + (self.huber_loss_weight * huber_loss)
            + (self.mse_loss_weight * mse_loss)
        )

        return loss

    @property
    def metrics(self):
        return [self.loss_metric]

    def train_step(self, batch_data):
        input, target = batch_data
        with tf.GradientTape() as tape:
            pred = self(input, training=True)
            loss = self.calculate_loss(target, pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_metric.update_state(loss)
        return {
            "loss": self.loss_metric.result(),
        }

    def test_step(self, batch_data):
        input, target = batch_data

        pred = self(input, training=False)
        loss = self.calculate_loss(target, pred)

        self.loss_metric.update_state(loss)
        return {
            "loss": self.loss_metric.result(),
        }

    def call(self, x):
        c1, p1 = self.downscale_blocks[0](x)
        c2, p2 = self.downscale_blocks[1](p1)
        c3, p3 = self.downscale_blocks[2](p2)
        c4, p4 = self.downscale_blocks[3](p3)

        bn = self.bottle_neck_block(p4)

        u1 = self.upscale_blocks[0](bn, c4)
        u2 = self.upscale_blocks[1](u1, c3)
        u3 = self.upscale_blocks[2](u2, c2)
        u4 = self.upscale_blocks[3](u3, c1)

        return self.conv_layer(u4)


## Model training
# This is where we actually train the model on the train set, and test it on the validation set.
def evaluate_model(EPOCHS,LR,batch,cross_entropy):
 
    model = DepthEstimationModel()
    # Define the loss function
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR, amsgrad=False,)

    # Compile the model
    model.compile(optimizer, loss= cross_entropy)

    train_loader = DataGenerator(data= train_df.reset_index(drop="true"), batch_size= batch, dim= (HEIGHT, WIDTH))
    validation_loader = DataGenerator(data= test_df.reset_index(drop="true"), batch_size= batch, dim= (HEIGHT, WIDTH))

    history_callback = model.fit(train_loader, epochs= EPOCHS, validation_data= validation_loader,)

    ## SAVE MODEL

    model.save("trained_model")
    print("MODEL SAVED AS: trained_model")

    visualize_loss_histogram(history_callback.history, LR,batch,EPOCHS)

    ## Visualizing model output
    test_loader = next(iter(DataGenerator(data=test_df.reset_index(drop="true"), batch_size=6, dim=(HEIGHT, WIDTH))))
    visualize_depth_map(test_loader, LR, batch, EPOCHS, model=model)
    return()

##  Helper method to get our custom dataset
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



########################################################################
####                                                                ####
####                    MAIN CODE EXECUTION                         ####
####                                                                ####
########################################################################

train_df = get_data(TRAIN_PATH, 1)
test_df = get_data(TEST_PATH, 1)

if __name__ == '__main__':

    print("Train size:",len(train_df.index))
    print("Test size:",len(test_df.index))
    print("\n\n")

    ## TRAIN NEW MODEL AND SAVE AS "trained_model"
    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
    evaluate_model(EPOCHS,LEARNING_RATE,BATCH_SIZE,cross_entropy)
