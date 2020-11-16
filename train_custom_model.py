# -*- coding: utf-8 -*-
"""

This script can be used to train a custom model on a locally saved dataset. 
However, this script is not used for the final model. The main reason for this is 
that due to the small amount of training data, I was not able to achieve good results on 
the validation data. Instead, the model overfitted quite drastically. 
This was the main reason why I switched to a fine tuning approach of mobile Net which can be found inside of the jupyther notebook.

File:
    train.py
 
Authors: soe
Date:
    15.09.20

"""
import os

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from model.Model import Model_Class

img_height = 144
img_width = 144
batch_size = 12

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


@tf.function
def train_one_batch(input_images, y_true, model, optimizer):
    """

    """
    with tf.GradientTape() as tape:
        softmax = model.model(input_images, training=True)
        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        loss = cce(y_true=y_true, y_pred=softmax)

    gradients = tape.gradient(loss, model.model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.model.trainable_variables))

    return softmax, loss


@tf.function
def eval_one_batch(input_images, y_true, model):
    """
    """
    softmax = model.model(input_images, training=False)
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    loss = cce(y_true=y_true, y_pred=softmax)
    return softmax, loss


train_generator = ImageDataGenerator(rescale=1. / 255,
                                     horizontal_flip=True,
                                     validation_split=0.1)

test_generator = ImageDataGenerator(rescale=1. / 255,
                                    validation_split=0.1)

ds_train = train_generator.flow_from_directory(directory='dataset/rock_paper_scissor_videobased/',
                                               batch_size=batch_size,
                                               shuffle=True,
                                               target_size=(img_height, img_width),
                                               seed=42,
                                               classes=["rock", "paper", "scissor"],
                                               subset="training")

ds_test = test_generator.flow_from_directory(directory='dataset/rock_paper_scissor_videobased/',
                                             batch_size=batch_size,
                                             shuffle=False,
                                             target_size=(img_height, img_width),
                                             seed=42,
                                             classes=["rock", "paper", "scissor"],
                                             subset="validation")

train_image = cv2.imread(ds_train.filepaths[0])


model = Model_Class(num_class=3,
                    input_dim=(img_height, img_width, 3),
                    verbose=True)
optimizer = tf.keras.optimizers.Adam(lr=0.0001)

for epoch in range(44):
    epoch_train_loss = 0
    epoch_train_accuracy = 0
    epoch_validation_loss = 0
    epoch_validation_accuracy = 0
    train_counter = 0
    validation_counter = 0

    # Train one Epoch
    for idx in range(ds_train.__len__()):
        images, labels = ds_train.__getitem__(idx)
        softmax, loss = train_one_batch(input_images=images, y_true=labels, model=model, optimizer=optimizer)
        prediction = np.argmax(softmax.numpy(), axis=-1)
        label_list = np.argmax(labels, axis=-1)
        accuracy = (prediction == label_list).mean()
        epoch_train_loss += loss.numpy()
        epoch_train_accuracy += accuracy
        train_counter += 1

    # Test one Epoch
    for idx in range(ds_test.__len__()):
        images, labels = ds_test.__getitem__(idx)
        labels = np.asarray(labels, dtype=np.uint8)
        softmax, loss = eval_one_batch(input_images=images, y_true=labels, model=model)
        prediction = np.argmax(softmax.numpy(), axis=-1)
        label_list = np.argmax(labels, axis=-1)
        accuracy = (prediction == label_list).mean()
        epoch_validation_loss += loss.numpy()
        epoch_validation_accuracy += accuracy
        validation_counter += 1

    print(
        f"Epoch = {epoch} - train_loss = {epoch_train_loss / train_counter} - train_accuracy = "
        f"{epoch_train_accuracy / train_counter} - validation_loss = {epoch_validation_loss / validation_counter} - "
        f"validation_accuracy = {epoch_validation_accuracy / validation_counter}")

saved_model_dir = os.path.join("log", 'saved_model_dir/')
tf.saved_model.save(model.model, os.path.join(saved_model_dir, 'model'))

