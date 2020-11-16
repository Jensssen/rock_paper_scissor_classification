# -*- coding: utf-8 -*-
"""
 
File:
    Model.py
 
Authors: soe
Date:
    26.09.20

"""
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Input, Flatten, Dense, AveragePooling2D, Dropout


class Model_Class:
    """

    """

    def __init__(self, num_class, input_dim, verbose):
        self.num_class = num_class
        self.loss = None
        self.input_dim = input_dim
        self.output_shape = None
        self.model = None
        self.verbose = verbose
        # build model
        self.build_model()

    @staticmethod
    def conv_block(input, filters, kernel_size, name):
        conv1 = Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal',
                       padding="same",
                       activation='relu', name=f"conv_{name}_1")(input)
        conv1_bn = BatchNormalization()(conv1)
        conv2 = Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal',
                       padding="same",
                       activation='relu', name=f"conv_{name}_2")(conv1_bn)
        conv2_bn = BatchNormalization()(conv2)
        conv_out = MaxPool2D(pool_size=(2, 2), padding='valid')(conv2_bn)
        return conv_out

    @staticmethod
    def conv(input, filters, kernel_size, name):
        conv = Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal',
                      padding="same",
                      activation='relu', name=f"conv_{name}_1")(input)

        return conv

    def build_model(self):
        inputs = Input(shape=self.input_dim, name="input")
        pooling = AveragePooling2D((4, 4), 3, input_shape=(144, 144, 3))(inputs)
        conv1 = Model_Class.conv(pooling, 64, 3, name="1")
        conv2 = Model_Class.conv(conv1, 32, 3, name="2")
        maxpool = MaxPool2D(pool_size=(2, 2), padding='valid')(conv2)
        flatten = Flatten()(maxpool)
        dense1 = Dense(64, activation='relu')(flatten)
        drop = Dropout(0.5)(dense1)
        output = Dense(self.num_class, activation="softmax")(drop)

        # encoder path: first conv_block (64):
        # conv1 = Model_Class.conv_block(input=inputs, filters=32, kernel_size=3, name="1")
        # conv2 = Model_Class.conv_block(input=conv1, filters=64, kernel_size=3, name="2")
        # conv3 = Model_Class.conv_block(input=conv2, filters=128, kernel_size=3, name="3")

        # flatten = Flatten()(conv3)
        # dropout = Dropout(0.33)(flatten)
        # dense1 = Dense(64, activation='relu')(flatten)
        # dropout = Dropout(0.33)(dense1)
        # output = Dense(self.num_class, activation='softmax')(dropout)
        # self.output_shape = output.shape

        self.model = Model(inputs=inputs, outputs=output)
        if self.verbose:
            self.model.summary()
