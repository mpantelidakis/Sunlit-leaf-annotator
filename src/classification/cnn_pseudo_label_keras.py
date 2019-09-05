#!/usr/bin/python
# -*- coding: utf-8 -*-
#
"""
    Pseudo label classifier with multiple models
    Models -> (Xception, VGG16, VGG19, ResNet50, InceptionV3, MobileNet)

    Name: cnn_keras.py
    Author: Gabriel Kirsten Menezes (gabriel.kirsten@hotmail.com)

"""
import time
import os
import shutil
import random
import numpy as np
import json
import logging
import sys

from PIL import Image
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Model, load_model
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend as K
from numpy import resize, expand_dims
from keras.preprocessing.image import load_img, img_to_array


from interface.interface import InterfaceException as IException
from classification.classifier import Classifier
from classification.pseudo_label import PseudoLabel

from collections import OrderedDict

from util.config import Config
from util.file_utils import File
from util.utils import TimeUtils


logger = logging.getLogger('PIL')
logger.setLevel(logging.WARNING)

# =========================================================
# Constants
# =========================================================
IMG_WIDTH, IMG_HEIGHT = 256, 256

START_TIME = time.time()


class CNNPseudoLabel(Classifier):
    """ Class for CNN classifiers based on Keras applications """

    def __init__(self,
                 architecture="VGG16",
                 learning_rate=0.001,
                 momentum=0.9,
                 batch_size=32,
                 epochs=50,
                 fine_tuning_rate=100,
                 transfer_learning=False,
                 save_weights=True,
                 recreate_dataset=False,
                 train_data_directory="",
                 validation_data_directory="",
                 test_data_directory="", 
                 no_label_data_directory=""):
        """
            Constructor of CNNKeras
        """

        self.architecture = Config(
            "Architecture", architecture, str)
        self.learning_rate = Config(
            "Learning rate", learning_rate, float)
        self.momentum = Config(
            "Momentum", momentum, float)
        self.batch_size = Config(
            "Batch size", batch_size, int)
        self.epochs = Config(
            "Epochs", epochs, int)
        self.fine_tuning_rate = Config(
            "Fine Tuning Rate", fine_tuning_rate, int)
        self.transfer_learning = Config(
            "Transfer Learning", transfer_learning, bool)
        self.save_weights = Config(
            "Save weights", save_weights, bool)
        self.recreate_dataset = Config(
            "Recreate Dataset", recreate_dataset, bool)
        self.train_data_directory = Config(
            "Train data directory", train_data_directory, str)
        self.validation_data_directory=Config(
            "Validation data directory", validation_data_directory, str)
        self.test_data_directory=Config(
            "Test data directory", test_data_directory, str)
        self.no_label_data_directory=Config(
            "No label data directory", no_label_data_directory, str)

        self.model=None
        self.pseudo_label=None
        self.trained=False

    def get_config(self):
        """Return configuration of classifier.

        Returns
        -------
        config : OrderedDict
            Current configs of classifier.
        """
        keras_config=OrderedDict()

        keras_config["Architecture"]=self.architecture
        keras_config["Learning rate"]=self.learning_rate
        keras_config["Momentum"]=self.momentum
        keras_config["Batch size"]=self.batch_size
        keras_config["Epochs"]=self.epochs
        keras_config["Fine Tuning rate"]=self.fine_tuning_rate
        keras_config["Transfer Learning"]=self.transfer_learning
        keras_config["Save weights"]=self.save_weights
        keras_config["Recreate Dataset"]=self.recreate_dataset
        keras_config["Train data directory"]=self.train_data_directory
        keras_config["Validation data directory"]=self.validation_data_directory
        keras_config["Test data directory"]=self.test_data_directory
        keras_config["No label data directory"]=self.no_label_data_directory
        return keras_config

    def set_config(self, configs):
        """Update configuration of classifier.

        Parameters
        ----------
        configs : OrderedDict
            New configs of classifier.
        """
        self.architecture=Config.nvl_config(
            configs["Architecture"], self.architecture)
        self.learning_rate=Config.nvl_config(
            configs["Learning rate"], self.learning_rate)
        self.momentum=Config.nvl_config(configs["Momentum"], self.momentum)
        self.batch_size=Config.nvl_config(
            configs["Batch size"], self.batch_size)
        self.epochs=Config.nvl_config(configs["Epochs"], self.epochs)
        self.fine_tuning_rate=Config.nvl_config(
            configs["Fine Tuning rate"], self.fine_tuning_rate)
        self.transfer_learning=Config.nvl_config(
            configs["Transfer Learning"], self.transfer_learning)
        self.save_weights=Config.nvl_config(
            configs["Save weights"], self.save_weights)
        self.recreate_dataset=Config.nvl_config(
            configs["Recreate Dataset"], self.recreate_dataset)
        self.train_data_directory = Config.nvl_config(
            configs["Train data directory"], self.train_data_directory)
        self.validation_data_directory=Config.nvl_config(
            configs["Validation data directory"], self.validation_data_directory)
        self.test_data_directory=Config.nvl_config(
            configs["Test data directory"], self.test_data_directory)
        self.no_label_data_directory=Config.nvl_config(
            configs["No label data directory"], self.no_label_data_directory)

    def get_summary_config(self):
        """Return fomatted summary of configuration.

        Returns
        -------
        summary : string
            Formatted string with summary of configuration.
        """
        keras_config=OrderedDict()

        keras_config[self.architecture.label]=self.architecture.value
        keras_config[self.learning_rate.label]=self.learning_rate.value
        keras_config[self.momentum.label]=self.momentum.value
        keras_config[self.batch_size.label]=self.batch_size.value
        keras_config[self.epochs.label]=self.epochs.value
        keras_config[self.fine_tuning_rate.label]=self.fine_tuning_rate.value
        keras_config[self.transfer_learning.label]=self.transfer_learning.value
        keras_config[self.save_weights.label]=self.save_weights.value
        keras_config[self.recreate_dataset.label]=self.recreate_dataset.value
        keras_config[self.train_data_directory.label]=self.train_data_directory.value
        keras_config[self.validation_data_directory.label]=self.validation_data_directory.value
        keras_config[self.test_data_directory.label]=self.test_data_directory.value
        keras_config[self.no_label_data_directory.label]=self.no_label_data_directory.value
        summary=''
        for config in keras_config:
            summary += "%s: %s\n" % (config, str(keras_config[config]))

        return summary

    def classify(self, dataset, test_dir, test_data, image):
        """"Perform the classification.

        Parameters
        ----------
        dataset : string
            Path to image dataset.
        test_dir : string
            Not used.
        test_data : string
            Name of test data file.

        Returns
        -------
        summary : list of string
            List of predicted classes for each instance in test data in ordered way.
        """

        predict_directory=File.make_path(dataset, test_dir)

        # Create a Keras class
        if not os.path.exists(File.make_path(predict_directory, "png")):
            os.makedirs(File.make_path(predict_directory, "png"))

        for file in os.listdir(predict_directory):
            print(File.make_path(predict_directory, file))
            if os.path.splitext(file)[-1] == ".tif":
                try:
                    img=Image.open(File.make_path(predict_directory, file))
                    new_file=os.path.splitext(file)[0] + ".png"
                    img.save(File.make_path(predict_directory,
                                            'png', new_file), "PNG", quality=100)
                except Exception as e:
                    print(e)
            else:
                print (File.make_path(predict_directory, file))
                os.symlink(File.make_path(predict_directory, file),
                           File.make_path(predict_directory, 'png', file))

        classify_datagen=ImageDataGenerator()

        classify_generator=classify_datagen.flow_from_directory(
            File.make_path(predict_directory, 'png'),
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=1,
            shuffle=False,
            class_mode=None)

        try:
            K.clear_session()
            if self.pseudo_label.weight_path is not None:
                self.create_model()
                self.model.load_weights(self.pseudo_label.weight_path)
        except Exception as e:
            raise IException("Can't load the model in " +
                             self.pseudo_label.weight_path + str(e))

        output_classification=self.model.predict_generator(
            classify_generator, classify_generator.samples, verbose=2)

        one_hot_output=np.argmax(output_classification, axis=1)

        one_hot_output=one_hot_output.tolist()

        return one_hot_output

    def train(self, dataset, training_data, force=False):
        """Perform the training of classifier.

        Parameters
        ----------
        dataset : string
            Path to image dataset.
        training_data : string
            Name of ARFF training file.
        force : boolean, optional, default = False
            If False don't perform new training if there is trained data.
        """

        self.create_model()

        self.pseudo_label.fit_with_pseudo_label(
            steps_per_epoch=self.pseudo_label.train_generator.samples // self.batch_size.value,
            validation_steps=self.pseudo_label.validation_generator.samples // self.batch_size.value)

    def create_model(self):
        self.pseudo_label=PseudoLabel(image_width=IMG_WIDTH,
                    image_height=IMG_HEIGHT,
                    image_channels=3,
                    train_data_directory=self.train_data_directory.value,
                    validation_data_directory=self.validation_data_directory.value,
                    test_data_directory=self.test_data_directory.value,
                    no_label_data_directory=self.no_label_data_directory.value,
                    epochs=self.epochs.value,
                    batch_size=self.batch_size.value,
                    pseudo_label_batch_size=self.batch_size.value*2,
                    transfer_learning={
                        'use_transfer_learning': self.transfer_learning.value,
                        'fine_tuning': self.fine_tuning_rate.value
                    },
                    architecture=self.architecture.value,
                    alpha=0.1)

        self.model=self.pseudo_label.model

    def must_train(self):
        """Return if classifier must be trained.

        Returns
        -------
        True
        """
        return not self.trained

    def must_extract_features(self):
        """Return if classifier must be extracted features.

        Returns
        -------
        False
        """
        return False

    def single_classify(self, image_path, directory, extractors, dict_classes):
        preprocess_input, decode_predictions=dict_preprocessing[self.app]
        pil_image=load_img(image_path)
        np_image=img_to_array(pil_image)
        res_image=resize(np_image, (IMG_HEIGHT, IMG_WIDTH, 3))
        tensor=expand_dims(res_image, axis=0)
        tensor=preprocess_input(tensor)
        predict=self.model.predict(tensor)
        predict=np.argmax(predict, axis=1)
        return dict_classes[predict[0]]
