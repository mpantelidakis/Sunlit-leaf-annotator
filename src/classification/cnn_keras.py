#!/usr/bin/python
# -*- coding: utf-8 -*-
#
"""
    Generic classifier with multiple models
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

from collections import OrderedDict

from util.config import Config
from util.file_utils import File
from util.utils import TimeUtils
        


logger = logging.getLogger('PIL')
logger.setLevel(logging.WARNING)

START_TIME = time.time()

# =========================================================
# Constants
# =========================================================

IMG_WIDTH, IMG_HEIGHT = 256, 256
weight_path = None

dict_preprocessing = {}
dict_preprocessing[0] = applications.xception.preprocess_input, applications.xception.decode_predictions
dict_preprocessing[1] = applications.vgg16.preprocess_input, applications.vgg16.decode_predictions
dict_preprocessing[2] = applications.vgg19.preprocess_input, applications.vgg19.decode_predictions
dict_preprocessing[3] = applications.resnet50.preprocess_input, applications.resnet50.decode_predictions
dict_preprocessing[4] = applications.inception_v3.preprocess_input, applications.inception_v3.decode_predictions
dict_preprocessing[5] = applications.mobilenet.preprocess_input, applications.mobilenet.decode_predictions

class CNNKeras(Classifier):
    """ Class for CNN classifiers based on Keras applications """

    def __init__(self, architecture="ResNet50", learning_rate=0.001, momentum=0.9, batch_size=32, epochs=50, fine_tuning_rate=100, transfer_learning=False, save_weights=True, perc_train=80, perc_validation=20, recreate_dataset=False):
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
        self.perc_train = Config(
            "Perc Train", perc_train, float)
        self.perc_validation = Config(
            "Perc Validation", perc_validation, float)
        self.recreate_dataset = Config(
            "Recreate Dataset", recreate_dataset, bool)
        self.file_name = "kerasCNN"

        self.model = None

        self.trained = False

    def get_config(self):
        """Return configuration of classifier.

        Returns
        -------
        config : OrderedDict
            Current configs of classifier.
        """
        keras_config = OrderedDict()

        keras_config["Architecture"] = self.architecture
        keras_config["Learning rate"] = self.learning_rate
        keras_config["Momentum"] = self.momentum
        keras_config["Batch size"] = self.batch_size
        keras_config["Epochs"] = self.epochs
        keras_config["Fine Tuning rate"] = self.fine_tuning_rate
        keras_config["Transfer Learning"] = self.transfer_learning
        keras_config["Save weights"] = self.save_weights
        keras_config["Perc Train"] = self.perc_train
        keras_config["Perc Validation"] = self.perc_validation
        keras_config["Recreate Dataset"] = self.recreate_dataset
        return keras_config

    def set_config(self, configs):
        """Update configuration of classifier.

        Parameters
        ----------
        configs : OrderedDict
            New configs of classifier.
        """
        self.architecture = Config.nvl_config(
            configs["Architecture"], self.architecture)
        self.learning_rate = Config.nvl_config(
            configs["Learning rate"], self.learning_rate)
        self.momentum = Config.nvl_config(configs["Momentum"], self.momentum)
        self.batch_size = Config.nvl_config(
            configs["Batch size"], self.batch_size)
        self.epochs = Config.nvl_config(configs["Epochs"], self.epochs)
        self.fine_tuning_rate = Config.nvl_config(
            configs["Fine Tuning rate"], self.fine_tuning_rate)
        self.transfer_learning = Config.nvl_config(
            configs["Transfer Learning"], self.transfer_learning)
        self.save_weights = Config.nvl_config(
            configs["Save weights"], self.save_weights)
        self.perc_train = Config.nvl_config(
            configs["Perc Train"], self.perc_train)
        self.perc_validation = Config.nvl_config(
            configs["Perc Validation"], self.perc_validation)
        self.recreate_dataset = Config.nvl_config(
            configs["Recreate Dataset"], self.recreate_dataset)

    def get_summary_config(self):
        """Return fomatted summary of configuration.

        Returns
        -------
        summary : string
            Formatted string with summary of configuration.
        """
        keras_config = OrderedDict()

        keras_config[self.architecture.label] = self.architecture.value
        keras_config[self.learning_rate.label] = self.learning_rate.value
        keras_config[self.momentum.label] = self.momentum.value
        keras_config[self.batch_size.label] = self.batch_size.value
        keras_config[self.epochs.label] = self.epochs.value
        keras_config[self.fine_tuning_rate.label] = self.fine_tuning_rate.value
        keras_config[self.transfer_learning.label] = self.transfer_learning.value
        keras_config[self.save_weights.label] = self.save_weights.value
        keras_config[self.perc_train.label] = self.perc_train.value
        keras_config[self.perc_validation.label] = self.perc_validation.value
        keras_config[self.recreate_dataset.label] = self.recreate_dataset.value
        summary = ''
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

        predict_directory = File.make_path(dataset, test_dir)

        # Create a Keras class
        if not os.path.exists(File.make_path(predict_directory, "png")):
            os.makedirs(File.make_path(predict_directory, "png"))

        for file in os.listdir(predict_directory):
            print(File.make_path(predict_directory, file))
            if os.path.splitext(file)[-1] == ".tif":
                try:
                    img = Image.open(File.make_path(predict_directory, file))
                    # img.thumbnail(img.size)
                    new_file = os.path.splitext(file)[0] + ".png"
                    img.save(File.make_path(predict_directory,
                                            'png', new_file), "PNG", quality=100)
                except Exception as e:
                    print(e)
            else:
                print(File.make_path(predict_directory, file))
                os.symlink(File.make_path(predict_directory, file),
                           File.make_path(predict_directory, 'png', file))

        classify_datagen = ImageDataGenerator()

        classify_generator = classify_datagen.flow_from_directory(
            File.make_path(predict_directory, 'png'),
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=1,
            shuffle=False,
            class_mode=None)

        try:
            # self.model.load_weights(
            #"../models_checkpoints/" + self.file_name + ".h5")
            K.clear_session()
            if self.weight_path is not None:
                self.model = load_model(self.weight_path)
                path_classes = self.weight_path.replace(
                    "_model.h5", "_classes.npy")
                print("Load Model H5:"+self.weight_path)
                CLASS_NAMES = np.load(path_classes).item().keys()
        except Exception as e:
            raise IException("Can't load the model in " +
                             self.weight_path + str(e))

        output_classification = self.model.predict_generator(
            classify_generator, classify_generator.samples, verbose=2)

        one_hot_output = np.argmax(output_classification, axis=1)

        one_hot_output = one_hot_output.tolist()

        for index in range(0, len(one_hot_output)):
            one_hot_output[index] = CLASS_NAMES[one_hot_output[index]]

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

        # select .h5 filename
        if self.fine_tuning_rate.value == 100:
            self.file_name = str(self.architecture.value) + \
                '_learning_rate' + str(self.learning_rate.value) + \
                '_transfer_learning'
        elif self.fine_tuning_rate.value == -1:
            self.file_name = str(self.architecture.value) + \
                '_learning_rate' + str(self.learning_rate.value) + \
                '_without_transfer_learning'
        else:
            self.file_name = str(self.architecture.value) + \
                '_learning_rate' + str(self.learning_rate.value) + \
                '_fine_tunning_' + str(self.fine_tuning_rate.value)

        File.remove_dir(File.make_path(dataset, ".tmp"))

        train_generator, validation_generator, test_generator = self.make_dataset(
            dataset)

        # Save the model according to the conditions
        if self.save_weights:
            if not os.path.exists("../models_checkpoints/"):
                os.makedirs("../models_checkpoints/")

            checkpoint = ModelCheckpoint("../models_checkpoints/" + self.file_name + ".h5", monitor='val_acc',
                                         verbose=1, save_best_only=True, save_weights_only=False,
                                         mode='auto', period=1)
        else:
            checkpoint = None

        self.model = self.select_model_params(train_generator.num_classes)

        tensorboard = TensorBoard(
            log_dir="../models_checkpoints/logs_" + self.file_name, write_images=False)
        # tensorboard.set_model(self.model)
        # compile the model
        self.model.compile(loss="categorical_crossentropy",
                           optimizer=optimizers.SGD(
                               lr=self.learning_rate.value, momentum=self.momentum.value),
                           metrics=["accuracy"])

        # Train the model
        self.model.fit_generator(
            train_generator,
            steps_per_epoch=train_generator.samples // self.batch_size.value,
            epochs=self.epochs.value,
            callbacks=[checkpoint, tensorboard],
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // self.batch_size.value)

        if self.save_weights:
            # self.model.save_weights(
            #    "../models_checkpoints/" + self.file_name + ".h5")
            self.model.save(
                "../models_checkpoints/" + self.file_name + "_model.h5")
            self.weight_path = "../models_checkpoints/" + self.file_name + "_model.h5"

            dict_classes = validation_generator.class_indices
            np.save("../models_checkpoints/" + self.file_name +
                    "_classes.npy", dict_classes)

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

    def select_model_params(self, num_classes):
        if self.fine_tuning_rate.value != -1:
            if self.architecture.value == "Xception":
                self.app = 0
                model = applications.Xception(
                    weights="imagenet", include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
            elif self.architecture.value == "VGG16":
                self.app = 1
                model = applications.VGG16(
                    weights="imagenet", include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
            elif self.architecture.value == "VGG19":
                self.app = 2
                model = applications.VGG19(
                    weights="imagenet", include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
            elif self.architecture.value == "ResNet50":
                self.app = 3
                model = applications.ResNet50(
                    weights="imagenet", include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
            elif self.architecture.value == "InceptionV3":
                self.app = 4
                model = applications.InceptionV3(
                    weights="imagenet", include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
            elif self.architecture.value == "MobileNet":
                self.app = 5
                model = applications.MobileNet(
                    weights="imagenet", include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))

            for layer in model.layers[:int(len(model.layers) * (self.fine_tuning_rate.value / 100.0))]:
                layer.trainable = False

        else:  # without transfer learning
            if self.architecture.value == "Xception":
                self.app = 0
                model = applications.Xception(
                    weights=None, include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
            elif self.architecture.value == "VGG16":
                self.app = 1
                model = applications.VGG16(
                    weights=None, include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
            elif self.architecture.value == "VGG19":
                self.app = 2
                model = applications.VGG19(
                    weights=None, include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
            elif self.architecture.value == "ResNet50":
                self.app = 3
                model = applications.ResNet50(
                    weights=None, include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
            elif self.architecture.value == "InceptionV3":
                self.app = 4
                model = applications.InceptionV3(
                    weights=None, include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
            elif self.architecture.value == "MobileNet":
                self.app = 5
                model = applications.MobileNet(
                    weights=None, include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
            for layer in model.layers:
                layer.trainable = True

        # Adding custom Layers
        new_custom_layers = model.output
        new_custom_layers = Flatten()(new_custom_layers)
        new_custom_layers = Dense(1024, activation="relu")(new_custom_layers)
        new_custom_layers = Dropout(0.5)(new_custom_layers)
        new_custom_layers = Dense(1024, activation="relu")(new_custom_layers)
        predictions = Dense(num_classes,
                            activation="softmax")(new_custom_layers)

        # creating the final model
        model = Model(inputs=model.input, outputs=predictions)

        return model

    def make_dataset(self, dataset):

        # create symbolic links to the dataset
        KERAS_DATASET_DIR_NAME = ".keras_dataset"
        #KERAS_DATASET_DIR_NAME = File.make_path("..", os.path.split(dataset)[-1] + "_keras_dataset")
        KERAS_DIR_TRAIN_NAME = "train"
        KERAS_DIR_VALIDATION_NAME = "validation"
        KERAS_DIR_TEST_NAME = "test"
        PERC_TRAIN = self.perc_train.value
        PERC_VALIDATION = self.perc_validation.value

        # create keras dir dataset
        if not os.path.exists(File.make_path(dataset, KERAS_DATASET_DIR_NAME)) or self.recreate_dataset.value:
            if os.path.exists(File.make_path(dataset, KERAS_DATASET_DIR_NAME)):
                shutil.rmtree(File.make_path(dataset, KERAS_DATASET_DIR_NAME))

            os.makedirs(File.make_path(dataset, KERAS_DATASET_DIR_NAME))

            # create keras dir train
            if not os.path.exists(File.make_path(dataset, KERAS_DATASET_DIR_NAME, KERAS_DIR_TRAIN_NAME)):
                os.makedirs(File.make_path(
                    dataset, KERAS_DATASET_DIR_NAME, KERAS_DIR_TRAIN_NAME))

            # create keras dir validation
            if not os.path.exists(File.make_path(dataset, KERAS_DATASET_DIR_NAME, KERAS_DIR_VALIDATION_NAME)):
                os.makedirs(File.make_path(
                    dataset, KERAS_DATASET_DIR_NAME, KERAS_DIR_VALIDATION_NAME))

            # create keras dir test
            if not os.path.exists(File.make_path(dataset, KERAS_DATASET_DIR_NAME, KERAS_DIR_TEST_NAME)):
                os.makedirs(File.make_path(
                    dataset, KERAS_DATASET_DIR_NAME, KERAS_DIR_TEST_NAME))

            dir_classes = sorted(File.list_dirs(dataset))

            if KERAS_DATASET_DIR_NAME in dir_classes:
                dir_classes.remove(KERAS_DATASET_DIR_NAME)

            for dir_class in dir_classes:
                root = File.make_path(dataset, dir_class)
                files = os.listdir(root)
                random.shuffle(files)
                quant_files = len(files)
                quant_train = int((quant_files / 100.0) * PERC_TRAIN)
                quant_validation = int((quant_files / 100.0) * PERC_VALIDATION)

                files_train = files[0:quant_train]
                files_validation = files[quant_train:quant_train +
                                         quant_validation]
                files_test = files[quant_train + quant_validation:quant_files]
                print("Processing class %s - %d itens - %d train items - %d validation items" %
                      (dir_class, quant_files, quant_train, quant_validation))

                for file in files_train:
                    dir_class_train = File.make_path(
                        dataset, KERAS_DATASET_DIR_NAME, KERAS_DIR_TRAIN_NAME, dir_class)
                    if not os.path.exists(dir_class_train):
                        os.makedirs(dir_class_train)

                    if os.path.splitext(file)[-1] == ".tif":
                        img = Image.open(File.make_path(root, file))
                        # img.thumbnail(img.size)
                        new_file = os.path.splitext(file)[0] + ".png"
                        img.save(File.make_path(dir_class_train,
                                                new_file), "PNG", quality=100)
                    else:
                        print(100*'-')
                        print(File.make_path(root, file))
                        print(100*'-')
                        os.symlink(File.make_path(root, file),
                                   File.make_path(dir_class_train, file))

                for file in files_validation:
                    dir_class_validation = File.make_path(
                        dataset, KERAS_DATASET_DIR_NAME, KERAS_DIR_VALIDATION_NAME, dir_class)
                    if not os.path.exists(dir_class_validation):
                        os.makedirs(dir_class_validation)

                    if os.path.splitext(file)[-1] == ".tif":
                        img = Image.open(File.make_path(root, file))
                        # img.thumbnail(img.size)
                        new_file = os.path.splitext(file)[0] + ".png"
                        img.save(File.make_path(dir_class_validation,
                                                new_file), "PNG", quality=100)
                    else:
                        os.symlink(File.make_path(root, file),
                                   File.make_path(dir_class_validation, file))

                for file in files_test:
                    dir_class_test = File.make_path(
                        dataset, KERAS_DATASET_DIR_NAME, KERAS_DIR_TEST_NAME, dir_class)
                    if not os.path.exists(dir_class_test):
                        os.makedirs(dir_class_test)

                    if os.path.splitext(file)[-1] == ".tif":
                        img = Image.open(File.make_path(root, file))
                        # img.thumbnail(img.size)
                        new_file = os.path.splitext(file)[0] + ".png"
                        img.save(File.make_path(dir_class_test,
                                                new_file), "PNG", quality=100)
                    else:
                        os.symlink(File.make_path(root, file),
                                   File.make_path(dir_class_test, file))

        train_datagen = ImageDataGenerator()

        train_generator = train_datagen.flow_from_directory(
            File.make_path(dataset, KERAS_DATASET_DIR_NAME,
                           KERAS_DIR_TRAIN_NAME),
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=self.batch_size.value,
            shuffle=True,
            class_mode="categorical")

        validation_datagen = ImageDataGenerator()

        validation_generator = validation_datagen.flow_from_directory(
            File.make_path(dataset, KERAS_DATASET_DIR_NAME,
                           KERAS_DIR_VALIDATION_NAME),
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=self.batch_size.value,
            shuffle=True,
            class_mode="categorical")

        test_datagen = ImageDataGenerator()

        test_generator = test_datagen.flow_from_directory(
            File.make_path(dataset, KERAS_DATASET_DIR_NAME,
                           KERAS_DIR_TEST_NAME),
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=self.batch_size.value,
            shuffle=True,
            class_mode="categorical")

        return train_generator, validation_generator, test_generator

    def single_classify(self, image_path, directory, extractors, dict_classes):
        preprocess_input, decode_predictions = dict_preprocessing[self.app]
        pil_image = load_img(image_path)
        np_image = img_to_array(pil_image)
        res_image = resize(np_image, (IMG_HEIGHT, IMG_WIDTH, 3))
        tensor = expand_dims(res_image, axis=0)
        tensor = preprocess_input(tensor)
        predict = self.model.predict(tensor)
        predict = np.argmax(predict, axis=1)
        return dict_classes[predict[0]]
