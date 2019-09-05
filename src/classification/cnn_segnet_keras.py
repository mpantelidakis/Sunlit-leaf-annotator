import time
import os
import shutil
import random
import numpy as np
import json
import itertools
from PIL import Image
from matplotlib import pyplot as plt

import keras
from keras import applications, models, optimizers
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers import Dropout, Flatten, Dense, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Reshape, Permute
from keras.callbacks import ModelCheckpoint, TensorBoard

from interface.interface import InterfaceException as IException
from classification.classifier import Classifier
from classification.segnet_keras import SEGNETKeras
from classification.cnn_keras import CNNKeras

from collections import OrderedDict

from util.config import Config
from util.file_utils import File
from util.utils import TimeUtils

from skimage.transform import resize


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warnings
START_TIME = time.time()


IMG_WIDTH, IMG_HEIGHT = 1024, 1024


class CNN_SEGNET_Keras(Classifier):

    def __init__(self, path_segnet="/home/diogo/Documentos/pynovisao/pynovisao/BaseOriginal_BOIEMPE/segnet/models_segnet_checkpoints/VGG19_learning_rate0.01_fine_tunning_0_model.h5", path_cnn="/home/diogo/Documentos/pynovisao/pynovisao/BaseOriginal_BOIEMPE/superpixels_CNN/models_checkpoints_k_1000/ResNet50_learning_rate0.001_fine_tunning_0_model.h5", architecture_segnet="VGG19", combination_type = "POSPROC", architecture_cnn="ResNet50", learning_rate=0.01, momentum=0.9, batch_size=1, epochs=50, fine_tuning_rate=0, transfer_learning=False, save_weights=True, perc_train=80, perc_validation=20, recreate_dataset=False, num_classes = 2):
        self.path_segnet = Config(
            "Path Segnet", path_segnet, str)
        self.path_cnn = Config(
            "Path Cnn", path_cnn, str)
        self.architecture_segnet = Config(
            "Architecture Segnet", architecture_segnet, str)
        self.architecture_cnn = Config(
            "Architecture Cnn", architecture_cnn, str)
        self.combination_type = Config(
            "Combination Type", combination_type, str)
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
        self.num_classes = Config(
            "Num Classes", num_classes, int)
        self.file_name = "segnet_keras"

        self.model_segnet = None
        self.model_cnn = None

        self.trained = False

    def get_config(self):
        """Return configuration of classifier.

        Returns
        -------
        config : OrderedDict
            Current configs of classifier.
        """
        keras_config = OrderedDict()

        keras_config["Path Segnet"] = self.path_segnet
        keras_config["Path Cnn"] = self.path_cnn
        keras_config["Architecture Segnet"] = self.architecture_segnet
        keras_config["Architecture Cnn"] = self.architecture_cnn
        keras_config["Combination Type"] = self.combination_type
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
        keras_config["Num Classes"] = self.num_classes
        return keras_config

    def set_config(self, configs):
        """Update configuration of classifier.

        Parameters
        ----------
        configs : OrderedDict
            New configs of classifier.
        """

        self.path_segnet = Config.nvl_config(configs["Path Segnet"], self.path_segnet)
        self.path_cnn = Config.nvl_config(configs["Path Cnn"], self.path_cnn)
        self.architecture_segnet = Config.nvl_config(configs["Architecture Segnet"], self.architecture_segnet)
        self.architecture_cnn = Config.nvl_config(configs["Architecture Cnn"], self.architecture_cnn)
        self.combination_type = Config.nvl_config(configs["Combination Type"], self.combination_type)
        self.learning_rate = Config.nvl_config(configs["Learning rate"], self.learning_rate)
        self.momentum = Config.nvl_config(configs["Momentum"], self.momentum)
        self.batch_size = Config.nvl_config(configs["Batch size"], self.batch_size)
        self.epochs = Config.nvl_config(configs["Epochs"], self.epochs)
        self.fine_tuning_rate = Config.nvl_config(configs["Fine Tuning rate"], self.fine_tuning_rate)
        self.transfer_learning = Config.nvl_config(configs["Transfer Learning"], self.transfer_learning)
        self.save_weights = Config.nvl_config(configs["Save weights"], self.save_weights)
        self.perc_train = Config.nvl_config(configs["Perc Train"], self.perc_train)
        self.perc_validation = Config.nvl_config(configs["Perc Validation"], self.perc_validation)
        self.recreate_dataset = Config.nvl_config(configs["Recreate Dataset"], self.recreate_dataset)
        self.num_classes = Config.nvl_config(configs["Num Classes"], self.num_classes)

    def get_summary_config(self):
        """Return fomatted summary of configuration.

        Returns
        -------
        summary : string
            Formatted string with summary of configuration.
        """
        keras_config = OrderedDict()

        keras_config[self.path_segnet.label] = self.path_segnet.value
        keras_config[self.path_cnn.label] = self.path_cnn.value
        keras_config[self.architecture_segnet.label] = self.architecture_segnet.value
        keras_config[self.architecture_cnn.label] = self.architecture_cnn.value
        keras_config[self.combination_type.label] = self.combination_type.value
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
        keras_config[self.num_classes.label] = self.num_classes.value
        summary = ''
        for config in keras_config:
            summary += "%s: %s\n" % (config, str(keras_config[config]))

        return summary



    def classify(self, dataset, test_dir, test_data, image):
        try:
            #self.model.load_weights(
                #"../models_checkpoints/" + self.file_name + ".h5")
            K.clear_session()
            if self.path_segnet.value is not None:
                self.model_segnet = SEGNETKeras(architecture=self.architecture_segnet.value)
                self.model_segnet.weight_path = self.path_segnet.value
                self.model_segnet.IMG_HEIGHT = IMG_HEIGHT
                self.model_segnet.IMG_WIDTH = IMG_WIDTH

        except Exception, e:
            raise IException("Can't load the model in " +
                             self.path_segnet.value + str(e))

        try:
            #self.model.load_weights(
                #"../models_checkpoints/" + self.file_name + ".h5")
            if self.path_cnn.value is not None:
                self.model_cnn = CNNKeras(architecture=self.architecture_cnn.value)
                self.model_cnn.weight_path = self.path_cnn.value

        except Exception, e:
            raise IException("Can't load the model in " +
                             self.path_cnn.value + str(e))


        labeled_segnet = self.model_segnet.classify(dataset, test_dir, test_data, image)
        probs_segnet_small = self.model_segnet.probs[0]

        [h, w] = self.segmenter._segments.shape
        
        probs_segnet = np.zeros((h, w, self.num_classes.value))

        for i in range(self.num_classes.value):
            image_resized = resize(probs_segnet_small[:,:,i], (h, w))
            probs_segnet[:, :, i] = image_resized

        rgb = np.zeros((h, w, 3))
        rgb[:, :, 0] = probs_segnet[:,:,1]*255
        im = Image.fromarray(np.uint8(rgb))
        im.save('/home/diogo/to.png')

        print "PROBS SEGNET AQUI!!!!!!!"
        print probs_segnet
        print "SHAPE PROBSS SEGNET !!!!!!!!!"
        print probs_segnet.shape

        n_superpixels = len(self.segmenter.get_list_segments())
        if not self.combination_type.value == "POSPROC":
            labeled_cnn = self.model_cnn.classify(dataset, test_dir, test_data, image)
            probs_cnn_superpixels = self.model_cnn.probs

        
            probs_cnn = np.zeros((h, w, self.num_classes.value))
            for idx in range(n_superpixels):
                idxs = self.segmenter._segments == idx
                probs_cnn[idxs, 0] = probs_cnn_superpixels[idx,0]
                probs_cnn[idxs, 1] = probs_cnn_superpixels[idx,1]

            print "PROBS CNN AQUI!!!!!!!"
            print probs_cnn_superpixels
            print "SHAPE PROBSS CNN !!!!!!!!!"
            print probs_cnn_superpixels.shape
            np.save('probs_cnn', probs_cnn)


        if self.combination_type.value == "MAX":
            probs = self.max_combination(probs_segnet, probs_cnn)
            labeled = np.argmax(probs, axis=-1)

        if self.combination_type.value == "MEAN":
            probs = self.mean_combination(probs_segnet, probs_cnn)
            labeled = np.argmax(probs, axis=-1)

        if self.combination_type.value == "MULT":
            probs = self.mult_combination(probs_segnet, probs_cnn)
            labeled = np.argmax(probs, axis=-1)

        if self.combination_type.value == "AND":
            labeled = self.and_combination(probs_segnet, probs_cnn)

        if self.combination_type.value == "OR":
            labeled = self.or_combination(probs_segnet, probs_cnn)

        if self.combination_type.value == "POSPROC":
            labeled = self.posproc_combination(probs_segnet, self.segmenter._segments, n_superpixels)

        np.save('probs_segnet', probs_segnet)
        np.save('segments', self.segmenter._segments)

        return labeled

    def max_combination(self, probs_segnet, probs_cnn):
        return np.max((probs_segnet,probs_cnn),axis=0)

    def mean_combination(self, probs_segnet, probs_cnn):
        return np.mean((probs_segnet,probs_cnn),axis=0)

    def and_combination(self, probs_segnet, probs_cnn):
        labeled_segnet = np.argmax(probs_segnet, axis=-1)
        labeled_cnn = np.argmax(probs_cnn, axis=-1)
        labeled = labeled_segnet * labeled_cnn

        return labeled

    def or_combination(self, probs_segnet, probs_cnn):
        labeled_segnet = np.argmax(probs_segnet, axis=-1)
        labeled_cnn = np.argmax(probs_cnn, axis=-1)
        labeled2 = labeled_segnet + labeled_cnn
        labeled = np.zeros(labeled2.shape)
        labeled[labeled2>0] = 1

        return labeled

    def mult_combination(self, probs_segnet, probs_cnn):
        return probs_segnet*probs_cnn

    def posproc_combination(self, probs_segnet, segments, n_superpixels):
        labeled_segnet = np.argmax(probs_segnet, axis=-1)
        h, w = segments.shape
        labeled = np.zeros((h, w))
        for idx in range(n_superpixels):
            idxs = self.segmenter._segments == idx
            labeled_segment = labeled_segnet[idxs]
            hist_classes = np.zeros(self.num_classes.value)
            for i in range (self.num_classes.value):
                hist_classes[i] = np.sum(labeled_segment == i)
            
            class_segment = np.argmax(hist_classes)
            labeled[idxs] = class_segment

        return labeled

    def set_segmenter(self, segmenter):
        self.segmenter = segmenter


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

        [train_data, train_mask, validation_data, validation_mask, test_data, test_mask] = self.make_dataset(dataset)
        if self.save_weights:
            if not os.path.exists("../models_segnet_checkpoints/"):
                os.makedirs("../models_segnet_checkpoints/")
            
            checkpoint = ModelCheckpoint("../models_segnet_checkpoints/" + self.file_name + ".h5", monitor='val_acc',
                verbose=1, save_best_only=True, save_weights_only=False,
                mode='auto', period=1)
        else:
            checkpoint = None

        self.model = self.select_model(self.num_classes.value)
        self.model.summary()
        tensorboard = TensorBoard(log_dir="../models_segnet_checkpoints/logs_" + self.file_name, write_images=False)
        tensorboard.set_model(self.model)

        self.model.compile(loss="categorical_crossentropy",
            optimizer=optimizers.SGD(lr=self.learning_rate.value, momentum=self.momentum.value, decay=0.0005, nesterov=False),
            metrics=["accuracy"])

        self.model.fit(
            train_data,
            train_mask,
            batch_size=self.batch_size.value,
            #steps_per_epoch=self.train_samples // self.batch_size.value,
            epochs=self.epochs.value,
            callbacks=[checkpoint, tensorboard],
            validation_data=(validation_data, validation_mask))
            #validation_steps=self.validation_samples // self.batch_size.value)

        if self.save_weights:
            #self.model.save_weights(
            #    "../models_segnet_checkpoints/" + self.file_name + ".h5")
            self.model.save(
                "../models_segnet_checkpoints/" + self.file_name + "_model.h5")
            self.weight_path = "../models_segnet_checkpoints/" + self.file_name + "_model.h5"

            #dict_classes = validation_generator.class_indices
            #np.save("../models_segnet_checkpoints/" + self.file_name + "_classes.npy", dict_classes)


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




    def make_dataset(self, dataset):
        KERAS_DATASET_DIR_NAME = ".keras_dataset"
        #KERAS_DATASET_DIR_NAME = File.make_path("..", os.path.split(dataset)[-1] + "_keras_dataset")
        KERAS_DIR_TRAIN_NAME = "train"
        KERAS_DIR_TRAIN_MASK_NAME = "train_mask"
        KERAS_DIR_VALIDATION_NAME = "validation"
        KERAS_DIR_VALIDATION_MASK_NAME = "validation_mask"
        KERAS_DIR_TEST_NAME = "test"
        KERAS_DIR_TEST_MASK_NAME = "test_mask"
        PERC_TRAIN = self.perc_train.value
        PERC_VALIDATION = self.perc_validation.value


        # create keras dir dataset
        if not os.path.exists(File.make_path(dataset, KERAS_DATASET_DIR_NAME)) or self.recreate_dataset.value:
            if os.path.exists(File.make_path(dataset, KERAS_DATASET_DIR_NAME)):
                shutil.rmtree(File.make_path(dataset, KERAS_DATASET_DIR_NAME))

            os.makedirs(File.make_path(dataset, KERAS_DATASET_DIR_NAME))

            # create keras dir train
            os.makedirs(File.make_path(dataset, KERAS_DATASET_DIR_NAME, KERAS_DIR_TRAIN_NAME, 'images'))
            os.makedirs(File.make_path(dataset, KERAS_DATASET_DIR_NAME, KERAS_DIR_TRAIN_MASK_NAME, 'images'))
            os.makedirs(File.make_path(dataset, KERAS_DATASET_DIR_NAME, KERAS_DIR_VALIDATION_NAME, 'images'))
            os.makedirs(File.make_path(dataset, KERAS_DATASET_DIR_NAME, KERAS_DIR_VALIDATION_MASK_NAME, 'images'))
            os.makedirs(File.make_path(dataset, KERAS_DATASET_DIR_NAME, KERAS_DIR_TEST_NAME, 'images'))
            os.makedirs(File.make_path(dataset, KERAS_DATASET_DIR_NAME, KERAS_DIR_TEST_MASK_NAME, 'images'))

            valid_images_extension = ['.jpg', '.png', '.gif', '.jpeg', '.tif']
            fileimages = [name for name in os.listdir(dataset)
                    if os.path.splitext(name)[-1].lower() in valid_images_extension]

            random.shuffle(fileimages)
            quant_files = len(fileimages)
            quant_train = int(round((quant_files / 100.0) * PERC_TRAIN))
            quant_validation = int(round((quant_files / 100.0) * PERC_VALIDATION))

            files_train = fileimages[0:quant_train]
            files_validation = fileimages[quant_train:quant_train+quant_validation]
            files_test = fileimages[quant_train+quant_validation:quant_files]

            for file in files_train:
                if os.path.splitext(file)[-1] == ".tif":
                    img = Image.open(File.make_path(dataset, file))
                    new_file = os.path.splitext(file)[0]+".png"
                    img.save(File.make_path(dataset, KERAS_DATASET_DIR_NAME, KERAS_DIR_TRAIN_NAME, 'images', new_file), "PNG", quality=100)
                    img = Image.open(File.make_path(dataset, os.path.splitext(file)[0] + "_json", "label.png"))
                    img.save(File.make_path(dataset, KERAS_DATASET_DIR_NAME, KERAS_DIR_TRAIN_MASK_NAME, 'images', new_file), "PNG", quality=100)
                else:
                    os.symlink(File.make_path(dataset, file), File.make_path(dataset, KERAS_DATASET_DIR_NAME, KERAS_DIR_TRAIN_NAME, 'images', file))
                    os.symlink(File.make_path(dataset, os.path.splitext(file)[0] + "_json", "label.png"), File.make_path(dataset, KERAS_DATASET_DIR_NAME, KERAS_DIR_TRAIN_MASK_NAME, 'images', file))

            for file in files_validation:
                if os.path.splitext(file)[-1] == ".tif":
                    img = Image.open(File.make_path(dataset, file))
                    new_file = os.path.splitext(file)[0]+".png"
                    img.save(File.make_path(dataset, KERAS_DATASET_DIR_NAME, KERAS_DIR_VALIDATION_NAME, 'images', new_file), "PNG", quality=100)
                    img = Image.open(File.make_path(dataset, os.path.splitext(file)[0] + "_json", "label.png"))
                    img.save(File.make_path(dataset, KERAS_DATASET_DIR_NAME, KERAS_DIR_VALIDATION_MASK_NAME, 'images', new_file), "PNG", quality=100)
                else:
                    os.symlink(File.make_path(dataset, file), File.make_path(dataset, KERAS_DATASET_DIR_NAME, KERAS_DIR_VALIDATION_NAME, 'images', file))
                    os.symlink(File.make_path(dataset, os.path.splitext(file)[0] + "_json", "label.png"), File.make_path(dataset, KERAS_DATASET_DIR_NAME, KERAS_DIR_VALIDATION_MASK_NAME, 'images', file))

            for file in files_test:
                if os.path.splitext(file)[-1] == ".tif":
                    img = Image.open(File.make_path(dataset, file))
                    #img.thumbnail(img.size)
                    new_file = os.path.splitext(file)[0]+".png"
                    img.save(File.make_path(dataset, KERAS_DATASET_DIR_NAME, KERAS_DIR_TEST_NAME, 'images', new_file), "PNG", quality=100)
                    img = Image.open(File.make_path(dataset, os.path.splitext(file)[0] + "_json", "label.png"))
                    #img.thumbnail(img.size)
                    img.save(File.make_path(dataset, KERAS_DATASET_DIR_NAME, KERAS_DIR_TEST_MASK_NAME, 'images', new_file), "PNG", quality=100)
                else:
                    os.symlink(File.make_path(dataset, file), File.make_path(dataset, KERAS_DATASET_DIR_NAME, KERAS_DIR_TEST_NAME, 'images', file))
                    os.symlink(File.make_path(dataset, os.path.splitext(file)[0] + "_json", "label.png"), File.make_path(dataset, KERAS_DATASET_DIR_NAME, KERAS_DIR_TEST_MASK_NAME, 'images', file))


        train_data = self.get_images(File.make_path(dataset, KERAS_DATASET_DIR_NAME, KERAS_DIR_TRAIN_NAME, 'images'))
        train_mask = self.get_mask_images(File.make_path(dataset, KERAS_DATASET_DIR_NAME, KERAS_DIR_TRAIN_MASK_NAME, 'images'))

        validation_data = self.get_images(File.make_path(dataset, KERAS_DATASET_DIR_NAME, KERAS_DIR_VALIDATION_NAME, 'images'))
        validation_mask = self.get_mask_images(File.make_path(dataset, KERAS_DATASET_DIR_NAME, KERAS_DIR_VALIDATION_MASK_NAME, 'images'))

        test_data = self.get_images(File.make_path(dataset, KERAS_DATASET_DIR_NAME, KERAS_DIR_TEST_NAME, 'images'))
        test_mask = self.get_mask_images(File.make_path(dataset, KERAS_DATASET_DIR_NAME, KERAS_DIR_TEST_MASK_NAME, 'images'))

        #rgb = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3))
        #rgb[:, :, 0] = train_mask[0,:,:,1]*255 
        #im = Image.fromarray(np.uint8(rgb))
        #im.save('/home/diogo/to.png')
        
        self.train_samples = train_data.shape[0]
        self.validation_samples = validation_data.shape[0]
        self.test_samples = test_data.shape[0]

        return train_data, train_mask, validation_data, validation_mask, test_data, test_mask

    def get_images(self, path):
        data = []
        valid_images_extension = ['.jpg', '.png', '.gif', '.jpeg', '.tif']
        file_imgs = [name for name in os.listdir(path) if os.path.splitext(name)[-1].lower() in valid_images_extension]

        for i, fi in enumerate(file_imgs):
            img = Image.open(os.path.join(path, fi))
            img = img.resize((IMG_HEIGHT, IMG_WIDTH))
            img = np.array(img)
            print "opening image " + str(i) + " of " + str(len(file_imgs))

            data.append(img)

        data = np.array(data)
        return data


    def get_mask_images(self, pathLabels):
        label = []
        extenstions = ['jpg', 'bmp', 'png', 'gif', 'JPG']
        file_imgs = [f for f in os.listdir(pathLabels) if any(f.endswith(ext) for ext in extenstions)]

        for i, fi in enumerate(file_imgs):
            gt = Image.open(os.path.join(pathLabels, fi))
            gt = gt.resize((IMG_HEIGHT, IMG_WIDTH))
            gt = np.array(gt)
            label.append(self.label_map(gt, self.num_classes.value))
            print "opening mask image " + str(i) + " of " + str(len(file_imgs))

        label = np.array(label).reshape((len(file_imgs), IMG_HEIGHT * IMG_WIDTH, self.num_classes.value))
        #label = np.array(label)
        return label

    def label_map(self, labels, n_labels):
        [img_h, img_w] = labels.shape
        label_map = np.zeros([img_h, img_w, n_labels], dtype=np.int8)
        for r in range(img_h):
            for c in range(img_w):
                label_map[r, c, labels[r][c]] = 1
        return label_map


    def select_model(self, num_classes):
        if self.fine_tuning_rate.value != -1:
            if self.architecture.value == "VGG16":
                model = self.getVGG16Model(IMG_WIDTH, IMG_HEIGHT, num_classes, True)
            elif self.architecture.value == "VGG19":
                model = self.getVGG19Model(IMG_WIDTH, IMG_HEIGHT, num_classes, True)
            elif self.architecture.value == "ResNet50":
                model = self.getResNet50Model(IMG_WIDTH, IMG_HEIGHT, num_classes, True)

            for layer in model.layers[:int(len(model.layers) * (self.fine_tuning_rate.value / 100.0))]:
                layer.trainable = False

        else:  # without transfer learning
            if self.architecture.value == "VGG16":
                model = self.getVGG16Model(IMG_WIDTH, IMG_HEIGHT, num_classes, False)
            elif self.architecture.value == "VGG19":
                model = self.getVGG19Model(IMG_WIDTH, IMG_HEIGHT, num_classes, False)
            elif self.architecture.value == "ResNet50":
                model = applications.ResNet50(
                    weights=None, include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))

            for layer in model.layers:
                layer.trainable = True

        return model

    def getVGG16Model(self, img_w=224, img_h=224, n_labels=2, pretrainedVGG=False):

        kernel = 3

        model = models.Sequential()
        model.add( Conv2D(filters=64, kernel_size=(kernel, kernel), padding='same', input_shape=(img_h, img_w, 3), name='block1_conv1') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( Conv2D(filters=64, kernel_size=(kernel, kernel), padding='same', name='block1_conv2') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( MaxPooling2D() )

        model.add( Conv2D(filters=128, kernel_size=(kernel, kernel), padding='same', name='block2_conv1') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( Conv2D(filters=128, kernel_size=(kernel, kernel), padding='same', name='block2_conv2') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( MaxPooling2D() )

        model.add( Conv2D(filters=256, kernel_size=(kernel, kernel), padding='same', name='block3_conv1') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( Conv2D(filters=256, kernel_size=(kernel, kernel), padding='same', name='block3_conv2') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( Conv2D(filters=256, kernel_size=(kernel, kernel), padding='same', name='block3_conv3') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( MaxPooling2D() )

        model.add( Conv2D(filters=512, kernel_size=(kernel, kernel), padding='same', name='block4_conv1') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( Conv2D(filters=512, kernel_size=(kernel, kernel), padding='same', name='block4_conv2') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( Conv2D(filters=512, kernel_size=(kernel, kernel), padding='same', name='block4_conv3') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( MaxPooling2D() )

        model.add( Conv2D(filters=512, kernel_size=(kernel, kernel), padding='same', name='block5_conv1') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( Conv2D(filters=512, kernel_size=(kernel, kernel), padding='same', name='block5_conv2') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( Conv2D(filters=512, kernel_size=(kernel, kernel), padding='same', name='block5_conv3') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( MaxPooling2D() )

        if pretrainedVGG:
            weights_path = os.path.expanduser(os.path.join('~', '.keras', 'models', 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'))
            model.load_weights(weights_path, by_name=True)

        #decoder
        model.add( UpSampling2D() )
        model.add( Conv2D(512, kernel_size=(kernel, kernel), padding='same') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( Conv2D(512, kernel_size=(kernel, kernel), padding='same') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( Conv2D(512, kernel_size=(kernel, kernel), padding='same') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )

        model.add( UpSampling2D() )
        model.add( Conv2D(512, kernel_size=(kernel, kernel), padding='same') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( Conv2D(512, kernel_size=(kernel, kernel), padding='same') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( Conv2D(256, kernel_size=(kernel, kernel), padding='same') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )

        model.add( UpSampling2D() )
        model.add( Conv2D(256, kernel_size=(kernel, kernel), padding='same') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( Conv2D(256, kernel_size=(kernel, kernel), padding='same') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( Conv2D(128, kernel_size=(kernel, kernel), padding='same') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )

        model.add( UpSampling2D() )
        model.add( Conv2D(128, kernel_size=(kernel, kernel), padding='same') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( Conv2D(64, kernel_size=(kernel, kernel), padding='same') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )

        model.add( UpSampling2D() )
        model.add( Conv2D(64, kernel_size=(kernel, kernel), padding='same') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( Conv2D(n_labels, kernel_size=(1, 1), padding='valid') )
        model.add( BatchNormalization() )

        model.add(Reshape((img_h * img_w, n_labels)))
        model.add(Activation('softmax'))

        return model


    def getVGG19Model(self, img_w=224, img_h=224, n_labels=2, pretrainedVGG=False):

        kernel = 3

        model = models.Sequential()
        model.add(Conv2D(64, (3, 3), padding='same', input_shape=(img_h, img_w, 3), name='block1_conv1'))
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add(Conv2D(64, (3, 3), padding='same', name='block1_conv2'))
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( MaxPooling2D() )

        model.add( Conv2D(filters=128, kernel_size=(kernel, kernel), padding='same', name='block2_conv1') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( Conv2D(filters=128, kernel_size=(kernel, kernel), padding='same', name='block2_conv2') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( MaxPooling2D() )

        model.add( Conv2D(filters=256, kernel_size=(kernel, kernel), padding='same', name='block3_conv1') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( Conv2D(filters=256, kernel_size=(kernel, kernel), padding='same', name='block3_conv2') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( Conv2D(filters=256, kernel_size=(kernel, kernel), padding='same', name='block3_conv3') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( Conv2D(filters=256, kernel_size=(kernel, kernel), padding='same', name='block3_conv4') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( MaxPooling2D() )

        model.add( Conv2D(filters=512, kernel_size=(kernel, kernel), padding='same', name='block4_conv1') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( Conv2D(filters=512, kernel_size=(kernel, kernel), padding='same', name='block4_conv2') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( Conv2D(filters=512, kernel_size=(kernel, kernel), padding='same', name='block4_conv3') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( Conv2D(filters=512, kernel_size=(kernel, kernel), padding='same', name='block4_conv4') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( MaxPooling2D() )

        model.add( Conv2D(filters=512, kernel_size=(kernel, kernel), padding='same', name='block5_conv1') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( Conv2D(filters=512, kernel_size=(kernel, kernel), padding='same', name='block5_conv2') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( Conv2D(filters=512, kernel_size=(kernel, kernel), padding='same', name='block5_conv3') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( Conv2D(filters=512, kernel_size=(kernel, kernel), padding='same', name='block5_conv4') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( MaxPooling2D() )


        if pretrainedVGG:
            weights_path = os.path.expanduser(os.path.join('~', '.keras', 'models', 'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'))
            model.load_weights(weights_path, by_name=True)


        model.add( UpSampling2D() )
        model.add( Conv2D(512, kernel_size=(kernel, kernel), padding='same') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( Conv2D(512, kernel_size=(kernel, kernel), padding='same') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( Conv2D(512, kernel_size=(kernel, kernel), padding='same') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( Conv2D(512, kernel_size=(kernel, kernel), padding='same') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )


        model.add( UpSampling2D() )
        model.add( Conv2D(512, kernel_size=(kernel, kernel), padding='same') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( Conv2D(512, kernel_size=(kernel, kernel), padding='same') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( Conv2D(512, kernel_size=(kernel, kernel), padding='same') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( Conv2D(256, kernel_size=(kernel, kernel), padding='same') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )


        model.add( UpSampling2D() )
        model.add( Conv2D(256, kernel_size=(kernel, kernel), padding='same') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( Conv2D(256, kernel_size=(kernel, kernel), padding='same') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( Conv2D(256, kernel_size=(kernel, kernel), padding='same') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( Conv2D(128, kernel_size=(kernel, kernel), padding='same') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )


        model.add( UpSampling2D() )
        model.add( Conv2D(128, kernel_size=(kernel, kernel), padding='same') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( Conv2D(64, kernel_size=(kernel, kernel), padding='same') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )


        model.add( UpSampling2D() )
        model.add( Conv2D(64, kernel_size=(kernel, kernel), padding='same') )
        model.add( BatchNormalization() )
        model.add( Activation('relu') )
        model.add( Conv2D(n_labels, kernel_size=(1, 1), padding='valid') )
        model.add( BatchNormalization() )

        model.add(Reshape((img_h * img_w, n_labels)))
        model.add(Activation('softmax'))

        return model


    def identity_block(self, input_tensor, kernel_size, filters, stage, block):
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)
        x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)
        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
        x = keras.layers.add([x, input_tensor])
        x = Activation('relu')(x)
        return x


    def conv_block(self, input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)
        x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)
        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
        shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)
        x = keras.layers.add([x, shortcut])
        x = Activation('relu')(x)
        return x

    def getResNet50Model(self, img_w=224, img_h=224, n_labels=2, pretrainedVGG=False):

        bn_axis = 3
        #224,244
        img_input = Input(shape=(img_w, img_h, 3))
        #112,112
        x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(img_input)
        x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
        x = Activation('relu')(x)
        #55,55
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        #55,55
        x = self.conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        #28,28
        x = self.conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='d')

        #14,14
        x = self.conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

        #7,7
        x = self.conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

        #load
        x = self.identity_block(x, 3, [512, 512, 2048], stage=6, block='c')
        x = self.identity_block(x, 3, [512, 512, 2048], stage=6, block='b')
        x = UpSampling2D()(x)
        x = self.conv_block(x, 3, [512, 512, 1024], stage=6, block='a')

        x = UpSampling2D()(x)
        x = self.identity_block(x, 3, [256, 256, 1024], stage=7, block='f')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=7, block='e')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=7, block='d')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=7, block='c')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=7, block='b')
        x = UpSampling2D()(x)
        x = self.conv_block(x, 3, [256, 256, 512], stage=7, block='a')

        x = UpSampling2D()(x)
        x = self.identity_block(x, 3, [128, 128, 512], stage=8, block='d')
        x = self.identity_block(x, 3, [128, 128, 512], stage=8, block='c')
        x = self.identity_block(x, 3, [128, 128, 512], stage=8, block='b')
        x = UpSampling2D()(x)
        x = self.conv_block(x, 3, [128, 128, 256], stage=8, block='a')

        x = UpSampling2D()(x)
        x = self.identity_block(x, 3, [64, 64, 256], stage=9, block='c')
        x = self.identity_block(x, 3, [64, 64, 256], stage=9, block='b')
        x = self.conv_block(x, 3, [64, 64, 256], stage=9, block='a', strides=(1, 1))

        x = UpSampling2D()(x)
        x = Conv2D(64, (3, 3), padding='same', name='conv1_f1')(x)
        x = BatchNormalization(axis=bn_axis, name='bn_conv1_f1')(x)
        x = Activation('relu')(x)

        x = UpSampling2D()(x)
        x = Conv2D(64, (3, 3), padding='same', name='conv1_f2')(x)
        x = BatchNormalization(axis=bn_axis, name='bn_conv1_f2')(x)
        x = Activation('relu')(x)

        x = Conv2D(n_labels, kernel_size=(1, 1), padding='valid')(x)
        x = BatchNormalization()(x)
        x = Reshape((img_h * img_w, n_labels))(x)
        x = Activation('softmax')(x)

        model = Model(img_input, x, name='resnet50')
        model.summary()
        if pretrainedVGG:
            weights_path = os.path.expanduser(os.path.join('~', '.keras', 'models', 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'))
            model.load_weights(weights_path, by_name=True)

        return model
