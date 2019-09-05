#!/usr/bin/python
# -*- coding: utf-8 -*-
#
"""
    Runs ImageNet Convolutional Neural Network implemented in software Caffe.
    This module only implements the classification. The network must be trained previously using caffe.
    
    Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton, Imagenet classification with deep convolutional neural networks, Advances in neural information processing systems, 2012.
    Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor, Caffe: Convolutional Architecture for Fast Feature Embedding, arXiv preprint arXiv:1408.5093, 2014.
    
    Name: cnn_caffe.py
    Author: Alessandro dos Santos Ferreira ( santosferreira.alessandro@gmail.com )
"""

# Make sure that caffe is on the python path:
caffe_root = '/var/tmp/caffe/' 
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

import cv2
import numpy as np
import os

from collections import OrderedDict

from util.config import Config
from util.file_utils import File
from util.utils import TimeUtils

from classifier import Classifier

class CNNCaffe(Classifier):
    
    # I tried to use the default python interface to perform the classification as explained at here:
    # http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb
    # But for some unknown reason it didn't work as expected, it generated poor results.
    # I kept the implementation anyway, to use must be set CREATE_LMDB = False.
    # Otherwise it will used another approach that generates optimal results.
    CREATE_LMDB = True

    def __init__(self):
        """Constructor.
        """
        self.model_def = Config("ModelDef", '../examples/deploy.prototxt', str)
        self.model_weights = Config("ModelWeights", '../examples/caffenet_train_iter_15000.caffemodel', str)
        self.mean_image = Config("MeanImage", '../examples/imagenet_mean.binaryproto', str)
        self.labels_file = Config("LabelsFile", '../examples/labels.txt', str)
        
        self._create_net()
        
    def _create_net(self):     
        self.net = caffe.Net(self.model_def.value,          # defines the structure of the model
                                self.model_weights.value,   # contains the trained weights
                                caffe.TEST)                 # use test mode (e.g., don't perform dropout)

        # create transformer for the input called 'data'
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})

        self.transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
        self.transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
        self.transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
    

    def get_config(self):
        """Return configuration of classifier. 
        
        Returns
        -------
        config : OrderedDict
            Current configs of classifier.
        """
        caffe_config = OrderedDict()
        
        caffe_config["model_def"] = self.model_def
        caffe_config["model_weights"] = self.model_weights
        caffe_config["mean_image"] = self.mean_image
        caffe_config["labels_file"] = self.labels_file
        
        return caffe_config
        
    def set_config(self, configs):
        """Update configuration of classifier. 
        
        Parameters
        ----------
        configs : OrderedDict
            New configs of classifier.
        """
        self.model_def = Config.nvl_config(configs["model_def"], self.model_def)
        self.model_weights = Config.nvl_config(configs["model_weights"], self.model_weights)
        self.mean_image = Config.nvl_config(configs["mean_image"], self.mean_image)
        self.labels_file = Config.nvl_config(configs["labels_file"], self.labels_file)
        
        self._create_net()

    def get_summary_config(self):
        """Return fomatted summary of configuration. 
        
        Returns
        -------
        summary : string
            Formatted string with summary of configuration.
        """
        caffe_config = OrderedDict()
        
        caffe_config[self.model_def.label] = self.model_def.value
        caffe_config[self.model_weights.label] = self.model_weights.value
        caffe_config[self.mean_image.label] = self.mean_image.value
        caffe_config[self.labels_file.label] = self.labels_file.value

        summary = ''
        for config in caffe_config:
            summary += "%s: %s\n" % (config, str(caffe_config[config]))
        
        return summary

    
    def classify(self, dataset, test_dir, test_data, image):
        """Perform the classification. 
        
        Parameters
        ----------
        dataset : string
            Path to image dataset.
        test_dir : string
            Name of test data directory.
        test_data : string
            Not used.
            
        Returns
        -------
        summary : list of string
            List of predicted classes for each instance in test data in ordered way.
        """
        # if CNNCaffe.CREATE_LMDB = True use the alternative approach.
        if CNNCaffe.CREATE_LMDB:
            return self._classify_lmdb(dataset, test_dir, test_data)
        
        test_dir = File.make_path(dataset, test_dir)
        
        classes = []
        labels = np.loadtxt(self.labels_file.value, str)

        images = sorted(os.listdir(File.make_path(test_dir)))
        
        # convert mean.binaryproto to mean.npy
        blob = caffe.proto.caffe_pb2.BlobProto()
        data = open( self.mean_image.value, 'rb' ).read()
        blob.ParseFromString(data)
        np.save( File.make_path(test_dir, 'mean.npy'), np.array( caffe.io.blobproto_to_array(blob) )[0] )
        
        # load the mean image for subtraction
        mu = np.load( File.make_path(test_dir, 'mean.npy') )
        mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
        
        self.transformer.set_mean('data', mu)   # subtract the dataset-mean value in each channel
        
        self.net.blobs['data'].reshape(1,          # batch size
                                        3,         # 3-channel (BGR) images
                                        227, 227)  # image size is 227x227

        for im in images:
            filepath = File.make_path(test_dir, im)
            image = cv2.imread(filepath)
                
            # resize the segment
            resized_image = np.zeros((512, 512, image.shape[2]), dtype="uint8")
            resized_image[0:image.shape[0], 0:image.shape[1]] = image[:,:]
            resized_image = resized_image[0:256, 0:256]
            cv2.imwrite(filepath.replace('.tif', '.jpeg'), resized_image)
            
            # load the image
            input_image = caffe.io.load_image(filepath)
            transformed_image = self.transformer.preprocess('data', input_image)
            
            # copy the image data into the memory allocated for the net
            self.net.blobs['data'].data[...] = [transformed_image]

            # perform classification
            output = self.net.forward()
            
            # the output probability vector for the each image in the batch
            prediction = output['prob'][0] 
            print(["%0.4f" % pr for pr in prediction ])

            # append the class with max probability.
            classes.append(labels[prediction.argmax()])

        return classes


    def _classify_lmdb(self, dataset, test_dir, test_data):
        """Perform the alternative classification creating LMDB backend. 
        
        Parameters
        ----------
        dataset : string
            Path to image dataset.
        test_dir : string
            Name of test data directory.
        test_data : string
            Not used.
            
        Returns
        -------
        summary : list of string
            List of predicted classes for each instance in test data in ordered way.
        """
        test_dir = File.make_path(dataset, test_dir)
            
        classes = []
        labels = np.loadtxt(self.labels_file.value, str)
        
        images = sorted(os.listdir(File.make_path(test_dir)))
        
        # create LMDB listfile
        listfile = open(File.make_path(test_dir, 'listfile.txt'), 'w')

        for im in images:
            filepath = File.make_path(test_dir, im)
            image = cv2.imread(filepath)
                
            # resize the segment and save in jpeg format
            resized_image = np.zeros((512, 512, image.shape[2]), dtype="uint8")
            resized_image[0:image.shape[0], 0:image.shape[1]] = image[:,:]
            resized_image = resized_image[0:256, 0:256]
            cv2.imwrite(filepath.replace('.tif', '.jpeg'), resized_image)
            
            # append imagename in listfile
            listfile.write("%s %d\n" %(im.replace('.tif', '.jpeg'), 0))

        listfile.close()
        
        # create LMDB backend to be used as source of data
        from subprocess import call
        
        call([caffe_root + 'build/tools/convert_imageset', 
                File.make_path(test_dir, ''),
                File.make_path(test_dir, 'listfile.txt'),
                File.make_path(test_dir, 'lmdb')])

        # read model_def
        with open(self.model_def.value, 'r') as model_def:
            prototxt = model_def.read()
        
        # change structure of layer data
        layers = prototxt.split('layer')
        layers[1] = (' {\n'
                    '    name: "data"\n'
                    '    type: "Data"\n'
                    '    top: "data"\n'
                    '    top: "label"\n'
                    '    transform_param {\n'
                    '        mirror: false\n'
                    '        crop_size: 227\n'
                    '        mean_file: "' + self.mean_image.value  + '"\n'
                    '    }\n'
                    '    data_param {\n'
                    '        source: "' + File.make_path(test_dir, 'lmdb') + '"\n'
                    '        batch_size: 1\n'
                    '        backend: LMDB\n'
                    '    }\n'
                    '}\n')
        prototxt = 'layer'.join(layers)
        
        # create new model_def
        new_model_def_path = File.make_path(test_dir, 'deploy.prototxt')
        with open(new_model_def_path, 'w') as new_model_def:
            new_model_def.write(prototxt)

        net = caffe.Net(new_model_def_path,         # defines the structure of the model
                        self.model_weights.value,   # contains the trained weights
                        caffe.TEST)                 # use test mode (e.g., don't perform dropout)
                                
        for im in images:
            # perform classification
            output = net.forward()
            
            # the output probability vector for the first image in the batch
            prediction = output['prob'][0] 
            print(["%0.4f" % pr for pr in prediction ])

            classes.append(labels[prediction.argmax()])

        return classes
