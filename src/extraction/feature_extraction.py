#!/usr/bin/python
# -*- coding: utf-8 -*-
#
"""
    Runs feature extraction algorithms.
    
    Name: feature_extractor.py
    Author: Alessandro dos Santos Ferreira ( santosferreira.alessandro@gmail.com )
"""

import io
import itertools
import os
import gc
import multiprocessing
from multiprocessing import Process, Manager
import threading
from interface.interface import InterfaceException as IException

from util.file_utils import File
from util.utils import ImageUtils
from util.utils import TimeUtils
import cv2
from .extractor import Extractor
from tqdm import tqdm
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

class FeatureExtractor(object):
    """Handle the feature extraction."""


    def __init__(self, extractors, tkParent=None):
        """Constructor.

        Parameters
        ----------
        extractor : list of Extractor
            Initial set of active extractors.
        """
        self.extractors = extractors
        self.tkParent=tkParent

    def extract_all(self, dataset, output_file=None, dirs=None, overwrite=True, processor_amd=False):
        self.processor_amd=processor_amd
        self.threads = []
        if self.processor_amd == True :
            self.data = Manager().list() #is a necessary because have a problem with use Process and normaly declaration
            self.labels = Manager().list()
            self.types = Manager().list()
        else:
            self.data = [] #is a necessary because have a problem with use Process and normaly declaration

            self.labels = []
            self.types = []


        """Runs the feature extraction algorithms on all images of dataset.

        Parameters
        ----------
        dataset : string
            Path to dataset.
        output_file : string, optional, default = None
            Name of output file continaing the features. If not informed is considered the name of dataset.
        dirs : list of string, optional, default = None
            List of directories to be serched. If not informed search in all directories with images inside dataset.
        overwrite : boolean, optional, default = True
            If False check if already exists a file containing the features.

        Returns
        -------
        out : tuple
            Returns a tuple containing the name of output file and time spent in milliseconds.

        Raises
        ------
        IException 'Please select at least one extractor'
            Empty list of extractors.
        IException 'Image %s is possibly corrupt'
            Error opening some image inside dataset.
        IException 'There are no images in dataset: %s'
            Dataset does not contain any image.
        """
        if len(self.extractors) == 0:
            raise IException("Please select at least one extractor")

        if output_file is None:
            output_file = File.get_filename(dataset)
        output_file = File.make_path(dataset, output_file + '.arff')

        # if already exists a output file and must not override, return current file
        if overwrite == False and os.path.isfile(output_file):
            return output_file, 0

        start_time = TimeUtils.get_time()

        classes = sorted(File.list_dirs(dataset))
        dirs = classes if dirs is None else dirs

        # Runs the feature extraction for all classes inside the dataset
        for cl in  dirs:
            # start job for each extractor
            self.job_extractor(dataset, cl, classes)


        # Output is in kb, here I convert it in Mb for readability
        RAM_stats = self.getRAMinfo()
        RAM_total = round(int(RAM_stats[0]) / 1000,1)
        RAM_used = round(int(RAM_stats[1]) / 1000,1)
        print("RAM Total : "+str(RAM_total))
        print("RAM Used : "+str(RAM_used))
        self.print_console("Wait a moment, the threads are processing "+str(len(self.threads)) +" images, it may be delayed depending on the size or quantity of the images!")
        with tqdm(total=len(self.threads)) as pbar:
            for  t in self.threads:
                 t.start()
                 if((RAM_total)<10000):#se menor que 10gb
                     RAM_stats = self.getRAMinfo()
                     RAM_used = round(int(RAM_stats[1]) / 1000,1)
                     if((RAM_total-RAM_used)<2000):
                         t.join()
                 pbar.update(1)
            pbar.close()

        self.print_console("Waiting for workers to finish extracting attributes from images!")
        with tqdm(total=len(self.threads)) as ppbar:
            for t in self.threads:
                t.join()

                ppbar.update(1)
            ppbar.close()
        self.print_console("The process was completed with "+str(len(self.threads))+" images!")
        if len(self.data) == 0:
            raise IException("There are no images in dataset: %s" % dataset)
        del self.threads
        gc.collect()
        # Save the output file in ARFF format
        # self._save_output(File.get_filename(dataset), classes, self.labels, self.types, self.data, output_file)
        self._save_output(File.get_filename(dataset), classes, self.labels[0], self.types[0], self.data, output_file)
        end_time = TimeUtils.get_time()

        return output_file, (end_time - start_time)

    # create one thread for folder
    def job_extractor(self, dataset, cl, classes):

        items = sorted(os.listdir(File.make_path(dataset, cl)))
        self.print_console("Processing class %s - %d itens" % (cl, len(items)))

        for  item in  items :
            if item.startswith('.'):
                continue

            if self.processor_amd==True :
                th = multiprocessing.Process(target=self.sub_job_extractor,args=(item, dataset, cl, classes))
            else:
                th = threading.Thread(target=self.sub_job_extractor,args=(item, dataset, cl, classes))

            self.threads.append(th)


    # create one thread each image for use extractor
    def sub_job_extractor(self, item, dataset, cl, classes):
        try:
            filepath = File.make_path(dataset, cl, item)
            image = cv2.imread(filepath)
            #image = self.equalize_size_image(image)

        except:
            raise IException("Image %s is possibly corrupt" % filepath)

        if len(self.data) > 0:
            if sys.version_info >= (3, 0):
                values = list(zip(*([extractor().run(image) for extractor in self.extractors])))
            else:
                values = list(itertools.chain.from_iterable(zip(*([extractor().run(image) for extractor in self.extractors]))[2]))
	    
            self.data.append(values + [cl if cl in classes else classes[0]])

        else:
            labs, tys, values = [list(itertools.chain.from_iterable(ret))
                                               for ret in
                                               zip(*(extractor().run(image) for extractor in self.extractors))]
            self.labels.append(labs)
            self.types.append(tys)
            self.data.append(values + [cl if cl in classes else classes[0]])
        image=None
        filepath=None
    def extract_one_file(self, dataset, image_path, output_file=None):
        """Runs the feature extraction algorithms on specific image.

        Parameters
        ----------
        dataset : string
            Path to dataset.
        image_path : string
            Path to image.
        output_file : string, optional, default = None
            Name of output file continaing the features. If not informed is considered the name of dataset.

        Returns
        -------
        out : tuple
            Returns a tuple containing the name of output file and time spent in milliseconds.

        Raises
        ------
        IException 'Please select at least one extractor'
            Empty list of extractors.
        IException 'Image %s is possibly corrupt'
            Error opening image.
        """
        if len(self.extractors) == 0:
            raise IException("Please select at least one extractor")

        if output_file is None:
            output_file = File.get_filename(dataset)
        output_file = File.make_path(dataset, output_file + '.arff')

        classes = sorted(File.list_dirs(dataset))

        start_time = TimeUtils.get_time()

        try:
            image = File.open_image(image_path, rgb=False)
        except:
            raise IException("Image %s is possibly corrupt" % filepath)

        labels, types, values = [list(itertools.chain.from_iterable(ret))
                                 for ret in zip(*([extractor().run(image) for extractor in self.extractors]))]

        self._save_output(File.get_filename(dataset), classes, labels, types, [values + [classes[0]]], output_file)

        end_time = TimeUtils.get_time()

        return output_file, (end_time - start_time)

    def _save_output(self, relation, classes, labels, types, data, output_file):
        """Save output file in ARFF format.
        
        Parameters
        ----------
        relation : string
            Name of relation.
        classes : list of string
            List of classes names.
        labels : list of string
            List of attributes names.
        types : list of string
            List of attributes types.
        data : list of list of string
            List of instances.
        output_file : string
            Path to output file.
        """

        arff = open(output_file, 'w')

        arff.write("%s %s\n\n" % ('@relation', relation))

        for label, t in zip(labels, types):
            arff.write("%s %s %s\n" % ('@attribute', label, t))

        arff.write("%s %s {%s}\n\n" % ('@attribute', 'classe', ', '.join(classes)))

        arff.write('@data\n\n')

        for instance in data:
            instance = map(str, instance)
            line = ",".join(instance)
            arff.write(line + "\n")

        arff.close()

    #method to equalize size of images
    def equalize_size_image(self, image):
            if (image.shape[0] > 1000):
                basewidth = 1000
                wpercent = (basewidth / float(image.shape[0] ))
                hsize = int((float(image.shape[1] ) * float(wpercent)))
                image = cv2.resize(image, (basewidth, hsize))
            elif (image.shape[1] > 1000):
                baseheight = 1000
                wpercent = (baseheight / float(image.shape[1] ))
                wsize = int((float(image.shape[1] ) * float(wpercent)))
                image = cv2.resize(image, (wsize, baseheight))
            elif (image.shape[1] <1000):
                baseheight = 1000
                wpercent = (baseheight / float(image.shape[1] ))
                wsize = int((float(image.shape[1] ) * float(wpercent)))
                image = cv2.resize(image, (wsize, baseheight))
            elif (image.shape[0] < 1000):
                basewidth = 1000
                wpercent = (basewidth / float(image.shape[0] ))
                hsize = int((float(image.shape[1] ) * float(wpercent)))
                image = cv2.resize(image, (basewidth, hsize))
            return image

    "Method for print message in console, Window or Both"
    def print_console(self,mensagem):
        if(self.tkParent==None):
            print(mensagem)
        else:
            print(mensagem)
            self.tkParent.append_log( mensagem)
            self.tkParent._root.update_idletasks()

    def getRAMinfo(self):
        p = os.popen('free')
        i = 0
        while 1:
            i = i + 1
            line = p.readline()
            if i==2:
                return(line.split()[1:4])
