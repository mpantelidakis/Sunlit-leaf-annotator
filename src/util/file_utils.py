#!/usr/bin/python
# -*- coding: utf-8 -*-
#
"""
    Provides file handle utilities.
    
    Name: file_utils.py
    Author: Alessandro dos Santos Ferreira ( santosferreira.alessandro@gmail.com )
"""

import cv2
#import numpy as np
import os
import shutil
from skimage.util import img_as_float
from scipy import ndimage
from PIL import Image
import numpy as np

class File(object):
    """Set of utilities to handle files."""

    @staticmethod
    def get_filename(filepath):
        """Return only the filename part from a filepath.
        
        Parameters
        ----------
        filepath : string
            Filepath of a file.
        
        Returns
        -------
        name : string
            Filename part from filepath.
        """
        path, name = os.path.split(filepath)
        
        return name
    
    @staticmethod
    def get_path(filepath):
        """Return only the path part from a filepath, excluding the name of file.
        
        Parameters
        ----------
        filepath : string
            Filepath of a file.
        
        Returns
        -------
        name : string
            Path from filepath, excluding the name of file.
        """
        path, name = os.path.split(filepath)
        
        return path    
        
    @staticmethod
    def open_image(filepath, rgb = True):
        """Open a image.
        
        Parameters
        ----------
        filepath : string
            Filepath of a file.
        rgb : boolean, optional, default = True
            If true return the image in RGB, else BGR.
        
        Returns
        -------
        image : opencv 3-channel color image.
            Return the image opened.
            
        Raises
        ------
        IOError
            Error opening the image.
        """
        image = cv2.imread(filepath)
        if image is None:
            raise IOError('Image not opened')
            
        #image = np.flipud(image)
        if rgb == True:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #return img_as_float(image)
        return image
        

    @staticmethod
    def open_image_lut(filepath):
        """Open a image.
        
        Parameters
        ----------
        filepath : string
            Filepath of a file.
        
        Returns
        -------
        image : 3-channel color image.
            Return the image opened.
            
        Raises
        ------
        IOError
            Error opening the image.
        """
        image = Image.open(filepath)
        if image is None:
            raise IOError('Image not opened')
            
        return np.array(image)
    

    @staticmethod
    def save_image(image, directory, filename, ext = '.tif'):
        """Save a image.
        
        Parameters
        ----------
        image : opencv image
            Image to be saved.
        directory : string
            Path where image must be saved.
        filename : string
            Name of image.
        ext : string, optional, default = '.tif'
            Extension which image must be saved in format .ext.
        
        Returns
        -------
        filepath : string
            Return the complete filepath where image was saved.
        """
        filepath = File.make_path(directory, filename + ext)
        if not os.path.exists(directory):
            os.makedirs(directory)
        cv2.imwrite(filepath, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        return filepath

    @staticmethod
    def save_class_image(image, dataset, directory, filename, idx, ext = '.tif'):
        """Save a class image.
        
        Parameters
        ----------
        image : opencv image
            Image to be saved.
        dataset : string
            Path do image dataset.
        directory : string
            Directory where image must be saved.
        image : string
            Name of image.
        image : string
            Name of image.
        idx : integer
            Index of image inside the class.
        ext : string, optional, default = '.tif'
            Extension which image must be saved in format .ext.
        
        Returns
        -------
        filepath : string
            Return the complete filepath where image was saved.
        """
        for root, dirs, files in os.walk(dataset):
            for d in dirs:
                filepath = File.make_path(dataset, d, filename + '_%05d' % idx + ext)
                if(os.path.isfile(filepath)):
                    os.remove(filepath)

        return File.save_image(image, File.make_path(dataset, directory), filename + '_%05d' % idx, ext)

    @staticmethod
    def save_only_class_image(image, dataset, directory, filename, idx, ext = '.tif'):
        """Save a class image only.
        
        Parameters
        ----------
        image : opencv image
            Image to be saved.
        dataset : string
            Path do image dataset.
        directory : string
            Directory where image must be saved.
        image : string
            Name of image.
        image : string
            Name of image.
        idx : integer
            Index of image inside the class.
        ext : string, optional, default = '.tif'
            Extension which image must be saved in format .ext.
        
        Returns
        -------
        filepath : string
            Return the complete filepath where image was saved.
        """
        return File.save_image(image, File.make_path(dataset, directory), filename + '_%05d' % idx, ext)
    @staticmethod
    def list_dirs(dataset):
        """List all directories inside the dataset.
        
        Parameters
        ----------
        dataset : string
            Filepath to dataset.
        
        Returns
        -------
        dirs : list of string
            Return a list containing the name of all directories inside the dataset.
        """
        return [name for name in os.listdir(dataset) 
                    if os.path.isdir(os.path.join(dataset, name)) and not name.startswith('.')]

    @staticmethod
    def create_dir(directory):
        """Make a directory.
        
        Parameters
        ----------
        dataset : string
            Filepath to directory to be created.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
            
    @staticmethod
    def remove_dir(directory):
        """Delete a directory recursively if needed.
        
        Parameters
        ----------
        dataset : string
            Filepath to directory to be removed.
        """
        if os.path.exists(directory):
            shutil.rmtree(directory)
            
    @staticmethod
    def make_path(*dirs):
        """Make a path from a list.
        
        Parameters
        ----------
        dirs : list of arguments 
            Ordered list used to make a path.
        """
        return '/'.join([ d for d in dirs])
