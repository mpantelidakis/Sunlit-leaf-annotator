#!/usr/bin/python
# -*- coding: utf-8 -*-
#
"""
    Extract HOG (Histogram of Oriented Gradient) features.
    
    Dalal, N and Triggs, B, Histograms of Oriented Gradients for Human Detection, IEEE Computer Society Conference on Computer Vision and Pattern Recognition 2005 San Diego, CA, USA
    
    Name: hog.py
    Author: Alessandro dos Santos Ferreira ( santosferreira.alessandro@gmail.com )
	
    Change parameter Visualise for Visualize because is deprecaded
    Date:02/01/2019
    Author: Diego Andre Sant Ana		 	
		
"""
from skimage import feature

from util.utils import ImageUtils

from .extractor import Extractor

class HOG(Extractor):
    """Implements HOG (Histogram of Oriented Gradient) feature extraction."""

    
    def __init__(self):
        pass        
        
    
    def run(self, image):
        """Extract HOG (Histogram of Oriented Gradient) features.
        
        Parameters
        ----------
        image : opencv image
            Image to be analyzed.
        
        Returns
        -------
        features : tuple
            Returns a tuple containing a list of labels, type and values for each feature extracted.
        """
        image_grayscale = ImageUtils.image_grayscale(image, bgr=True)
        image_128x128 = ImageUtils.image_resize(image_grayscale, 128, 128)

        values, _ = feature.hog(image_128x128, orientations=8, pixels_per_cell=(32, 32),
                                cells_per_block=(1, 1), visualise=True)

        labels = [m + n for m, n in zip(['hog_'] * len(values), map(str, range(0, len(values))))]
        types = [Extractor.NUMERIC] * len(labels)

        return labels, types, list(values)
