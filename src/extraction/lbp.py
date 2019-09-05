#!/usr/bin/python
# -*- coding: utf-8 -*-
#
"""
    Extract LBP (Local Binary Patterns) features.
    
    Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns. Timo Ojala, Matti Pietikainen, Topi Maenpaa. 2002.

    Name: lpb.py
    Author: Alessandro dos Santos Ferreira ( santosferreira.alessandro@gmail.com )
"""
import numpy as np
from skimage import feature

from util.utils import ImageUtils

from .extractor import Extractor

class LBP(Extractor):
    """Implements LBP (Local Binary Patterns) feature extraction."""
    
    def __init__(self, lbp_radius = 2, n_bins = 18):
        """Constructor.
        
        Parameters
        ----------
        lbp_radius : float, optional, default = 2
            Radius of circle (spatial resolution of the operator).
        n_bins : integer, optional, default = 18
            Number of circularly symmetric neighbour set points (quantization of the angular space).
        """
        self.lbp_radius = lbp_radius
        self.n_bins = n_bins
        
    
    def run(self, image):
        """Extract LBP (Local Binary Patterns) features.
        
        Parameters
        ----------
        image : opencv image
            Image to be analyzed.
        
        Returns
        -------
        features : tuple
            Returns a tuple containing a list of labels, type and values for each feature extracted.
        """
        image_grayscale = ImageUtils.image_grayscale(image, bgr = True)
        
        lbp = feature.local_binary_pattern(image_grayscale, 8 * self.lbp_radius, self.lbp_radius, 'uniform')
        values, _ = np.histogram(lbp, normed=True, bins=self.n_bins, range=(0, self.n_bins))        
        
        labels = [m+n for m,n in zip(['lbp_'] * len(values),map(str,range(0,len(values))))]
        types = [Extractor.NUMERIC] * len(labels)

        return labels, types, list(values)
