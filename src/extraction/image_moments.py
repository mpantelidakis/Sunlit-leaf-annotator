#!/usr/bin/python
# -*- coding: utf-8 -*-
#
"""
    Calculate raw, central and Hu's set of image moments.
    
    M. K. Hu, “Visual Pattern Recognition by Moment Invariants”, IRE Trans. Info. Theory, vol. IT-8, pp. 179-187, 1962
    
    Name: image_moments.py
    Author: Alessandro dos Santos Ferreira ( santosferreira.alessandro@gmail.com )

    Alter method HuMoments
"""
from skimage import measure
import cv2
from util.utils import ImageUtils
from skimage.measure import regionprops, moments, moments_central
from skimage.morphology import label
import numpy as np
from .extractor import Extractor

class RawCentralMoments(Extractor):
    """Calculate raw and central set of image moments."""
    
    def __init__(self):
        """Constructor.
        """
        self._moments_order = [(0, 1), (1, 0), (1, 1), (0, 2), (2, 0)]        
        
    
    def run(self, image):
        """Calculate raw and central set of image moments of order 1 and 2.
        
        Parameters
        ----------
        image : opencv image
            Image to be analyzed.
        
        Returns
        -------
        features : tuple
            Returns a tuple containing a list of labels, type and values for each feature extracted.
        """
        #raw_moments = moments(image)


        image_binary = ImageUtils.image_binary(image, bgr = True)
        
        m = measure.moments(image_binary)
        
        values_m = [m[p, q] for (p, q) in self._moments_order]
        labels_m = [M+str(p)+str(q) for M,(p,q) in zip(['M_'] * len(self._moments_order), self._moments_order)]
        
        row = m[0, 1] / m[0, 0]
        col = m[1, 0] / m[0, 0]
	
        mu = measure.moments_central(image_binary, center=(row, col), order=3)
        
        values_mu = [mu[p, q] for (p, q) in self._moments_order]
        labels_mu = [M+str(p)+str(q) for M,(p,q) in zip(['Mu_'] * len(self._moments_order), self._moments_order)]
        #print(values_mu)
        labels = labels_m + labels_mu
        types = [Extractor.NUMERIC] * len(labels)
        values = values_m + values_mu
        
        return labels, types, values


class HuMoments(Extractor):
    """Calculate Hu's set of image moments."""
    
    def __init__(self):
        pass
        
    
    def run(self, image):
        """Calculate Hu's set set of image moments.
        
        Parameters
        ----------
        image : opencv image
            Image to be analyzed.
        
        Returns
        -------
        features : tuple
            Returns a tuple containing a list of labels, type and values for each feature extracted.
        """

        """image_binary = ImageUtils.image_binary(image, bgr = True)
        
        m = measure.moments(image_binary)
        
        row = m[0, 1] / m[0, 0]
        col = m[1, 0] / m[0, 0]

        mu = measure.moments_central(image_binary, row, col)
        
        nu = measure.moments_normalized(mu)
        hu = measure.moments_hu(nu)

        values_hu = list(hu)"""

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        values_hu= cv2.HuMoments(cv2.moments(image)).flatten()

        values_hu = list(values_hu)

        values_hu= np.nan_to_num(values_hu)
        labels_hu = [m+n for m,n in zip(['Hu_'] * len(values_hu),map(str,range(0,len(values_hu))))]
        
        labels = labels_hu
        types = [Extractor.NUMERIC] * len(labels)
        values = values_hu




        return labels, types, values

