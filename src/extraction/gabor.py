#!/usr/bin/python
# -*- coding: utf-8 -*-
#
"""
    Extract Gabor features.
    
    Hans G. Feichtinger, Thomas Strohmer: "Gabor Analysis and Algorithms", Birkhäuser, 1998
    
    Name: gabor.py
    Author: Hemerson Pistori (pistori@ucdb.br)

    Based on: https://github.com/riaanvddool/scikits-image/blob/master/doc/examples/plot_gabor.py

"""

import numpy as np
#import cupy as np

from util.utils import ImageUtils
from .extractor import Extractor
from scipy import ndimage as ndi
from skimage import feature
from skimage.filters import gabor_kernel
from skimage.util import img_as_float



class GABOR(Extractor):
    """Implements GABOR feature extraction."""

    
    def __init__(self):
        pass        
        
    
    def run(self, image):
        """Extract GABOR features.
        
        Parameters
        ----------
        image : opencv image
            Image to be analyzed.
        
        Returns
        -------
        features : tuple
            Returns a tuple containing a list of labels, type and values for each feature extracted.
        """
        #deprecaded
        def compute_feats(image, kernels):
            feats = np.zeros((len(kernels), 2), dtype=np.double)
            for k, kernel in enumerate(kernels): 
                filtered = ndi.convolve(image, kernel, mode='wrap') 
                feats[k, 0] = filtered.mean()
                feats[k, 1] = filtered.var()

                values.append(feats[k, 0])
                values.append(feats[k, 1])
            return feats 

        #deprecaded
        def power(image, kernel):
                image = (image - image.mean()) / image.std()
                return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                            ndi.convolve(image, np.imag(kernel), mode='wrap')**2)


        image_grayscale = ImageUtils.image_grayscale(image, bgr = True)
        image_float = img_as_float(image_grayscale)


        #Prepare filter bank kernels
        labels = []
        values = []
        kernels = []
        index = 0
        for theta in range(8):
            theta = theta / 8. * np.pi
            for sigma in (1, 3):
                for frequency in (0.01, 0.10, 0.25, 0.5,0.9):
                    kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
                    #kernels.append(kernel)

                    filtered = ndi.convolve(image_float, kernel, mode='wrap')
                    import time

                    values.append(filtered.mean())
                    values.append(filtered.var())
                    #print ("Thet_%f_Sigma_%i_Frequencia_%.2f" % (theta, sigma, frequency))
                    for stat in ("Mean", "Variance"):
                        labels.append("Thet_%f_Sigma_%i_Frequencia_%.2f_%s" % (theta, sigma, frequency, stat))

   
        #compute_feats(image_float,kernels)

        types = ['numeric'] * len(labels)

        return labels, types, values





