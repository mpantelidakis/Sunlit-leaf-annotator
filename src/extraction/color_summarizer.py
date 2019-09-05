#!/usr/bin/python
# -*- coding: utf-8 -*-
#
"""
    Calculate min, max, mean and standard deviation for color channels RGB, HSV and CIELab.
    
    Name: color_summarizer.py
    Author: Alessandro dos Santos Ferreira ( santosferreira.alessandro@gmail.com )
"""
import cv2
import numpy as np

from util.utils import ImageUtils

from .extractor import Extractor

class ColorStats(Extractor):
    """Implements color feature extraction."""
    
    def __init__(self):
        """Constructor.
        """
        pass 

    
    def run(self, image):
        """Calculate min, max, mean and standard deviation for color channels RGB, HSV and CIELab in input image.
        
        Parameters
        ----------
        image : opencv image
            Image to be analyzed.
        
        Returns
        -------
        features : tuple
            Returns a tuple containing a list of labels, type and values for each feature extracted.
        """
        image_hsv = ImageUtils.image_to_hsv(image, bgr = True)
        image_cielab = ImageUtils.image_to_cielab(image, bgr = True)

        b, g, r = cv2.split(image)
        h, s, v = cv2.split(image_hsv)
        ciel, ciea, cieb = cv2.split(image_cielab)
        
        labels = [
            'cor_rmin', 'cor_rmax', 'cor_rmediamedia', 'cor_rdesvio',
            'cor_gmin', 'cor_gmax', 'cor_gmedia', 'cor_gdesvio',
            'cor_bmin', 'cor_bmax', 'cor_bmedia', 'cor_bdesvio',
            'cor_hmin', 'cor_hmax', 'cor_hmedia', 'cor_hdesvio',
            'cor_smin', 'cor_smax', 'cor_smedia', 'cor_sdesvio',
            'cor_vmin', 'cor_vmax', 'cor_vmedia', 'cor_vdesvio',
            'cor_cielmin', 'cor_cielmax', 'cor_cielmedia', 'cor_cieldesvio',
            'cor_cieamin', 'cor_cieamax', 'cor_cieamedia', 'cor_cieadesvio',
            'cor_ciebmin', 'cor_ciebmax', 'cor_ciebmedia', 'cor_ciebdesvio'
        ]
        
        types = [Extractor.NUMERIC] * len(labels)

        values = [
            np.min(r), np.max(r), np.mean(r), np.std(r),
            np.min(g), np.max(g), np.mean(g), np.std(g),
            np.min(b), np.max(b), np.mean(b), np.std(b),
            np.min(h), np.max(h), np.mean(h), np.std(h),
            np.min(s), np.max(s), np.mean(s), np.std(s),
            np.min(v), np.max(v), np.mean(v), np.std(v),
            np.min(ciel), np.max(ciel), np.mean(ciel), np.std(ciel),
            np.min(ciea), np.max(ciea), np.mean(ciea), np.std(ciea),
            np.min(cieb), np.max(cieb), np.mean(cieb), np.std(cieb)
        ]

        return labels, types, values
