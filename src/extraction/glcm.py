#!/usr/bin/python
# -*- coding: utf-8 -*-
#
"""
    Extract GLCM (Gray-Level Co-Occurrence Matrix) feature.
    
    Soh and Costas Tsatsoulis, Texture analysis of SAR sea ice imagery using gray level co-occurrence matrices, IEEE Transactions on geoscience and remote sensing, 1999.
    
    Name: glcm.py
    Author: Alessandro dos Santos Ferreira ( santosferreira.alessandro@gmail.com )
"""
import numpy as np
from skimage import feature

from util.utils import ImageUtils

from .extractor import Extractor

class GLCM(Extractor):
    """Implements GLCM (Gray-Level Co-Occurrence Matrix) feature extraction."""

    
    def __init__(self, glcm_levels = 256):
        self.glcm_levels = glcm_levels
        
    
    def run(self, image):
        """Extract the following texture properties defined in the GLCM: energy, contrast, correlation, homogeneity and dissimilarity,
        with distance 1 and 2 and angles 0, 45 and 90 degrees.
        
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

        g = feature.greycomatrix(image_grayscale, [1, 2], [0, np.pi / 4, np.pi / 2], self.glcm_levels, normed=True, symmetric=True)

        contrast = feature.greycoprops(g, 'contrast').tolist()
        dissimilarity = feature.greycoprops(g, 'dissimilarity').tolist()
        homogeneity = feature.greycoprops(g, 'homogeneity').tolist()
        asm = feature.greycoprops(g, 'ASM').tolist()
        energy = feature.greycoprops(g, 'energy').tolist()
        correlation = feature.greycoprops(g, 'correlation').tolist()

        labels = [
            'glcm_cont_1_0', 'glcm_cont_1_45', 'glcm_cont_1_90', 'glcm_cont_2_0', 'glcm_cont_2_45', 'glcm_cont_2_90',
            'glcm_diss_1_0', 'glcm_diss_1_45', 'glcm_diss_1_90', 'glcm_diss_2_0', 'glcm_diss_2_45', 'glcm_diss_2_90',
            'glcm_homo_1_0', 'glcm_homo_1_45', 'glcm_homo_1_90', 'glcm_homo_2_0', 'glcm_homo_2_45', 'glcm_homo_2_90',
            'glcm_asm_1_0', 'glcm_asm_1_45', 'glcm_asm_1_90', 'glcm_asm_2_0', 'glcm_asm_2_45', 'glcm_asm_2_90',
            'glcm_ener_1_0', 'glcm_ener_1_45', 'glcm_ener_1_90', 'glcm_ener_2_0', 'glcm_ener_2_45', 'glcm_ener_2_90',
            'glcm_corr_1_0', 'glcm_corr_1_45', 'glcm_corr_1_90', 'glcm_corr_2_0', 'glcm_corr_2_45', 'glcm_corr_2_90',

        ]
        types = [Extractor.NUMERIC] * len(labels)

        values = contrast[0] + contrast[1] + dissimilarity[0] + dissimilarity[1] + homogeneity[0] + \
                  homogeneity[1] + asm[0] + asm[1] + energy[0] + energy[1] + correlation[0] + correlation[1]

        return labels, types, values
        
