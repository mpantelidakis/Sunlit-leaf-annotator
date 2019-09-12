#!/usr/bin/python
# -*- coding: utf-8 -*-
#
"""
    Runs Segmenter SLIC (Simple Linear Iterative Clustering) implemented in skimage.segmentation.
    
    Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi, Pascal Fua, and Sabine Suesstrunk, SLIC Superpixels Compared to State-of-the-art Superpixel Methods, TPAMI, May 2012.
    
    Name: slic.py
    Author: Alessandro dos Santos Ferreira ( santosferreira.alessandro@gmail.com )
"""

from skimage.segmentation import slic
from collections import OrderedDict

from util.config import Config
from util.utils import TimeUtils
from util.x11_colors import X11Colors

from .segmenter import Segmenter
from .skimage_segmenter import SkimageSegmenter

class Slic(Segmenter, SkimageSegmenter):
    """Run SLIC (Simple Linear Iterative Clustering) segmentation."""

    def __init__(self, n_segments = 35, sigma = 2.0, compactness = 10.0, border_color = 'Yellow', border_outline = 'No'):
        """Constructor.

        Parameters
        ----------
        n_segments : integer, optional, default = 250
            The (approximate) number of labels in the segmented output image.
        sigma : float, optional, default = 5.0
            Width of Gaussian smoothing kernel for pre-processing.
        compactness : float, optional, default = 10.0
            Higher values give more weight to space proximity, making superpixel shapes more square/cubic.
        border_color : string
            X11Color name of segment border color.
        border_outline : string
            If 'yes' double the size of segment border.
        """
        super(self.__class__, self).__init__(border_color, border_outline)
        
        self.n_segments = Config("Segments", n_segments, int)
        self.sigma = Config("Sigma", sigma, float)
        self.compactness = Config("Compactness", compactness, float)
        
    
    def get_config(self):
        """Return configuration of segmenter. 
        
        Returns
        -------
        config : OrderedDict
            Current configs of segmenter.
        """
        slic_config = OrderedDict()
        
        slic_config["n_segments"] = self.n_segments
        slic_config["sigma"] = self.sigma
        slic_config["compactness"] = self.compactness
        slic_config["border_color"] = self.border_color
        slic_config["border_outline"] = self.border_outline
        
        return slic_config
        
    def set_config(self, configs):
        """Update configuration of segmenter. 
        
        Parameters
        ----------
        configs : OrderedDict
            New configs of segmenter.
        """
        self.n_segments = Config.nvl_config(configs["n_segments"], self.n_segments)
        self.sigma = Config.nvl_config(configs["sigma"], self.sigma)
        self.compactness = Config.nvl_config(configs["compactness"], self.compactness)
        self.border_color = Config.nvl_config(configs["border_color"], self.border_color)
        self.border_outline = Config.nvl_config(configs["border_outline"], self.border_outline)
        
        self.border_outline.value = self.border_outline.value if self.border_outline.value == 'Yes' else 'No'

    def get_summary_config(self):
        """Return fomatted summary of configuration. 
        
        Returns
        -------
        summary : string
            Formatted string with summary of configuration.
        """
        slic_config = OrderedDict()
        
        slic_config[self.n_segments.label] = self.n_segments.value
        slic_config[self.sigma.label] = self.sigma.value
        slic_config[self.compactness.label] = self.compactness.value
        slic_config[self.border_color.label] = self.border_color.value
        slic_config[self.border_outline.label] = self.border_outline.value
        
        summary = ''
        for config in slic_config:
            summary += "%s: %s\n" % (config, str(slic_config[config]))
        
        return summary
    

    def get_list_segments(self):
        """Return a list with segments after apply segmentation. 
        
        Returns
        -------
        segments : list
            List of segments of segmented image.
        """
        return self.get_list_segments_skimage()

    def get_segment(self, px = 0, py = 0, idx_segment = None, path_to_mask = None):
        """Return a specified segment using a index or position in image. 
        
        Parameters
        ----------
        px : integer, optional, default = 0
            Segment point inside the image in x-axis.
        py : integer, optional, default = 0
            Segment point inside the image in y-axis.
        idx_segment : integer, optional, default = None
            Index of segment returned by previous call of this method.
            
        Returns
        -------
        segment : opencv 3-channel color image.
            Rectangle encompassing the segment image.
        size_segment : integer
            Number of pixels of segment.
        idx_segment : integer
            Index of segment if found, -1 otherwise.
        run_time : integer
            Running time spent in milliseconds.
        """
        return self.get_segment_skimage(px, py, idx_segment, path_to_mask)
    
    def paint_segment(self, image, color, px = 0, py = 0, idx_segment = [], border = True, clear = False):
        """Paint a list of segments using a index or position in image.
        
        Parameters
        ----------
        image : opencv 3-channel color image.
            Segmented image.
        color : string
            X11Color name.
        px : integer, optional, default = 0
            Segment point inside the image in x-axis.
        py : integer, optional, default = 0
            Segment point inside the image in y-axis.
        idx_segment : list, optional, default = []
            List of segments.
        border : boolean, optional, default = True
            If true paint the border of segments with default color.
        clear : boolean, optional, default = False
            If true clear previous painting in the image.

        Returns
        -------
        new_image : opencv 3-channel color image.
            Painted image.
        run_time : integer
            Running time spent in milliseconds.
        """
        return self.paint_segment_skimage(image, color, px, py, idx_segment, border, clear)


    def run(self, image):
        """Perform the segmentation 
        
        Parameters
        ----------
        image : opencv 3-channel color image.
            Original image.
            
        Returns
        -------
        new_image : opencv 3-channel color image.
            Segmented image.
        run_time : integer
            Running time spent in milliseconds.
        """
        args = { "n_segments": self.n_segments.get_cast_val(), 
                   "sigma": self.sigma.get_cast_val(), 
                   "compactness": self.compactness.get_cast_val()
                }

        return self.run_skimage(image, slic, **args)
        

    def reset(self):
        """Clean all data of segmentation. 
        """
        return self.reset_skimage()
