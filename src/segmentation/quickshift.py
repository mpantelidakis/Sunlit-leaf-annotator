#!/usr/bin/python
# -*- coding: utf-8 -*-
#
"""
    Runs Segmenter Quickshift implemented in skimage.segmentation.
    
    Quick shift and kernel methods for mode seeking, Vedaldi, A. and Soatto, S. European Conference on Computer Vision, 2008
    
    Name: quickshift.py
    Author: Alessandro dos Santos Ferreira ( santosferreira.alessandro@gmail.com )
"""

from skimage.segmentation import quickshift
from collections import OrderedDict

from util.config import Config

from .segmenter import Segmenter
from .skimage_segmenter import SkimageSegmenter

class Quickshift(Segmenter, SkimageSegmenter):
    """Run Quickshift segmentation."""

    def __init__(self, ratio = 0.1, kernel_size = 2.1, max_dist = 15.0, border_color = 'Yellow', border_outline = 'No'):
        """Constructor.

        Parameters
        ----------
        ratio : float, optional, between 0 and 1, default = 0.5
            Balances color-space proximity and image-space proximity. Higher values give more weight to color-space.
        kernel_size : float, optional, default = 2.0
            Width of Gaussian kernel used in smoothing the sample density. Higher means fewer clusters.
        max_dist : float, optional, default = 10.0
            Cut-off point for data distances. Higher means fewer clusters.
        border_color : string
            X11Color name of segment border color.
        border_outline : string
            If 'yes' double the size of segment border.
        """
        super(self.__class__, self).__init__(border_color, border_outline)
        
        self.ratio = Config("Ratio [0-1]", ratio, float)
        self.kernel_size = Config("Kernel Size", kernel_size, float)
        #self.sigma = Config("Sigma", sigma, float)
        self.max_dist = Config("Max Dist", max_dist, float)
        
    
    def get_config(self):
        """Return configuration of segmenter. 
        
        Returns
        -------
        config : OrderedDict
            Current configs of segmenter.
        """
        quickshift_config = OrderedDict()
        
        quickshift_config["ratio"] = self.ratio
        quickshift_config["kernel_size"] = self.kernel_size
        #quickshift_config["sigma"] = self.sigma
        quickshift_config["max_dist"] = self.max_dist
        quickshift_config["border_color"] = self.border_color
        quickshift_config["border_outline"] = self.border_outline
        
        return quickshift_config
        
    def set_config(self, configs):
        """Update configuration of segmenter. 
        
        Parameters
        ----------
        configs : OrderedDict
            New configs of segmenter.
        """
        self.ratio = Config.nvl_config(configs["ratio"], self.ratio)
        self.kernel_size = Config.nvl_config(configs["kernel_size"], self.kernel_size)
        #self.sigma = Config.nvl_config(configs["sigma"], self.sigma)
        self.max_dist = Config.nvl_config(configs["max_dist"], self.max_dist)
        self.border_color = Config.nvl_config(configs["border_color"], self.border_color)
        self.border_outline = Config.nvl_config(configs["border_outline"], self.border_outline)
        
        self.ratio.value = self.ratio.value if self.ratio.value <= 1 else 1
        self.border_outline.value = self.border_outline.value if self.border_outline.value == 'Yes' else 'No'

    def get_summary_config(self):
        """Return fomatted summary of configuration. 
        
        Returns
        -------
        summary : string
            Formatted string with summary of configuration.
        """
        quickshift_config = OrderedDict()
        
        quickshift_config[self.ratio.label] = self.ratio.value
        quickshift_config[self.kernel_size.label] = self.kernel_size.value
        #quickshift_config[self.sigma.label] = self.sigma.value
        quickshift_config[self.max_dist.label] = self.max_dist.value
        quickshift_config[self.border_color.label] = self.border_color.value
        quickshift_config[self.border_outline.label] = self.border_outline.value
        
        summary = ''
        for config in quickshift_config:
            summary += "%s: %s\n" % (config, str(quickshift_config[config]))
        
        return summary


    def get_list_segments(self):
        """Return a list with segments after apply segmentation. 
        
        Returns
        -------
        segments : list
            List of segments of segmented image.
        """
        return self.get_list_segments_skimage()
        
    def get_segment(self, px = 0, py = 0, idx_segment = None, path_to_mask = None, color = None):
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
        return self.get_segment_skimage(px, py, idx_segment, path_to_mask, color)
    
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
        args = { "ratio": self.ratio.get_cast_val(), 
                   "kernel_size": self.kernel_size.get_cast_val(), 
                   #"sigma": self.sigma.get_cast_val(), 
                   "max_dist": self.max_dist.get_cast_val()
                }

        return self.run_skimage(image, quickshift, **args)

        
    def reset(self):
        """Clean all data of segmentation. 
        """
        return self.reset_skimage()
