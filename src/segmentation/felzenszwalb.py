#!/usr/bin/python
# -*- coding: utf-8 -*-
#
"""
    Runs Segmenter Felzenszwalb's method implemented in skimage.segmentation.
    
    Efficient graph-based image segmentation, Felzenszwalb, P.F. and Huttenlocher, D.P. International Journal of Computer Vision, 2004
    
    Name: felzenszwalb.py
    Author: Alessandro dos Santos Ferreira ( santosferreira.alessandro@gmail.com )
"""

from skimage.segmentation import felzenszwalb
from collections import OrderedDict

from util.config import Config

from .segmenter import Segmenter
from .skimage_segmenter import SkimageSegmenter

class Felzenszwalb(Segmenter, SkimageSegmenter):
    """Run Felzenszwalb's method segmentation."""
    
    def __init__(self, scale = 200.0, sigma = 0.1, min_size = 50, border_color = 'Yellow', border_outline = 'No'):
        """Constructor.

        Parameters
        ----------
        scale : integer, float, default = 100.0
            Free parameter. Higher means larger clusters.
        sigma : float, optional, default = 1.0
            Width of Gaussian kernel used in preprocessing.
        min_size : integer, optional, default = 20
            Minimum component size. Enforced using postprocessing.
        border_color : string
            X11Color name of segment border color.
        border_outline : string
            If 'yes' double the size of segment border.
        """
        super(self.__class__, self).__init__(border_color, border_outline)
        
        self.scale = Config("Scale", scale, float)
        self.sigma = Config("Sigma", sigma, float)
        self.min_size = Config("Min Size", min_size, int)
        
    
    def get_config(self):
        """Return configuration of segmenter. 
        
        Returns
        -------
        config : OrderedDict
            Current configs of segmenter.
        """
        felzenszwalb_config = OrderedDict()
        
        felzenszwalb_config["scale"] = self.scale
        felzenszwalb_config["sigma"] = self.sigma
        felzenszwalb_config["min_size"] = self.min_size
        felzenszwalb_config["border_color"] = self.border_color
        felzenszwalb_config["border_outline"] = self.border_outline
        
        return felzenszwalb_config
        
    def set_config(self, configs):
        """Update configuration of segmenter. 
        
        Parameters
        ----------
        configs : OrderedDict
            New configs of segmenter.
        """
        self.scale = Config.nvl_config(configs["scale"], self.scale)
        self.sigma = Config.nvl_config(configs["sigma"], self.sigma)
        self.min_size = Config.nvl_config(configs["min_size"], self.min_size)
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
        felzenszwalb_config = OrderedDict()
        
        felzenszwalb_config[self.scale.label] = self.scale.value
        felzenszwalb_config[self.sigma.label] = self.sigma.value
        felzenszwalb_config[self.min_size.label] = self.min_size.value
        felzenszwalb_config[self.border_color.label] = self.border_color.value
        felzenszwalb_config[self.border_outline.label] = self.border_outline.value
        
        summary = ''
        for config in felzenszwalb_config:
            summary += "%s: %s\n" % (config, str(felzenszwalb_config[config]))
        
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
        args = { "scale": self.scale.get_cast_val(), 
                   "sigma": self.sigma.get_cast_val(), 
                   "min_size": self.min_size.get_cast_val()
                }

        return self.run_skimage(image, felzenszwalb, **args)


    def reset(self):
        """Clean all data of segmentation. 
        """
        return self.reset_skimage()
