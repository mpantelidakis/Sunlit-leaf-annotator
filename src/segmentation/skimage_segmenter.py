#!/usr/bin/python
# -*- coding: utf-8 -*-
#
"""
    Abstract class for segmenters implemented in skimage.segmentation.
    
    Name: skimage_segmenter.py
    Author: Alessandro dos Santos Ferreira ( santosferreira.alessandro@gmail.com )
"""

import cv2
import numpy as np
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float, img_as_ubyte

from util.config import Config
from util.utils import TimeUtils
from util.x11_colors import X11Colors
from abc import ABCMeta, abstractmethod

import os


class SkimageSegmenter(object):
    """Abstract class for segmenters implemented in skimage.segmentation."""
    
    __metaclass__ = ABCMeta
    
    # flag FORCE_OPT highly increases speed of method paint_segment_skimage but performs a flawed painting
    FORCE_OPT = True
    
    def __init__(self, border_color = 'Yellow', border_outline = 'No'):
        """Constructor.

        Parameters
        ----------
        border_color : string
            X11Color name of segment border color.
        border_outline : string
            If 'yes' double the size of segment border.
        """
        self.border_color = Config("Border Color", border_color, 'color')
        self.border_outline = Config("Border Outline", border_outline, str)
        
        self._segments = None
        self._original_image = None
        
        
    def get_list_segments_skimage(self):
        """Return a list with segments after apply segmentation. 
        
        Returns
        -------
        segments : list
            List of segments of segmented image.
        """
        if self._segments is None:
            return []

        # print(np.unique(self._segments))
        return np.unique(self._segments)

    def get_segment_skimage(self, px = 0, py = 0, idx_segment = None, path_to_mask = None):
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
        # Check if segmentation was already performed
        if self._segments is None:
            return None, 0, -1, 0
        
        start_time = TimeUtils.get_time()
        
        # If idx_segment not passed, get the segment index using the points (px, py)
        if idx_segment is None:
            idx_segment = self._segments[py, px]
        
        # Create a mask, painting black all pixels outside of segment and white the pixels inside.
        mask_segment = np.zeros(self._original_image.shape[:2], dtype="uint8")
        mask_segment[self._segments == idx_segment] = 255

        minas_mask_segment = mask_segment = np.zeros(self._original_image.shape[:2], dtype="uint8")
        minas_mask_segment[self._segments == idx_segment] = 1
        # minas_idx_segment = idx_segment

        txt = np.loadtxt(path_to_mask)
        txt = np.add(txt, minas_mask_segment)
        np.savetxt(path_to_mask, txt, fmt='%d')

        print ("Modified mask: ", path_to_mask)

        size_segment = mask_segment[self._segments == idx_segment].size

        segment = self._original_image.copy()
        segment = cv2.bitwise_and(segment, segment, mask=mask_segment)

        # Get the countours around the segment
        contours, _  = cv2.findContours(mask_segment,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]

        m = -1
        max_contour = None
        for cnt in contours:
            if (len(cnt) > m):
                m = len(cnt)
                max_contour = cnt

        # Get the rectangle that encompasses the countour
        x,y,w,h = cv2.boundingRect(max_contour)
        segment = segment[y:y+h, x:x+w]

        end_time = TimeUtils.get_time()
        
        # Return the rectangle that encompasses the countour
        return segment, size_segment, idx_segment, (end_time - start_time)
        

    def paint_segment_skimage(self, image, color, px = 0, py = 0, idx_segment = [], border = True, clear = False):
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
        # Check if segmentation was already performed
        if self._segments is None:
            return image, 0
            
        start_time = TimeUtils.get_time()

        # If idx_segment not passed, get the segment index using the points (px, py)
        if len(idx_segment) == 0:
            idx_segment = [self._segments[py, px]]
        height, width, channels = self._original_image.shape
        
        # Create a mask, painting black all pixels outside of segments and white the pixels inside
        mask_segment = np.zeros(self._original_image.shape[:2], dtype="uint8")
        for idx in idx_segment:
            mask_segment[self._segments == idx] = 255
        mask_inv = cv2.bitwise_not(mask_segment)
            
        # Paint all pixels in original image with choosed color
        class_color = np.zeros((height,width,3), np.uint8)
        class_color[:, :] = X11Colors.get_color(color)
        if SkimageSegmenter.FORCE_OPT == False:
            colored_image = cv2.addWeighted(self._original_image, 0.7, class_color, 0.3, 0)
        else:
            colored_image = cv2.addWeighted(image, 0.7, class_color, 0.3, 0)
        colored_image = cv2.bitwise_and(colored_image, colored_image, mask=mask_segment)
        
        # Create a new image keeping the painting only in pixels inside of segments
        new_image = image if clear == False else self._original_image
        new_image = cv2.bitwise_and(new_image, new_image, mask=mask_inv)
        mask_segment[:] = 255
        new_image = cv2.bitwise_or(new_image, colored_image, mask=mask_segment)

        # If border is true, paint the segments borders
        if border == True and SkimageSegmenter.FORCE_OPT == False:
            color = X11Colors.get_color_zero_one(self.border_color.get_cast_val())
            outline_color = color if self.border_outline.value == 'Yes' else None
            
            new_image = img_as_ubyte( mark_boundaries(img_as_float(new_image), self._segments.astype(np.int8), color=color, outline_color=outline_color) ) 
            
        end_time = TimeUtils.get_time()
        
        # Return painted image
        return new_image, (end_time - start_time)


    def run_skimage(self, image, method, **kwargs):
        """Perform the segmentation 
        
        Parameters
        ----------
        image : opencv 3-channel color image.
            Original image.
        method : function
            Method from skyimage that performs the image segmentation.
        **kwargs : keyword arguments
            Dict of the keyword args passed to the function.
    
        Returns
        -------
        new_image : opencv 3-channel color image.
            Segmented image.
        run_time : integer
            Running time spent in milliseconds.
        """
        self._original_image = image

        # Run the segmentation using the method passed
        start_time = TimeUtils.get_time()
        self._segments = method(img_as_float(image), **kwargs)
        end_time = TimeUtils.get_time()
        
        color = X11Colors.get_color_zero_one(self.border_color.get_cast_val())
        outline_color = color if self.border_outline.value == 'Yes' else None

        #  Ignore UserWarning: Possible precision loss when converting from float64 to uint8
        #  because the returned image is used just for visualization
        #  The original image, without loss, is stored in self._original_image
        return img_as_ubyte( mark_boundaries(image, self._segments.astype(np.int8), color=color, outline_color=outline_color) ), (end_time - start_time)
        

    def reset_skimage(self):
        """Clean all data of segmentation. 
        """
        self._segments = None
        self._original_image = None
