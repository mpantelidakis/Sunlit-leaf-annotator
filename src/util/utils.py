#!/usr/bin/python
# -*- coding: utf-8 -*-
#
"""
    Generic utilities classes.
    
    Name: main.py
    Author: Alessandro dos Santos Ferreira ( santosferreira.alessandro@gmail.com )
"""
import cv2
import random
import time
from sklearn.metrics import confusion_matrix
import numpy as np

class ColorUtils(object):
    """Set of utilities to manipulate colors."""

    @staticmethod
    def rand_color():
        """Return a random color.
        
        Returns
        -------
        color : string
            Color in format #FFFFFF.
        """
        r = random.randint
        
        return '#%02X%02X%02X' % (r(0, 255), r(0, 255), r(0, 255))
        

class ImageUtils(object):
    """Set of utilities to manipulate images."""
    
    @staticmethod
    def image_to_bgr(image):
        """Convert a image to BGR.
        
        Parameters
        ----------
        image : opencv RGB color image
            Image to be converted.
        
        Returns
        -------
        image : opencv BGR color image
            Image converted to BGR.
        """
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
    @staticmethod
    def image_to_hsv(image, bgr = False):
        """Convert a image to HSV.
        
        Parameters
        ----------
        image : opencv 3-channel color image
            Image to be converted.
        bgr : boolean, optional, default = False
            If true image is BGR, else image is RGB.
        
        Returns
        -------
        image : opencv HSV color image
            Image converted to HSV.
        """
        if bgr == False:
            image = ImageUtils.image_to_bgr(image)
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
    @staticmethod
    def image_to_cielab(image, bgr = False):
        """Convert a image to CIELab.
        
        Parameters
        ----------
        image : opencv 3-channel color image
            Image to be converted.
        bgr : boolean, optional, default = False
            If true image is BGR, else image is RGB.
        
        Returns
        -------
        image : opencv CIELab color image
            Image converted to CIELab.
        """
        if bgr == False:
            image = ImageUtils.image_to_bgr(image)
        return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
    @staticmethod
    def image_grayscale(image, bgr = False):
        """Convert a image to grayscale.
        
        Parameters
        ----------
        image : opencv 3-channel color image
            Image to be converted.
        bgr : boolean, optional, default = False
            If true image is BGR, else image is RGB.
        
        Returns
        -------
        image : opencv grayscale image
            Image converted to grayscale.
        """
        if bgr == False:
            image = ImageUtils.image_to_bgr(image)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    @staticmethod
    def image_binary(image, bgr = False, grayscale = False):
        """Convert a image to binary.
        
        Parameters
        ----------
        image : opencv 3-channel color image
            Image to be converted.
        bgr : boolean, optional, default = False
            If true image is BGR, else image is RGB.
        grayscale : boolean, optional, default = False
            If true image is grayscale, else image is 3-channel color.
        
        Returns
        -------
        image : opencv binary image
            Image converted to binary.
        """
        if grayscale == False:
            image = ImageUtils.image_grayscale(image, bgr)
        
        image = cv2.GaussianBlur(image,(5,5),0)
        _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return image
        
    @staticmethod
    def image_edge_detector(image, canny_min = 100, canny_max = 200, bgr = False, grayscale = False):
        """Detect the edges of image.
        
        Parameters
        ----------
        image : opencv 3-channel color image
            Source image to be analyzed.
        canny_min : integer, optional, default = 100
            First threshold for the hysteresis procedure.
        canny_max : integer, optional, default = 200
            Second threshold for the hysteresis procedure.
        bgr : boolean, optional, default = False
            If true image is BGR, else image is RGB.
        grayscale : boolean, optional, default = False
            If true image is grayscale, else image is 3-channel color.
        
        Returns
        -------
        image : opencv grayscale image
            Image with edges detected.
        """
        if grayscale == False:
            image = ImageUtils.image_grayscale(image, bgr)
        return cv2.Canny(image, canny_min, canny_max)
        
    @staticmethod
    def image_resize(image, width, height):
        """Resize the image.
        
        Parameters
        ----------
        image : opencv image
            Image to be resized.
        width : integer
            Width of image.
        height : integer
            Height of image.
        
        Returns
        -------
        image : opencv image
            Image resized.
        """
        return cv2.resize(image, (width, height))

        
class TimeUtils(object):
    """Set of utilities to manipulate time."""

    @staticmethod
    def get_time():
        """Return the current time.
        
        Returns
        -------
        time : float
            Returns the time as a floating point number expressed in seconds since the epoch, in UTC.
        """
        return time.time()


class MetricUtils(object):
    """Calculate segmentation metrics."""

    @staticmethod
    def confusion_matrix(eval_segm, gt_segm):
        return confusion_matrix(gt_segm.flatten(), eval_segm.flatten())

    @staticmethod
    def pixel_accuracy(eval_segm, gt_segm):
        """sum_i(n_ii) / sum_i(t_i)"""

        MetricUtils.check_size(eval_segm, gt_segm)

        cl, n_cl = MetricUtils.extract_classes(gt_segm)
        eval_mask, gt_mask = MetricUtils.extract_both_masks(eval_segm, gt_segm, cl, n_cl)

        sum_n_ii = 0
        sum_t_i  = 0

        for i, c in enumerate(cl):
            curr_eval_mask = eval_mask[i, :, :]
            curr_gt_mask = gt_mask[i, :, :]

            sum_n_ii += np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
            sum_t_i  += np.sum(curr_gt_mask)
 
        if (sum_t_i == 0):
            pixel_accuracy_ = 0
        else:
            pixel_accuracy_ = sum_n_ii / sum_t_i

        return pixel_accuracy_


    @staticmethod
    def mean_accuracy(eval_segm, gt_segm):
        """(1/n_cl) sum_i(n_ii/t_i)"""

        MetricUtils.check_size(eval_segm, gt_segm)

        cl, n_cl = MetricUtils.extract_classes(gt_segm)
        eval_mask, gt_mask = MetricUtils.extract_both_masks(eval_segm, gt_segm, cl, n_cl)

        accuracy = list([0]) * n_cl

        for i, c in enumerate(cl):
            curr_eval_mask = eval_mask[i, :, :]
            curr_gt_mask = gt_mask[i, :, :]

            n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
            t_i  = np.sum(curr_gt_mask)
     
            if (t_i != 0):
                accuracy[i] = n_ii / t_i

        mean_accuracy_ = np.mean(accuracy)
        return mean_accuracy_, accuracy


    @staticmethod
    def mean_IU(eval_segm, gt_segm):
        """(1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))"""

        MetricUtils.check_size(eval_segm, gt_segm)

        cl, n_cl   = MetricUtils.union_classes(eval_segm, gt_segm)
        _, n_cl_gt = MetricUtils.extract_classes(gt_segm)
        eval_mask, gt_mask = MetricUtils.extract_both_masks(eval_segm, gt_segm, cl, n_cl)

        IU = list([0]) * n_cl

        for i, c in enumerate(cl):
            curr_eval_mask = eval_mask[i, :, :]
            curr_gt_mask = gt_mask[i, :, :]
     
            if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
                continue

            n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
            t_i  = np.sum(curr_gt_mask)
            n_ij = np.sum(curr_eval_mask)

            IU[i] = n_ii / (t_i + n_ij - n_ii)
     
        mean_IU_ = np.sum(IU) / n_cl_gt
        return mean_IU_, IU

    @staticmethod
    def frequency_weighted_IU(eval_segm, gt_segm):
        """sum_k(t_k)^(-1) * sum_i((t_i*n_ii)/(t_i + sum_j(n_ji) - n_ii))"""

        MetricUtils.check_size(eval_segm, gt_segm)

        cl, n_cl = MetricUtils.union_classes(eval_segm, gt_segm)
        eval_mask, gt_mask = MetricUtils.extract_both_masks(eval_segm, gt_segm, cl, n_cl)

        frequency_weighted_IU_ = list([0]) * n_cl

        for i, c in enumerate(cl):
            curr_eval_mask = eval_mask[i, :, :]
            curr_gt_mask = gt_mask[i, :, :]
     
            if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
                continue

            n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
            t_i  = np.sum(curr_gt_mask)
            n_ij = np.sum(curr_eval_mask)

            frequency_weighted_IU_[i] = (t_i * n_ii) / (t_i + n_ij - n_ii)
     
        sum_k_t_k = MetricUtils.get_pixel_area(eval_segm)
        
        frequency_weighted_IU_ = np.sum(frequency_weighted_IU_) / sum_k_t_k
        return frequency_weighted_IU_


    @staticmethod
    def get_pixel_area(segm):
        """Auxiliary functions used during evaluation."""
        return segm.shape[0] * segm.shape[1]

    @staticmethod
    def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
        eval_mask = MetricUtils.extract_masks(eval_segm, cl, n_cl)
        gt_mask   = MetricUtils.extract_masks(gt_segm, cl, n_cl)

        return eval_mask, gt_mask

    @staticmethod
    def extract_classes(segm):
        cl = np.unique(segm)
        n_cl = len(cl)

        return cl, n_cl

    @staticmethod
    def union_classes(eval_segm, gt_segm):
        eval_cl, _ = MetricUtils.extract_classes(eval_segm)
        gt_cl, _   = MetricUtils.extract_classes(gt_segm)

        cl = np.union1d(eval_cl, gt_cl)
        n_cl = len(cl)

        return cl, n_cl

    @staticmethod
    def extract_masks(segm, cl, n_cl):
        h, w  = MetricUtils.segm_size(segm)
        masks = np.zeros((n_cl, h, w))

        for i, c in enumerate(cl):
            masks[i, :, :] = segm == c

        return masks

    @staticmethod
    def segm_size(segm):
        try:
            height = segm.shape[0]
            width  = segm.shape[1]
        except IndexError:
            raise

        return height, width

    @staticmethod
    def check_size(eval_segm, gt_segm):
        h_e, w_e = MetricUtils.segm_size(eval_segm)
        h_g, w_g = MetricUtils.segm_size(gt_segm)

        if (h_e != h_g) or (w_e != w_g):
            raise EvalSegErr("DiffDim: Different dimensions of matrices!")

class BooleanUtils(object):

    @staticmethod
    def str_to_bool(s):
        if s == 'True' or s == 1:
            return True
        elif s == 'False' or s == 0:
            return False
        else:
            raise ValueError 