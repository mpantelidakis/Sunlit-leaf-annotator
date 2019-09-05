#!/usr/bin/python
# -*- coding: utf-8 -*-
#
"""
    Provides static method utilities from Tkinter package.
    
    Name: tk_utils.py
    Author: Alessandro dos Santos Ferreira ( santosferreira.alessandro@gmail.com )
"""
import os
import sys
if sys.version_info[0] < 3:
    import tkMessageBox, Tkconstants, tkFileDialog
    import tkColorChooser
else:
    import tkinter.messagebox as tkMessageBox
    import tkinter.constants as Tkconstants
    from tkinter import filedialog as  tkFileDialog
    from tkinter import colorchooser as tkColorChooser




class Utils(object):
    """Utilities functionalities from Tkinter."""
    
    image_dir = '../data/'
    image_file = 'demo.jpg'
    
    default_directory = '../data/demo'
    
    @staticmethod
    def ask_image_name(title = 'Open a image'):
        """Shows a default Tkinter dialogue to open a image.

        Parameters
        ----------
        title : string, optional, default = 'Open a image'
            Dialogue title.
        """
        file_opt = options = {}
        options['defaultextension'] = '.jpg'
        options['filetypes'] = [('All supported files', ("*.BMP", "*.bmp", "*.JPG", "*.jpg", "*.JPEG", "*.jpeg", "*.PNG", "*.png", "*.TIF", "*.tif")),
                                ('BMP', ("*.BMP", "*.bmp")),
                                ('JPEG', ("*.JPG", "*.jpg", "*.JPEG", "*.jpeg")),
                                ('PNG', ("*.PNG", "*.png")),
                                ('TIF', ("*.TIF", "*.tif"))]
        options['initialdir'] = Utils.image_dir
        options['initialfile'] = Utils.image_file
        #options['parent'] = self.__root
        options['title'] = title
        
        filename = tkFileDialog.askopenfilename(**file_opt)
        if filename:
            Utils.image_dir, Utils.image_file = os.path.split(filename)
        
        return filename

    @staticmethod
    def ask_weight_name(title = 'Open a weight'):
        """Shows a default Tkinter dialogue to open a image.

        Parameters
        ----------
        title : string, optional, default = 'Open a image'
            Dialogue title.
        """
        file_opt = options = {}
        options['defaultextension'] = '.jpg'
        options['filetypes'] = [('All supported files', ("*.h5")),
                                ('H5 weight', ("*.h5", "*.H5"))]
        options['initialdir'] = Utils.image_dir
        options['title'] = title
        
        filename = tkFileDialog.askopenfilename(**file_opt)
        if filename:
            Utils.image_dir, Utils.image_file = os.path.split(filename)
        
        return filename


    @staticmethod
    def ask_directory(title = 'Choose a directory', default_dir = None):
        """Shows a default Tkinter dialogue to choose a directory.

        Parameters
        ----------
        title : string, optional, default = 'Choose a directory'
            Dialogue title.
        default_dir : string, optional, default = None
            Path to default directory.
        """
        dir_opt = options = {}
        options['initialdir'] = Utils.default_directory if default_dir is None else default_dir
        options['mustexist'] = False
        #options['parent'] = root
        options['title'] = title
        
        directory = tkFileDialog.askdirectory(**dir_opt)
        if directory:
            Utils.default_directory = directory
        
        return directory
        
        
    @staticmethod
    def ask_color_choose(title = 'Choose a color', default_color = 'white'):
        """Shows a default Tkinter dialogue to choose a color.

        Parameters
        ----------
        title : string, optional, default = 'Choose a color'
            Dialogue title.
        default_color : string, optional, default = None
            Default color.
        """
        return tkColorChooser.askcolor(title=title, initialcolor=default_color)
