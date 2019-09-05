#!/usr/bin/python
# -*- coding: utf-8 -*-
#
"""
    Tools for manipulating X11 colors.
    
    Name: x11_colors.py
    Author: Alessandro dos Santos Ferreira ( santosferreira.alessandro@gmail.com )
"""

import random

# Dictionary of available colors
_x11_colors = {    
    "AliceBlue" : (240, 248, 255),
    "AntiqueWhite" : (250, 235, 215),
    "Aqua" : (0, 255, 255),
    "Aquamarine" : (127, 255, 212),
    "Azure" : (240, 255, 255),
    "Beige" : (245, 245, 220),
    "Bisque" : (255, 228, 196),
    "Black" : (0, 0, 0),
    "BlanchedAlmond" : (255, 235, 205),
    "Blue" : (0, 0, 255),
    "BlueViolet" : (138, 43, 226),
    "Brown" : (165, 42, 42),
    "Burlywood" : (222, 184, 135),
    "CadetBlue" : (95, 158, 160),
    "Chartreuse" : (127, 255, 0),
    "Chocolate" : (210, 105, 30),
    "Coral" : (255, 127, 80),
    "Cornflower" : (100, 149, 237),
    "Cornsilk" : (255, 248, 220),
    "Crimson" : (220, 20, 60),
    "Cyan" : (0, 255, 255),
    "DarkBlue" : (0, 0, 139),
    "DarkCyan" : (0, 139, 139),
    "DarkGoldenrod" : (184, 134, 11),
    "DarkGray" : (169, 169, 169),
    "DarkGreen" : (0, 100, 0),
    "DarkKhaki" : (189, 183, 107),
    "DarkMagenta" : (139, 0, 139),
    "DarkOliveGreen" : (85, 107, 47),
    "DarkOrange" : (255, 140, 0),
    "DarkOrchid" : (153, 50, 204),
    "DarkRed" : (139, 0, 0),
    "DarkSalmon" : (233, 150, 122),
    "DarkSeaGreen" : (143, 188, 143),
    "DarkSlateBlue" : (72, 61, 139),
    "DarkSlateGray" : (47, 79, 79),
    "DarkTurquoise" : (0, 206, 209),
    "DarkViolet" : (148, 0, 211),
    "DeepPink" : (255, 20, 147),
    "DeepSkyBlue" : (0, 191, 255),
    "DimGray" : (105, 105, 105),
    "DodgerBlue" : (30, 144, 255),
    "Firebrick" : (178, 34, 34),
    "FloralWhite" : (255, 250, 240),
    "ForestGreen" : (34, 139, 34),
    "Fuchsia" : (255, 0, 255),
    "Gainsboro" : (220, 220, 220),
    "GhostWhite" : (248, 248, 255),
    "Gold" : (255, 215, 0),
    "Goldenrod" : (218, 165, 32),
    "Gray" : (190, 190, 190),
    "Green" : (0, 255, 0),
    "GreenYellow" : (173, 255, 47),
    "Honeydew" : (240, 255, 240),
    "HotPink" : (255, 105, 180),
    "IndianRed" : (205, 92, 92),
    "Indigo" : (75, 0, 130),
    "Ivory" : (255, 255, 240),
    "Khaki" : (240, 230, 140),
    "Lavender" : (230, 230, 250),
    "LavenderBlush" : (255, 240, 245),
    "LawnGreen" : (124, 252, 0),
    "LemonChiffon" : (255, 250, 205),
    "LightBlue" : (173, 216, 230),
    "LightCoral" : (240, 128, 128),
    "LightCyan" : (224, 255, 255),
    "LightGoldenrod" : (250, 250, 210),
    "LightGray" : (211, 211, 211),
    "LightGreen" : (144, 238, 144),
    "LightPink" : (255, 182, 193),
    "LightSalmon" : (255, 160, 122),
    "LightSeaGreen" : (32, 178, 170),
    "LightSkyBlue" : (135, 206, 250),
    "LightSlateGray" : (119, 136, 153),
    "LightSteelBlue" : (176, 196, 222),
    "LightYellow" : (255, 255, 224),
    "Lime" : (0, 255, 0),
    "LimeGreen" : (50, 205, 50),
    "Linen" : (250, 240, 230),
    "Magenta" : (255, 0, 255),
    "Maroon" : (176, 48, 96),
    "MediumAquamarine" : (102, 205, 170),
    "MediumBlue" : (0, 0, 205),
    "MediumOrchid" : (186, 85, 211),
    "MediumPurple" : (147, 112, 219),
    "MediumSeaGreen" : (60, 179, 113),
    "MediumSlateBlue" : (123, 104, 238),
    "MediumSpringGreen" : (0, 250, 154),
    "MediumTurquoise" : (72, 209, 204),
    "MediumVioletRed" : (199, 21, 133),
    "MidnightBlue" : (25, 25, 112),
    "MintCream" : (245, 255, 250),
    "MistyRose" : (255, 228, 225),
    "Moccasin" : (255, 228, 181),
    "NavajoWhite" : (255, 222, 173),
    "NavyBlue" : (0, 0, 128),
    "OldLace" : (253, 245, 230),
    "Olive" : (128, 128, 0),
    "OliveDrab" : (107, 142, 35),
    "Orange" : (255, 165, 0),
    "OrangeRed" : (255, 69, 0),
    "Orchid" : (218, 112, 214),
    "PaleGoldenrod" : (238, 232, 170),
    "PaleGreen" : (152, 251, 152),
    "PaleTurquoise" : (175, 238, 238),
    "PaleVioletRed" : (219, 112, 147),
    "PapayaWhip" : (255, 239, 213),
    "PeachPuff" : (255, 218, 185),
    "Peru" : (205, 133, 63),
    "Pink" : (255, 192, 203),
    "Plum" : (221, 160, 221),
    "PowderBlue" : (176, 224, 230),
    "Purple" : (160, 32, 240),
    "RebeccaPurple" : (102, 51, 153),
    "Red" : (255, 0, 0),
    "RosyBrown" : (188, 143, 143),
    "RoyalBlue" : (65, 105, 225),
    "SaddleBrown" : (139, 69, 19),
    "Salmon" : (250, 128, 114),
    "SandyBrown" : (244, 164, 96),
    "SeaGreen" : (46, 139, 87),
    "Seashell" : (255, 245, 238),
    "Sienna" : (160, 82, 45),
    "Silver" : (192, 192, 192),
    "SkyBlue" : (135, 206, 235),
    "SlateBlue" : (106, 90, 205),
    "SlateGray" : (112, 128, 144),
    "Snow" : (255, 250, 250),
    "SpringGreen" : (0, 255, 127),
    "SteelBlue" : (70, 130, 180),
    "Tan" : (210, 180, 140),
    "Teal" : (0, 128, 128),
    "Thistle" : (216, 191, 216),
    "Tomato" : (255, 99, 71),
    "Turquoise" : (64, 224, 208),
    "Violet" : (238, 130, 238),
    "WebGray" : (128, 128, 128),
    "WebGreen" : (0, 128, 0),
    "WebPurple" : (127, 0, 127),
    "Wheat" : (245, 222, 179),
    "White" : (255, 255, 255),
    "WhiteSmoke" : (245, 245, 245),
    "Yellow" : (255, 255, 0),
    "YellowGreen" : (154, 205, 50)

}

class X11Colors(object): 
    
    @staticmethod
    def search_color(name):
        """Search index of color by name.
        
        Parameters
        ----------
        name : string
            X11 name of color. No case sensitive.
        
        Returns
        -------
        idx : integer
            Index of color in dictionary _x11_colors.
            
        Raises
        ------
        Exception 'Invalid color'
            Incorrect color name.
        """
        color = name.upper().strip()
        colors = [c.upper() for c in _x11_colors]
        
        if color in colors:
            return colors.index(color)
        
        raise Exception('Invalid color')
    
    
    @staticmethod
    def get_color(name):
        """Get color by name.
        
        Parameters
        ----------
        name : string
            X11 name of color. No case sensitive.
        
        Returns
        -------
        image : tuple
            Tuple containing the color in RGB format: red -> (255, 0, 0).
        """
        idx = X11Colors.search_color(name)
        
        return list(_x11_colors.values())[idx]

        
    @staticmethod
    def get_color_bgr(name):
        """Get color by name and return in BGR format.
        
        Parameters
        ----------
        name : string
            X11 name of color. No case sensitive.
        
        Returns
        -------
        image : tuple
            Tuple containing the color in BGR format: red -> (0, 0, 255).
        """
        return X11Colors.rgb_to_bgr( X11Colors.get_color(name) )
        
    @staticmethod
    def get_color_hex(name):
        """Get color by name and return in hex format.
        
        Parameters
        ----------
        name : string
            X11 name of color. No case sensitive.
        
        Returns
        -------
        image : string
            Tuple containing the color in hex format: red -> '#FF0000'.
        """
        return X11Colors.to_hex( X11Colors.get_color(name) )
        
    @staticmethod
    def get_color_zero_one(name):
        """Get color by name and return in 0-1 interval.
        
        Parameters
        ----------
        name : string
            X11 name of color. No case sensitive.
        
        Returns
        -------
        image : tuple
            Tuple containing the color in 0-1 interval: red -> (1.0, 0.0, 0.0).
        """
        return X11Colors.to_zero_one( X11Colors.get_color(name) )


    @staticmethod
    def bgr_to_rgb(color):
        """Convert color from BGR to RGB.
        """
        return ( color[2], color[1], color[0])
        
    @staticmethod
    def rgb_to_bgr(color):
        """Convert color from RGB to BGR.
        """
        return ( color[2], color[1], color[0])
        
    @staticmethod
    def to_hex(color):
        """Convert color to hex format.
        """
        return '#%02X%02X%02X' % ( color[0], color[1], color[2])
        
    @staticmethod
    def to_zero_one(color):
        """Convert color to 0-1 interval.
        """
        return ( color[0]/255.0, color[1]/255.0, color[2]/255.0)


    @staticmethod
    def random_color_not_in(names):
        """Choose a color at random, not present in names.
        
        Parameters
        ----------
        names : list of string
            List of X11 color names.
        
        Returns
        -------
        color : string
            Name of color chosen at random.
        """
        ncolors = len(_x11_colors)
        xcolorlist=list(_x11_colors.keys())
        for i in range(0, ncolors):
            index = random.randint(0, ncolors-1)

            name = xcolorlist[index]
            if name not in names:
                return name
        
        return xcolorlist[0]

    @staticmethod
    def random_color():
        """Choose a color at random.
        
        Returns
        -------
        color : string
            Name of color chosen at random.
        """
        empty = []
        return X11Colors.random_color_not_in(empty)
        
    @staticmethod
    def random_hex_color():
        """Choose a color at random and convert to hex.
        
        Returns
        -------
        color : string
            Tuple containing the color in hex format: red -> '#FF0000'.
        """
        return X11Colors.get_color_hex( X11Colors.random_color() )

