#!/usr/bin/python
# -*- coding: utf-8 -*-
#
"""
    Provides structured graphics interface from Tkinter package.
    
    Name: tk_canvas.py
    Author: Alessandro dos Santos Ferreira ( santosferreira.alessandro@gmail.com )
"""

import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk
    
import matplotlib
matplotlib.use('TkAgg')

from matplotlib import pylab as plt
import sys
if sys.version_info[0] < 3:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
else:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class Image(object):
    """Class to manipulate structured graphics interface."""
    
    
    def __init__(self, parent):
        """Constructor.

        Parameters
        ----------
        parent : Tk widget
            Parent widget of this class.
        """
        self.parent = parent
        
        self._im = None
        self._canvas = None
        self._fig = None
        self._ax = None
        self._toolbar = None
        
        self._axes_visible = True
        self._toolbar_visible = True
        
            
    def toggle_axes(self):
        """Toogle the axes of image.
        """
        self._axes_visible = not self._axes_visible

        if self._canvas is not None:
            self._ax.get_xaxis().set_visible(self._axes_visible)
            self._ax.get_yaxis().set_visible(self._axes_visible)
            
            self._fig.tight_layout()
            self._canvas.draw()
            
    def toggle_toolbar(self):
        """Toogle the matplotlib image toolbar.
        """
        self._toolbar_visible = not self._toolbar_visible
        
        self._show_toolbar() if self._toolbar_visible else self._hide_toolbar()
    
        
    def render(self, image, onclick = None):
        """Render a image in window GUI.

        Parameters
        ----------
        image : opencv 3-channel color image
            OpenCV Image.
        onclick : function, optional, default = None
            Callback to be executed on image click.
        """
        self.parent.image = image
            
        self._fig = plt.figure(facecolor='white', edgecolor='black', linewidth=1)
        self._im = plt.imshow(self.parent.image) # later use a.set_data(new_data)
        
        self._ax = plt.gca()
        self._ax.get_xaxis().set_visible(self._axes_visible)
        self._ax.get_yaxis().set_visible(self._axes_visible)

        # a tk.DrawingArea
        self._fig.tight_layout()
        self._canvas = FigureCanvasTkAgg(self._fig, master=self.parent)
        if sys.version_info[0] < 3:
            self._canvas.show()
        else:
            self._canvas.draw()

        self._canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        
        if onclick is not None:
            self._fig.canvas.mpl_connect('button_press_event', func = onclick)
            
        if self._toolbar_visible == True:
            self._show_toolbar()
            
    
    def refresh(self, image = None):
        """Refresh the image content.

        Parameters
        ----------
        image : opencv 3-channel color image, optional, default = None
            OpenCV Image.
        """
        if self._canvas is not None:
            if image is not None:
                self.parent.image = image
            self._im.set_data(self.parent.image)
            
            self._fig.tight_layout()
            self._canvas.draw()

    def close(self):
        """Close the image.
        """
        if self._canvas is not None:
            self.parent.image = None
            #self._im.set_data(np.zeros((0,0,3), float))
            #self._canvas.draw()
            self._canvas.get_tk_widget().pack_forget()
            self._canvas.get_tk_widget().destroy()

            if self._toolbar is not None:
                self._toolbar.pack_forget()
                self._toolbar.destroy()

            # The command plt.clf() is used to minimize the effects of follow warning:
            # "RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface 
            # (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory."
            # However this part of code need to be refactored to prevent the memory leak!!!
            plt.clf()
            
            self._im = None
            self._canvas = None
            self._fig = None
            self._ax = None
            self._toolbar = None
            
            return True
        return False

        
    def _show_toolbar(self):
        """Show the matplotlib image toolbar.
        """
        if self._toolbar is None and self._canvas is not None:
            if sys.version_info[0] < 3:
                self._toolbar = NavigationToolbar2TkAgg(self._canvas, self.parent)
            else:
                self._toolbar = NavigationToolbar2Tk(self._canvas, self.parent)
            self._toolbar.configure(background='white', borderwidth=0)
            for child in self._toolbar.winfo_children():
                if sys.version_info[0] < 3:
                    #child.configure(background='white', foreground='black')
                    child.configure(background='white')
                else:
                    child.configure(background='white')
            #self._toolbar.update()
            self._toolbar.pack(side=Tk.TOP, fill=Tk.X, expand=False)
            
    def _hide_toolbar(self):
        """Hide the matplotlib image toolbar.
        """
        if self._toolbar is not None:
            self._toolbar.pack_forget()
            self._toolbar.destroy()
            self._toolbar = None
