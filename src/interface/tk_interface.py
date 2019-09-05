#!/usr/bin/python
# -*- coding: utf-8 -*-
#
"""
    Implements graphical interface using Tkinter package modules.
    
    Name: tk_interface.py
    Author: Alessandro dos Santos Ferreira ( santosferreira.alessandro@gmail.com )
"""

import sys
if sys.version_info < (3, 0):
    # for Python2
    import Tkinter as Tk
    import tkMessageBox
    import tk as tk_local
    from interface import Interface, InterfaceException as IException
else:
    # for Python3
    import tkinter as Tk
    import tkinter.messagebox as tkMessageBox
    from .tk import tk_utils,tk_popup,tk_customframe,tk_canvas,tk_logger,tk_customdialog,tk_menu
    class TkLocal:
        def __init__(self,Utils, Menu,Image,Log,Pop, CustomGrid,SelectDialog,ChooseOneDialog,ConfigDialog):
            self.Utils=Utils
            self.Menu=Menu
            self.Image=Image
            self.Log=Log
            self.Pop=Pop
            self.CustomGrid=CustomGrid
            self.SelectDialog=SelectDialog
            self.ChooseOneDialog=ChooseOneDialog
            self.ConfigDialog=ConfigDialog
    tk_local=TkLocal
    tk_local.Utils=tk_utils.Utils
    tk_local.Menu=tk_menu.Menu
    tk_local.Image=tk_canvas.Image
    tk_local.Log=tk_logger.Log
    tk_local.Pop=tk_popup.Popup
    tk_local.CustomGrid=tk_customframe.CustomGrid
    tk_local.SelectDialog=tk_customdialog.SelectDialog
    tk_local.ChooseOneDialog=tk_customdialog.ChooseOneDialog
    tk_local.ConfigDialog=tk_customdialog.ConfigDialog

    from interface import Interface, InterfaceException as IException



class TkInterface(Interface):
    """Implements graphical interface and functionalities using Tkinter."""
    
    MAX_CLASSES = 200

    if sys.version_info < (3, 0):
        utils = tk_local.Utils
    else:
        utils = tk_local.Utils

    def __init__(self, title):
        """Constructor.

        Parameters
        ----------
        title : string
            Title of interface window.
        """
        self.title = title
        
        self._root = Tk.Tk()
        self._root.wm_title(self.title)
        self._root.geometry('%dx%d+%d+%d' % (800, 800, 0, 0))
        
        self._menus = []
        
        self._image = tk_local.Image(self._root)
        self._grid = None
        self._logger = None
        self._conf_dialog = None
        
    def set_subtitle(self, subtitle):
        """Set complement of window title.

        Parameters
        ----------
        subtitle : string
            Complement of window title.
        """
        self._root.wm_title(self.title + ' - ' + subtitle)
    

    def add_menu(self, label):
        """Add a menu.

        Parameters
        ----------
        label : string
            Menu label.
        """
        self._menus.append( tk_local.Menu(self._root, label) )
    
    def add_command(self, label, action, shortcut = None):
        """Add a menu option to last menu added.

        Parameters
        ----------
        label : string
            Menu option label.
        action : function
            Callback to be executed on menu option click.
        shortcut : string
            Menu option shortcut number ou letter. Not case sensitive.
        """
        self._menus[-1].add_command( label = label, action = lambda *_: self._apply( action ), shortcut = shortcut )

    def add_separator(self):
        """Add a separator to last menu added.
        """
        self._menus[-1].add_separator( )
        
    def add_check_button(self, label, action, shortcut = None, default_state = True):
        """Add a check button to last menu added.

        Parameters
        ----------
        label : string
            Check button option label.
        action : function
            Callback to be executed on check button click.
        shortcut : string, optional, default = None
            Check button shortcut number ou letter. Not case sensitive.
        default_state : boolean, optional, default = True
            Initial state of check button. If true set to on.
        """
        self._menus[-1].add_check_button(label = label, action = lambda *_: self._apply( action ), shortcut = shortcut, default_state = default_state)

    def render_menu(self):
        """Add to window GUI the last menu added.
        """
        menubar = Tk.Menu(self._root)

        for menu in self._menus:
            menu.render(menubar)
          
        self._root.config(menu=menubar)
        
    
    def add_image(self, image, title = None, onclick = None):
        """Add a image to window GUI.

        Parameters
        ----------
        image : opencv 3-channel color image
            OpenCV Image.
        title : string
            Title of image. It will be concatenate to window title.
        onclick : function
            Callback to be executed on image click.
        """
        if self._image is not None:
            self._image.close()
           
        self._image.render(image, lambda event, *_: self._apply( onclick, event ))
            
        if title is not None:
            self.set_subtitle(title)
            
    def toggle_image_toolbar(self):
        """Toogle the matplotlib image toolbar.
        """
        self._image.toggle_toolbar()
            
    def toggle_image_axes(self):
        """Toogle the axes of image.
        """
        self._image.toggle_axes()

    def refresh_image(self, image = None, title = None):
        """Refresh the image content.

        Parameters
        ----------
        image : opencv 3-channel color image
            OpenCV Image.
        title : string
            Title of image. It will be concatenate to window title.
        """
        self._image.refresh(image)
            
        if title is not None:
            self.set_subtitle(title)

    def close_image(self):
        """Close the image.
        """
        if tkMessageBox.askokcancel("Quit", "Do you want to close the image?"):
            return self._image.close()


    def open_log(self):
        """Add a console log to window GUI.
        """
        self._logger = tk_local.Log(self._root)
        
    def write_log(self, fmt, *args):
        """Log a new formatted message.
        
        Parameters
        ----------
        fmt : string
            Message with format variables.
        *args : arguments
            List of arguments of message.
        """
        if self._logger:
            self._logger.write_logger(fmt, *args)
            self.refresh_image()
        
    def append_log(self, fmt, *args):
        """Append a formatted message to log.
        
        Parameters
        ----------
        fmt : string
            Message with format variables.
        *args : arguments
            List of arguments of message.
        """
        if self._logger:
            self._logger.append_logger(fmt, *args)
            self.refresh_image()
        
    def clear_log(self):
        """Clear log content.
        """
        if self._logger:
            self._logger.clear_logger()
            
    def toggle_log(self):
        """Toogle console log.
        """
        if self._logger:
            self.close_log()
        else:
            self.open_log()
        
    def close_log(self):
        """Close console log.
        """
        if self._logger:
            self._logger.destroy_logger()
            self._logger = None

        
    def add_panel(self, left = True):
        """Add a vertical panel to window GUI.
        
        Parameters
        ----------
        left : boolean, optional, default True
            Side where the panel will be placed. If true left, otherwise right.
        """
        self._grid = tk_local.CustomGrid(self._root, 175)
        
        side = Tk.LEFT if left == True else Tk.RIGHT
        self._grid.pack(side=side, fill=Tk.Y, expand=False)
        
    def add_panel_classes(self, classes = None, selected = 0):
        """Add a vertical panel with classes to window GUI.
        
        Parameters
        ----------
        classes : list of dictionary class_config, optional, default = None
            List of panel classes.
        selected : integer, optional, default = 0
            Index in list of selected class.
        """
        if self._grid is None:
            self._grid = tk_local.CustomGrid(self._root)
            self.clear_panel_classes()
        
        classes = classes if classes is not None else []

        self._grid.add_cell_label("", 0, 0, 1, 4)
        self._grid.add_cell_label("Classes   ", 0, 1, 12, 4)
        self._grid.add_cell_label("", 0, 3, 2, 4)
        
        length = len(classes)
        for i in range(1, length+1):
            self._grid.add_cell_radio_button(classes[i-1]["name"].value[:12], i, i, 1, 12, selected=True if i == selected+1 else False, 
                                                command=classes[i-1]["callback"].value, command_args=classes[i-1]["args"].value )
            self._grid.add_cell_button_color("", i, 2, 2, bg=classes[i-1]["color"].value, 
                                                command=classes[i-1]["callback_color"].value, command_args=classes[i-1]["args"].value )
            
        self._grid.pack(side=Tk.LEFT, fill=Tk.Y, expand=False)
        
    def refresh_panel_classes(self, classes = None, selected = 0):
        """Update the panel classes.
        
        Parameters
        ----------
        classes : list of dictionary class_config, optional, default = None
            List of panel classes.
        selected : integer, optional, default = 0
            Index in list of selected class.
        """
        self.clear_panel_classes()
        self.add_panel_classes(classes, selected)
        
    def clear_panel_classes(self):
        """Remove all classes from panel.
        """
        for i in range(1, self.MAX_CLASSES+1):
            self._grid.add_cell_label("", i, 1, 15)
            self._grid.add_cell_label("", i, 2, 2)

    def close_panel(self):
        """Remove the panel from window GUI.
        """
        if self._grid is not None:
            self._grid.pack_forget();
            self._grid.destroy();
        
        
    def popup(self, message):
        """Open a non-blocking popup.
        
        Parameters
        ----------
        message : string
            Message to be shown in popup.
        """
        tk_local.Popup(self._root).show(message)

        
    def dialogue_config(self, title, configs, callback):
        """Create a configuration dialogue window.
        
        Parameters
        ----------
        title : string
            Configuration window title.
        config : dictionary
            Dictionary with the configs.
        callback: function
            Method to be excecuted after success.
        """
        self._conf_dialog = tk_local.ConfigDialog(self._root, title, configs, command_ok = lambda *_: self._apply( callback ))
        
    def dialogue_choose_one(self, title, configs, callback):
        """Create a choose one dialogue window.
        
        Parameters
        ----------
        title : string
            Configuration window title.
        config : dictionary
            Dictionary with the options.
        callback: function
            Method to be excecuted after success.
        """
        self._conf_dialog = tk_local.ChooseOneDialog(self._root, title, configs, command_ok = lambda *_: self._apply( callback ))
        
    def dialogue_select(self, title, configs, callback):
        """Create a select dialogue window.
        
        Parameters
        ----------
        title : string
            Configuration window title.
        config : dictionary
            Dictionary with the options.
        callback: function
            Method to be excecuted after success.
        """
        self._conf_dialog = tk_local.SelectDialog(self._root, title, configs, command_ok = lambda *_: self._apply( callback ))
        
    def get_config_and_destroy(self):
        """Execute last created dialogue, get the results and erase it.
        """
        if self._conf_dialog is None:
            return None
        
        try:
            self._conf_dialog.update_and_validate_configs()
            
            configs = self._conf_dialog.get_configs()
            
            self._conf_dialog.destroy()
            self._conf_dialog = None
            
            return configs
        except:    
            self._conf_dialog.destroy()
            raise IException("Illegal values, please try again")


    def show_error(self, fmt, *args):
        """Show error window message.
        
        Parameters
        ----------
        fmt : string
            Message with format variables.
        *args : arguments
            List of arguments of message.
        """
        tkMessageBox.showerror("Error", fmt % args)
        
    def show_info(self, fmt, *args):
        """Show info window message.
        
        Parameters
        ----------
        fmt : string
            Message with format variables.
        *args : arguments
            List of arguments of message.
        """
        tkMessageBox.showinfo("Info", fmt % args)
        
    def show_warning(self, fmt, *args):
        """Show warning window message.
        
        Parameters
        ----------
        fmt : string
            Message with format variables.
        *args : arguments
            List of arguments of message.
        """
        tkMessageBox.showwarning("Warning", fmt % args)
        
    
    def ask_ok_cancel(self, title, question):
        """Show ok or cancel window dialogue.
        
        Parameters
        ----------
        title : string
            Window dialogue title.
        question : string
            Question to be answered with ok or cancel.
        """
        return tkMessageBox.askokcancel(title, question)

        
    def show(self):
        """Open a rendered GUI.
        """
        self._root.protocol("WM_DELETE_WINDOW", self.quit)
        self._root.mainloop()
        
    def quit(self):
        """Ask if user want to quit. If yes, close the GUI.
        """
        if tkMessageBox.askokcancel("Quit", "Do you want to quit?"):
            try:
                import weka.core.jvm as jvm
                jvm.stop()
            except:
                pass
            self._root.quit()
            self._root.destroy()

    def debug(self, event):
        """Just a debug test.
        """
        print("DEBUG")
        

    def _apply(self, f, *args):
        """Apply a method catching and handling any exception.
        
        Parameters
        ----------
        f : function
            Method to be executed.
        *args : arguments
            List of method arguments.
        """
        try:
            f(*args)
        except IException as exc:
            self._log_exception( str(exc), True )
        except:
            self._log_exception( IException.format_exception() )
                
    
    def _log_exception(self, message, dialog = False):
        """Log a catched exception.
        
        Parameters
        ----------
        message : string
            Exception message.
        dialog : boolean, optional, default = False
            If True show a error dialogue message.
        """
        if dialog == True:
            self.show_error( message )
        elif IException.DEBUG == True:
            self.append_log( message )
        else:
            self.show_error( message )
