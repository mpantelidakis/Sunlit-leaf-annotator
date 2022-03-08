#!/usr/bin/python
# -*- coding: utf-8 -*-
#
"""
    Read the args, creates GUI and add all actions of pynovisao.
    
    This file is the main code of Pynovisao.
    To add a new action ( functionality ) to this software, you must implement the method body of action in file pynovisao.py 
    and bind the method to menu, using the call tk.add_command or tk.add_check_button.
    
    To add a new Feature Extractor, Segmenter or Classifier, place it in correct directory and extend its abstract class.
    
    As PEP 20 says, "Readability counts". So, follow the current conventions adopted in this project.
    For more information access PEP 8 -- Style Guide for Python Code ( https://www.python.org/dev/peps/pep-0008/ ). 
    Use it as your guideline. Only use CamelCase in class names. DON'T USE CamelCase ou mixedCase in variables or functions.
    
    All declarations and comments in this project must be made in English.
    
    Name: main.py
    Author: Alessandro dos Santos Ferreira ( santosferreira.alessandro@gmail.com )
"""
import argparse

import interface

from pynovisao import Act


def get_args():
    """Read the arguments of program."""
    ap = argparse.ArgumentParser()

    ap.add_argument("-d", "--dataset", required=False, help="Dataset path", default="../data/demo", type=str)
    
    ap.add_argument("-cl", "--classes", required=False, help="Classes names", default="Sunlit Noise", type=str)
    ap.add_argument("-co", "--colors", required=False, help="Classes colors (X11 color)", default="Green Brown", type=str)
    
    return vars(ap.parse_args())


if __name__ == "__main__":
    # Interface of software
    tk = interface.TkInterface("Pynovisao")
    
    # Bind all Pynovisao actions to menuim=np.copy(image)
    act = Act(tk, get_args())
    
    tk.add_menu("File")
    tk.add_command("Open a image", act.open_image, 'O') 
    tk.add_command("Restore image", act.restore_image, 'R')
    tk.add_separator()
    tk.add_command("Close image", act.close_image, 'W') 
    tk.add_separator()
    tk.add_command("Quit", tk.quit, 'Q')
    
    tk.add_menu("View")
    tk.add_check_button("Show image axes", tk.toggle_image_axes)
    tk.add_check_button("Show image toolbar", tk.toggle_image_toolbar)
    tk.add_separator()
    tk.add_check_button("Show log", tk.toggle_log)
    
    tk.add_menu("Dataset")
    tk.add_command("Add new class", act.add_class, 'A')
    tk.add_command("Set dataset path", act.set_dataset_path, 'd')
    tk.add_separator()
    tk.add_check_button("Dataset generator", act.toggle_dataset_generator)
    
    tk.add_menu("Segmentation")
    tk.add_command("Choose segmenter", act.select_segmenter)
    tk.add_command("Configure", act.config_segmenter, 'g')
    tk.add_separator()
    tk.add_command("Execute", act.run_segmenter, 'S')
    tk.add_separator()
    tk.add_command("Assign using labeled image", act.assign_using_labeled_image, 'l')
    tk.add_command("Execute folder", act.run_segmenter_folder)
    
    tk.add_menu("Feature Extraction")
    tk.add_command("Select extractors", act.select_extractors, 'e')
    tk.add_separator()
    tk.add_command("Execute", act.run_extractors, 'F')

    tk.add_separator()
    tk.add_command("Extract frames", act.run_extract_frame, 'V')

    tk.add_menu("Training")
    tk.add_command("Choose classifier", act.select_classifier)
    tk.add_command("Configure", act.configure_classifier)
    tk.add_separator()
    tk.add_command("Execute", act.run_training, 'T')
    
    tk.add_menu("Classification")
    tk.add_command("Load h5 weight (only for CNNs)", act.open_weight)
    tk.add_command("Execute", act.run_classifier, 'C')
    tk.add_command("Execute folder", act.run_classifier_folder)

    tk.add_menu("Experimenter")
    tk.add_check_button("Ground Truth", act.toggle_ground_truth, default_state = False)
    tk.add_separator()
    tk.add_command("Execute Graphical Confusion Matrix", act.run_grafic_confusion_matrix)
    tk.add_separator()
    tk.add_command("Cross Validation", act.cross_validation, 'X')
    tk.add_command("Experimenter All", act.experimenter_all, 'p')

    tk.add_menu("Help")
    tk.add_command("About Pynovisao", act.about, 'b')
    
    
    # Render the GUI
    tk.render_menu()
    
    tk.add_panel_classes( act.classes )
    
    tk.open_log()

    # Open the GUI
    tk.show()
