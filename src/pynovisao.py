#
"""
    This file must contain the implementation code for all actions of pynovisao.
    
    Name: pynovisao.py
    Author: Alessandro dos Santos Ferreira ( santosferreira.alessandro@gmail.com )
"""
import sys
import gc
from collections import OrderedDict
import numpy as np
import os
import interface
import types
import cv2
from interface.interface import InterfaceException as IException
from PIL import Image

import segmentation
import extraction
from extraction import FeatureExtractor

import classification
from classification import Classifier

import util
from extraction.extractor_frame_video import ExtractFM

from util.config import Config
from util.file_utils import File
from util.utils import TimeUtils
from util.utils import MetricUtils
from util.x11_colors import X11Colors
import multiprocessing
from multiprocessing import Process, Manager
import  threading
from tqdm import tqdm
class Act(object):
    """Store all actions of Pynovisao."""

    def __init__(self, tk, args):
        """Constructor.

        Parameters
        ----------
        tk : Interface
            Pointer to interface that handles UI.
        args : Dictionary
            Arguments of program.
        """
        self.tk = tk
        self.has_trained = False
        
        self.segmenter = [segmentation._segmenter_list[segmenter].meta for segmenter in segmentation._segmenter_list
                            if segmentation._segmenter_list[segmenter].value == True ][0]()
        
        self.extractors = [extraction._extractor_list[extractor].meta for extractor in extraction._extractor_list
                            if extraction._extractor_list[extractor].value == True ]
        
        try:
            self.classifier = [classification._classifier_list[classifier].meta for classifier in classification._classifier_list
                                if classification._classifier_list[classifier].value == True ][0]()
        except:
            self.classifier = None

        self._image = None
        self._const_image = None
        self._mask_image = None
        self._image_name = None
        self._image_path = None
                    
        self._init_dataset(args["dataset"])
        self._init_classes(args["classes"], args["colors"])

        self._dataset_generator = True
        self._ground_truth = False
        self._gt_segments = None
        self.weight_path = None

    
    def _init_dataset(self, directory):
        """Initialize the directory of image dataset.

        Parameters
        ----------
        directory : string
            Path to directory.
        """
        if(directory[-1] == '/'):
            directory = directory[:-1]
            
        self.dataset = directory
        File.create_dir(self.dataset)
    
    def _init_classes(self, classes = None, colors = None):
        """Initialize the classes of dataset.

        Parameters
        ----------
        classes : list of string, optional, default = None
            List of classes. If not informed, the metod set as classes all classes in dataset. 
            If there's no classes in dataset, adds two default classes.
        colors : list of string, optional, default = None
            List de colors representing the color of classe, in same order. If not informed, chooses a color at random.
        """
        self.classes = []

        dataset_description_path = File.make_path(self.dataset, '.dataset_description.txt')

        if os.path.exists(dataset_description_path):
            colors = []
            classes = []
            file = open(dataset_description_path, "r") 
            for line in file:
                class_info = line.replace("\n", "").split(",")
                classes.append(class_info[0])
                colors.append(class_info[1])                 
        else:
            classes = sorted(File.list_dirs(self.dataset)) if classes is None else classes.split()
            colors = [] if colors is None else colors.split()

        if(len(classes) > 0):
            for i in range(0, len(classes)):
                self.add_class(dialog = False, name=classes[i], color=colors[i] if i < len(colors) else None)
        else:
            self.add_class(dialog = False, color='Green')
            self.add_class(dialog = False, color='Yellow')
                
        self._current_class = 0
        

    def open_image(self, imagename = None):
        """Open a new image.

        Parameters
        ----------
        imagename : string, optional, default = None
            Filepath of image. If not informed open a dialog to choose.
        """
        
        def onclick(event):
            """Binds dataset generator event to click on image."""
            print(event)
            if event.xdata != None and event.ydata != None and int(event.ydata) != 0 and self._dataset_generator == True:
                x = int(event.xdata)
                y = int(event.ydata)
                self.tk.write_log("Coordinates: x = %d y = %d", x, y)
                
                segment, size_segment, idx_segment, run_time = self.segmenter.get_segment(x, y)
                
                if size_segment > 0:
                    self.tk.append_log("\nSegment = %d: %0.3f seconds", idx_segment, run_time)
                    
                    self._image, run_time = self.segmenter.paint_segment(self._image, self.classes[self._current_class]["color"].value, x, y)
                    self.tk.append_log("Painting segment: %0.3f seconds", run_time)
                    self.tk.refresh_image(self._image)
                    
                    if self._ground_truth == True:
                        self._gt_segments[idx_segment] = self.classes[self._current_class]["name"].value

                    elif self._dataset_generator == True:
                        filepath = File.save_class_image(segment, self.dataset, self.classes[self._current_class]["name"].value, self._image_name, idx_segment)
                        if filepath:
                            self.tk.append_log("\nSegment saved in %s", filepath)
        
        if imagename is None:
            imagename = self.tk.utils.ask_image_name()

        if imagename:
            self._image = File.open_image(imagename)
            self._image_name = File.get_filename(imagename)

            self.tk.write_log("Opening %s...", self._image_name)
            self.tk.add_image(self._image, self._image_name, onclick)
            self._const_image = self._image
            
            self.segmenter.reset()
            self._gt_segments = None

        

    def open_weight(self):
        """Open a new weight."""
        self.weight_path = self.tk.utils.ask_weight_name()
        self.classifier.weight_path = self.weight_path
        print(self.weight_path)
        
    def restore_image(self):
        """Refresh the image and clean the segmentation.
        """
        if self._const_image is not None:
            self.tk.write_log("Restoring image...")
            self.tk.refresh_image(self._const_image)
            
            self.segmenter.reset()
            self._gt_segments = None
        
    def close_image(self):
        """Close the image.
        
        Raises
        ------
        IException 'Image not found'
            If there's no image opened.
        """
        if self._const_image is None:
            raise IException("Image not found!  Open an image to test, select in the menu the option File>Open Image!")
        
        if self.tk.close_image():
            self.tk.write_log("Closing image...")
            self._const_image = None
            self._image = None
            self._image_path = None

    def add_class(self, dialog = True, name = None, color = None):
        """Add a new class.

        Parameters
        ----------
        dialog : boolean, optional, default = True
            If true open a config dialog to add the class.
        name : string, optional, default = None
            Name of class. If not informed set the name 'Class_nn' to class.
        color : string, optional, default = None
            Name of color in X11Color format, representing the class. It will used to paint the segments of class.
            If not informed choose a color at random.
            
        Raises
        ------
        IException 'You have reached the limite of %d classes'
            If you already have created self.tk.MAX_CLASSES classes.
        """
        n_classes = len(self.classes)
        if n_classes >= self.tk.MAX_CLASSES:
            raise IException("You have reached the limite of %d classes" % self.tk.MAX_CLASSES)
                
        def edit_class(index):
            """Calls method that edit the class."""
            self.edit_class(index)
            
        def update_current_class(index):
            """Calls method that update the class."""
            self.update_current_class(index)
        
        def process_config():
            """Add the class and refresh the panel of classes."""
            new_class = self.tk.get_config_and_destroy()
            new_class["name"].value = '_'.join(new_class["name"].value.split())

            self.classes.append( new_class )
            self.tk.write_log("New class: %s", new_class["name"].value)
            self.tk.refresh_panel_classes(self.classes, self._current_class)
            
        if name is None:
            name = "Class_%02d" % (n_classes+1)
        if color is None:
            color = util.X11Colors.random_color()
            
        class_config = OrderedDict()
        class_config["name"] = Config(label="Name", value=name, c_type=str)
        class_config["color"] = Config(label="Color (X11 Colors)", value=color, c_type='color')
        class_config["callback"] = Config(label=None, value=update_current_class, c_type=None, hidden=True)
        class_config["callback_color"] = Config(label=None, value=edit_class, c_type=None, hidden=True)
        class_config["args"] = Config(label=None, value=n_classes, c_type=int, hidden=True)
        
        if dialog == False:
            self.classes.append( class_config )
            return 

        title = "Add a new classe"
        self.tk.dialogue_config(title, class_config, process_config)        
      

    def edit_class(self, index):
        """Edit a class.

        Parameters
        ----------
        index : integer.
            Index of class in list self.classes.
        """
        def process_update(index):
            """Update the class."""
            updated_class = self.tk.get_config_and_destroy()
            updated_class["name"].value = '_'.join(updated_class["name"].value.split())
            
            self.classes[index] = updated_class
            self.tk.write_log("Class updated: %s", updated_class["name"].value)
            self.tk.refresh_panel_classes(self.classes, self._current_class)
        
        current_config = self.classes[index]
            
        title = "Edit class %s" % current_config["name"].value
        self.tk.dialogue_config(title, current_config, lambda *_ : process_update(index))
            
    def update_current_class(self, index):
        """Update the current class.
        """
        self._current_class = index
        
    def get_class_by_name(self, name):
        """Return the index for class.
        
        Parameters
        ----------
        name : string
            Name of class.
            
        Returns
        -------
        index : integer
            Index of class in list self.classes.

        Raises
        ------
        Exception 'Class not found'
            If name not found in self.classes.
        """
        name = name.strip()
        
        for cl in self.classes:
            if cl["name"].value == name:
                return cl
        raise Exception("Class not found")

        
    def set_dataset_path(self):
        """Open a dialog to choose the path to directory of image dataset.
        """
        directory = self.tk.utils.ask_directory(default_dir = self.dataset)
        if directory:
            self._init_dataset(directory)
            self.tk.write_log("Image dataset defined: %s", self.dataset)
            
            self._init_classes()
            self.tk.refresh_panel_classes(self.classes)
            
            if self.classifier: self.classifier.reset()
        self.has_trained=False
            
    def toggle_dataset_generator(self):
        """Enable/disable the dataset generator on click in image.
        """
        self._dataset_generator = not self._dataset_generator

            
    def select_segmenter(self):
        """Open a dialog to choose the segmenter.
        """
        title = "Choosing a segmenter"
        self.tk.write_log(title)

        current_config = segmentation.get_segmenter_config()
        
        def process_config():
            """Update the current segmenter."""
            new_config = self.tk.get_config_and_destroy()

            self.segmenter = [new_config[segmenter].meta for segmenter in new_config
                                if new_config[segmenter].value == True ][0]()

            self.tk.append_log("\nSegmenter: %s\n%s", str(self.segmenter.get_name()), str(self.segmenter.get_summary_config()))
            segmentation.set_segmenter_config(new_config)

        self.tk.dialogue_choose_one(title, current_config, process_config)

    def config_segmenter(self):
        """Open a dialog to configure the current segmenter.
        """
        title = "Configuring %s" % self.segmenter.get_name()
        self.tk.write_log(title)

        current_config = self.segmenter.get_config()
        
        def process_config():
            """Update the configs of current segmenter."""
            new_config = self.tk.get_config_and_destroy()

            self.segmenter.set_config(new_config)
            self.tk.append_log("\nConfig updated:\n%s", str(self.segmenter.get_summary_config()))
            self.segmenter.reset()

        self.tk.dialogue_config(title, current_config, process_config)
        
    def run_segmenter(self, refresh_image=True):
        """Do the segmentation of image, using the current segmenter.
        
        Raises
        ------
        IException 'Image not found'
            If there's no image opened.
        """
        if self._const_image is None:
            raise IException("Image not found!  Open an image to test, select in the menu the option File>Open Image!")
        
        self.tk.write_log("Running %s...", self.segmenter.get_name())

        self.tk.append_log("\nConfig: %s", str(self.segmenter.get_summary_config()))
        self._image, run_time = self.segmenter.run(self._const_image)
        self.tk.append_log("Time elapsed: %0.3f seconds", run_time)
        
        self._gt_segments = [None]*(max(self.segmenter.get_list_segments())+1)

        if refresh_image:
            self.tk.refresh_image(self._image)


    def select_extractors(self):
        """Open a dialog to select the collection of extractors.
        
        Raises
        ------
        IException 'Please select at least one extractor'
            If no extractor was selected.
        """
        title = "Selecting extractors"
        self.tk.write_log(title)

        current_config = extraction.get_extractor_config()
        
        def process_config():
            """Update the collection of extractors."""
            new_config = self.tk.get_config_and_destroy()

            self.extractors = [new_config[extractor].meta for extractor in new_config
                                if new_config[extractor].value == True ]
            
            if len(self.extractors) == 0:
                raise IException("Please select an extractor from the menu under Features Extraction> Select extractors! ")
            
            self.tk.append_log("\nConfig updated:\n%s", 
                                '\n'.join(["%s: %s" % (new_config[extractor].label, "on" if new_config[extractor].value==True else "off")
                                            for extractor in new_config]))
            extraction.set_extractor_config(new_config)

        self.tk.dialogue_select(title, current_config, process_config)
        
    def run_extractors(self):
        """Perform a feature extraction on all images of dataset, using the current collection of extractors.
        """
        self.tk.write_log("Running extractors on all images in %s", self.dataset)
        self.tk._root.update_idletasks()
        fextractor = FeatureExtractor(self.extractors,self.tk)
        self.tk.append_log("%s", '\n'.join([extraction._extractor_list[extractor].label for extractor in extraction._extractor_list
                                                if extraction._extractor_list[extractor].value == True ]))
        
        output_file, run_time = fextractor.extract_all(self.dataset, "training")
        self.tk.append_log("\nOutput file saved in %s", output_file)
        self.tk.append_log("Time elapsed: %0.3f seconds", run_time)
        
        if self.classifier: self.classifier.reset()

    def run_extract_frame(self):
        self.tk.write_log("Running extract frames from videos")
        extract_frame=ExtractFM()
        extract_frame.run(self.tk)

    def select_classifier(self):
        """Open a dialog to select the classifier.
        
        Raises
        ------
        IException 'You must install python-weka-wrapper'
            The user must install the required dependencies to classifiers.
        """
        if self.classifier is None:
            raise IException("Classifier not found! Select from the menu the option Training>Choose Classifier!")
        
        title = "Choosing a classifier"
        self.tk.write_log(title)

        current_config = classification.get_classifier_config()

        
        def process_config():
            """Update the current classifier."""
            new_config = self.tk.get_config_and_destroy()
            
            self.classifier = [new_config[classifier].meta for classifier in new_config
                                if new_config[classifier].value == True ][0]()

            self.tk.append_log("\nClassifier: %s\n%s", str(self.classifier.get_name()), str(self.classifier.get_summary_config()))
            classification.set_classifier_config(new_config)

        self.tk.dialogue_choose_one(title, current_config, process_config)
        
    def configure_classifier(self):
        """Set the configuration of current classifier.
        
        Raises
        ------
        IException 'You must install python-weka-wrapper'
            The user must install the required dependencies to classifiers.
        """
        if self.classifier is None:
            raise IException("Classifier not found! Select from the menu the option Training>Choose Classifier!")
        
        title = "Configuring %s" % self.classifier.get_name()
        self.tk.write_log(title)

        current_config = self.classifier.get_config()
        
        def process_config():
            new_config = self.tk.get_config_and_destroy()

            self.classifier.set_config(new_config)
            self.tk.append_log("\nConfig updated:\n%s", str(self.classifier.get_summary_config()))
            
            if self.classifier: self.classifier.reset()

        self.tk.dialogue_config(title, current_config, process_config)
    
    
    def run_classifier(self):
        """Run the classifier on the current image.
        As result, paint the image with color corresponding to predicted class of all segment.

        Raises
        ------
        IException 'You must install python-weka-wrapper'
            The user must install the required dependencies to classifiers.
        IException 'Image not found'
            If there's no image opened.
        """
        if self.classifier is None:
            raise IException("Classifier not found! Select from the menu the option Training>Choose Classifier!")

        if self._const_image is None:
            raise IException("Image not found!  Open an image to test, select in the menu the option File>Open Image!")

        self.tk.write_log("Running %s...", self.classifier.get_name())
        self.tk.append_log("\n%s", str(self.classifier.get_summary_config()))

        #self.classifier.set

        start_time = TimeUtils.get_time()

        # Perform a segmentation, if needed.
        list_segments = self.segmenter.get_list_segments()
        if len(list_segments) == 0:
            self.tk.append_log("Running %s... (%0.3f seconds)", self.segmenter.get_name(), (TimeUtils.get_time() - start_time))

            self._image, _ = self.segmenter.run(self._const_image)
            self.tk.refresh_image(self._image)
            list_segments = self.segmenter.get_list_segments()
            self._gt_segments = [None]*(max(list_segments)+1)

        #  New and optimized classification
        tmp = ".tmp"
        File.remove_dir(File.make_path(self.dataset, tmp))

        self.tk.append_log("Generating test images... (%0.3f seconds)", (TimeUtils.get_time() - start_time))

        len_segments = {}

        print("Wait to complete processes all images!")
        with tqdm(total=len(list_segments)) as pppbar:
            for idx_segment in list_segments:
                segment, size_segment, idx_segment = self.segmenter.get_segment(self, idx_segment=idx_segment)[:-1]
                # Problem here! Dataset removed.
                filepath = File.save_only_class_image(segment, self.dataset, tmp, self._image_name, idx_segment)
                len_segments[idx_segment] = size_segment
                pppbar.update(1)
            pppbar.close()


        # Perform the feature extraction of all segments in image ( not applied to ConvNets ).
        if self.classifier.must_extract_features():
            self.tk.append_log("Running extractors on test images... (%0.3f seconds)", (TimeUtils.get_time() - start_time))
            fextractor = FeatureExtractor(self.extractors)
            output_file, _ = fextractor.extract_all(self.dataset, "test", dirs=[tmp])

        self.tk.append_log("Running classifier on test data... (%0.3f seconds)", (TimeUtils.get_time() - start_time))

        # Get the label corresponding to predict class for each segment of image.
        labels = self.classifier.classify(self.dataset, test_dir=tmp, test_data="test.arff", image=self._const_image)
        File.remove_dir(File.make_path(self.dataset, tmp))

        # Result is the class for each superpixel
        if type(labels) is types.ListType:
            self.tk.append_log("Painting segments... (%0.3f seconds)", (TimeUtils.get_time() - start_time))

            # If ground truth mode, show alternative results
            if self._ground_truth == True:
                return self._show_ground_truth(list_segments, len_segments, labels, start_time)

            # Create a popup with results of classification.
            popup_info = "%s\n" % str(self.classifier.get_summary_config())

            len_total = sum([len_segments[idx] for idx in len_segments])
            popup_info += "%-16s%-16s%0.2f%%\n" % ("Total", str(len_total), (len_total*100.0)/len_total)

            # Paint the image.
            self._mask_image = np.zeros(self._const_image.shape[:-1], dtype="uint8")
            height, width, channels = self._image.shape
            self.class_color = np.zeros((height,width,3), np.uint8)
            for (c, cl) in enumerate(self.classes):
                idx_segment = [ list_segments[idx] for idx in range(0, len(labels)) if cl["name"].value == labels[idx] or c == labels[idx]]
                if len(idx_segment) > 0:
                    self._image, _ = self.segmenter.paint_segment(self._image, cl["color"].value, idx_segment=idx_segment, border=False)
                    for idx in idx_segment:
                        self._mask_image[self.segmenter._segments == idx] = c
                        self.class_color[self.segmenter._segments == idx] = X11Colors.get_color(cl["color"].value)

                len_classes = sum([len_segments[idx] for idx in idx_segment])
                popup_info += "%-16s%-16s%0.2f%%\n" % (cl["name"].value, str(len_classes), (len_classes*100.0)/len_total)


            self.tk.refresh_image(self._image)
            self.tk.popup(popup_info)
        else:
            # Result is an image
            self._mask_image = labels
            height, width, channels = self._image.shape
            self.class_color = np.zeros((height,width,3), np.uint8)

            for (c, cl) in enumerate(self.classes):
                self.class_color[labels == c] = X11Colors.get_color(cl["color"].value)

            self._image = cv2.addWeighted(self._const_image, 0.7, self.class_color, 0.3, 0)
            self.tk.refresh_image(self._image)


        end_time = TimeUtils.get_time()

        self.tk.append_log("\nClassification finished")
        self.tk.append_log("Time elapsed: %0.3f seconds", (end_time - start_time))
        gc.collect()

    def run_training(self):
        start_time = TimeUtils.get_time()
        
        # Training do not need an image opened (consider removing these two lines)
        #      if self._const_image is None:
        #          raise IException("Image not found")
        
        if self.classifier.must_train():
            
            if self.classifier.must_extract_features():
                self.tk.append_log("Creating training data... (%0.3f seconds)", (TimeUtils.get_time() - start_time))
                fextractor = FeatureExtractor(self.extractors)
                output_file, run_time = fextractor.extract_all(self.dataset, "training", overwrite = False)
        
            self.tk.append_log("Training classifier...")
            
            self.classifier.train(self.dataset, "training")

            self.tk.append_log("DONE (%0.3f seconds)",  (TimeUtils.get_time() - start_time))
        
        self._image = self._const_image
        self.has_trained=True

    
    def _show_ground_truth(self, list_segments, len_segments, labels, start_time):
        """Paint only wrong classified segments and show ground truth confusion matrix.
        
        Parameters
        ----------
        list_segments : list of integer
            List of index segments.
        len_segments : list of integer
            List of segments sizes.
        labels : list of string
            List of predicted class name for each segment.
        start_time : floating point
            Start time of classification.
        """
        classes = list(set(labels))
        classes.sort()
        
        n_segments = len(labels)
        spx_matrix = np.zeros((len(classes), len(classes)), np.int) 
        px_matrix = np.zeros((len(classes), len(classes)), np.int) 

        # Create the confusion matrix and paint wrong classified segments individually.
        for idx_segment in list_segments:
            if self._gt_segments[idx_segment] is not None:
                gt_class = classes.index(self._gt_segments[idx_segment])
                predicted_class = classes.index(labels[idx_segment])
                
                spx_matrix[ gt_class ][ predicted_class ] += 1
                px_matrix[ gt_class ][ predicted_class ] += len_segments[idx_segment]
        
                if gt_class != predicted_class:
                    self._image, _ = self.segmenter.paint_segment(self._image, self.get_class_by_name(labels[idx_segment])["color"].value, idx_segment=[idx_segment], border=False)
        
        # Create a popup with results of classification.
        popup_info = "%s\n" % str(self.classifier.get_summary_config())
        popup_info += Classifier.confusion_matrix(classes, spx_matrix, "Superpixels")
        popup_info += Classifier.confusion_matrix(classes, px_matrix, "PixelSum")
        
        self.tk.refresh_image(self._image)
        self.tk.popup(popup_info)

        end_time = TimeUtils.get_time()
            
        self.tk.append_log("\nClassification finished")
        self.tk.append_log("Time elapsed: %0.3f seconds", (end_time - start_time))
        

    def toggle_ground_truth(self):
        """Enable/disable ground truth mode.
        """
        self._ground_truth = not self._ground_truth
        
    def cross_validation(self):
        """Run a cross validation on all generated segments in image dataset.
        
        Raises
        ------
        IException 'You must install python-weka-wrapper'
            The user must install the required dependencies to classifiers.
        """
        if self.classifier is None:
            raise IException("Classifier not found! Select from the menu the option Training>Choose Classifier!")
        
        if self.classifier.must_train():
            self.tk.write_log("Creating training data...")
            
            fextractor = FeatureExtractor(self.extractors)
            output_file, run_time = fextractor.extract_all(self.dataset, "training", overwrite = False)
            self.classifier.train(self.dataset, "training")
        
        self.tk.write_log("Running Cross Validation on %s...", self.classifier.get_name())
        self.tk.append_log("\n%s", str(self.classifier.get_summary_config()))
        
        popup_info = self.classifier.cross_validate()
        self.tk.append_log("Cross Validation finished")
        self.tk.popup(popup_info)
        
    def experimenter_all(self):
        """Perform a test in all availabel classifiers e show the results.
        
        Raises
        ------
        IException 'You must install python-weka-wrapper'
            The user must install the required dependencies to classifiers.
        """
        if self.classifier is None:
            raise IException("Classifier not found! Select from the menu the option Training>Choose Classifier!")
        
        if self.tk.ask_ok_cancel("Experimenter All", "This may take several minutes to complete. Are you sure?"):
            if self.classifier.must_train():
                self.tk.write_log("Creating training data...")

                fextractor = FeatureExtractor(self.extractors)
                output_file, run_time = fextractor.extract_all(self.dataset, "training", overwrite = False)
                self.classifier.train(self.dataset, "training")
                
            self.tk.write_log("Running Experimenter All on %s...", self.classifier.get_name())
            
            popup_info = self.classifier.experimenter()
            self.tk.append_log("\nExperimenter All finished")
            self.tk.popup(popup_info)


    def about(self):
        self.tk.show_info("Pynovisao\n\nVersion 1.0.0\n\nAuthors:\nAdair da Silva Oliveira Junior\nAlessandro dos Santos Ferreira\nDiego Andre Sant Ana\nDiogo Nunes Goncalves\nEverton Castelao Tetila\nFelipe Silveira\nGabriel Kirsten Menezes\nGilberto Astolfi\nHemerson Pistori\nNicolas Alessandro de Souza Belete")
        
            
    def func_not_available(self):
        """Use this method to bind menu options not available."""
        self.tk.write_log("This functionality is not available right now.")

    def assign_using_labeled_image(self, imagename = None, refresh_image=True):
        """Open a new image.

        Parameters
        ----------
        imagename : string, optional, default = None
            Filepath of image. If not informed open a dialog to choose.
        """

        if len(self.segmenter.get_list_segments()) == 0:
            self.tk.write_log("Error: Image not segmented")
            return

        if self._image is None:
            self.tk.write_log("Error: Open the image to be targeted")
            return

        if imagename is None:
            imagename = self.tk.utils.ask_image_name()

        if imagename:
            self._image_gt = File.open_image_lut(imagename)
            self._image_gt_name = File.get_filename(imagename)

            self.tk.write_log("Opening %s...", self._image_gt_name)

            qtd_classes = len(self.classes)
            qtd_superpixel = len(self.segmenter.get_list_segments())

        tam_gt = self._image_gt.shape
        tam_im = self._image.shape
        if len(tam_gt) > 2:
            self.tk.write_log("Color image is not supported. You must open a gray-scale image")
            return

        if tam_gt[0] != tam_im[0] or tam_gt[1] != tam_im[1]:
            self.tk.write_log("Images with different sizes")
            return
            
        #hist_classes_superpixels = np.zeros((qtd_superpixel, qtd_classes), np.int)      
    
        #for i in range(0, tam_gt[0]):
        #    for j in range(0, tam_gt[1]):          
        #        class_pixel = self._image_gt[i,j]
        #        if class_pixel > qtd_classes:
        #            self.tk.write_log("There is no class for the pixel [%d,%d] = %d on the image", i, j, class_pixel)
        #        else:
        #            #segment, size_segment, idx_segment, run_time = self.segmenter.get_segment(px = j, py = i)
        #            idx_segment = self.segmenter._segments[i, j]
        #            hist_classes_superpixels[idx_segment, class_pixel] = hist_classes_superpixels[idx_segment, class_pixel] + 1
        #    if i % 10 == 0:
        #        self.tk.write_log("Annotating row %d of %d", i, tam_gt[0])
                
        qtd_bad_superpixels = 0
        
        for idx_segment in range(0, qtd_superpixel):
            hist_classes_superpixels = np.histogram(self._image_gt[self.segmenter._segments == idx_segment], bins=range(0,len(self.classes)+1))[0]

            idx_class = np.argmax(hist_classes_superpixels)
            sum_vector = np.sum(hist_classes_superpixels)
            if refresh_image:
                self._image, run_time = self.segmenter.paint_segment(self._image, self.classes[idx_class]["color"].value, idx_segment = [idx_segment])
            #self.tk.append_log("posicao maior = %x  --  soma vetor %d", x, sum_vector)
            if hist_classes_superpixels[idx_class]/sum_vector < 0.5:
                qtd_bad_superpixels = qtd_bad_superpixels + 1

            if self._ground_truth == True:
                self._gt_segments[idx_segment] = self.classes[self._current_class]["name"].value

            elif self._dataset_generator == True:
                if idx_segment % 10 == 0:
                    self.tk.write_log("Saving %d of %d", (idx_segment+1), qtd_superpixel)

                segment, size_segment, idx_segment, run_time = self.segmenter.get_segment(idx_segment = idx_segment)
                filepath = File.save_class_image(segment, self.dataset, self.classes[idx_class]["name"].value, self._image_name, idx_segment)
                if filepath:
                    self.tk.append_log("\nSegment saved in %s", filepath)

        self.tk.refresh_image(self._image)
        self.tk.write_log("%d bad annotated superpixels of %d superpixel (%0.2f)", qtd_bad_superpixels, qtd_superpixel, (float(qtd_bad_superpixels)/qtd_superpixel)*100)



    def run_segmenter_folder(self, foldername=None):

        if foldername is None:
            foldername = self.tk.utils.ask_directory()

        valid_images_extension = ['.jpg', '.png', '.gif', '.jpeg', '.tif']

        fileimages = [name for name in os.listdir(foldername)
                    if os.path.splitext(name)[-1].lower() in valid_images_extension]

        for (i,file) in enumerate(fileimages):
            path_file = os.path.join(foldername, file)
            self.open_image(path_file)
            self.run_segmenter(refresh_image=False)
            label_image = (os.path.splitext(file)[-2] + '_json')
            self.assign_using_labeled_image(os.path.join(foldername, label_image, 'label.png'), refresh_image=False)
            self.tk.write_log("%d of %d images", i, len(fileimages))

    def run_classifier_folder(self, foldername=None):

        if self.classifier is None:
            raise IException("Classifier not found! Select from the menu the option Training>Choose Classifier!")

        if foldername is None:
            foldername = self.tk.utils.ask_directory()

        valid_images_extension = ['.jpg', '.png', '.gif', '.jpeg', '.tif']

        fileimages = [name for name in os.listdir(foldername)
                    if os.path.splitext(name)[-1].lower() in valid_images_extension]

        fileimages.sort()

        all_accuracy = []
        all_IoU = []
        all_frequency_weighted_IU = []

        for file in fileimages:
            path_file = os.path.join(foldername, file)
            self.open_image(path_file)
            self.run_classifier()
            label_image = os.path.join(foldername, (os.path.splitext(file)[-2] + '_json'), 'label.png')
            self._image_gt = File.open_image_lut(label_image)
            self._image_gt_name = File.get_filename(label_image)

            tam_gt = self._image_gt.shape
            tam_im = self._mask_image.shape
            if len(tam_gt) > 2:
                self.tk.write_log("Color image is not supported. You must open a gray-scale image")
                return

            if tam_gt[0] != tam_im[0] or tam_gt[1] != tam_im[1]:
                self.tk.write_log("Images with different sizes")
                return

            
            confusion_matrix = MetricUtils.confusion_matrix(self._mask_image, self._image_gt)
            [mean_accuracy, accuracy] = MetricUtils.mean_accuracy(self._mask_image, self._image_gt)
            [mean_IoU, IoU] = MetricUtils.mean_IU(self._mask_image, self._image_gt)
            frequency_weighted_IU = MetricUtils.frequency_weighted_IU(self._mask_image, self._image_gt)

            print('Matriz de Confusao')
            print(confusion_matrix)

            print('Mean Pixel Accuracy')
            print(mean_accuracy)

            print('Pixel accuracy per class')
            print(accuracy)

            print('Mean Intersction over Union')
            print(mean_IoU)

            print('Intersction over Union per class')
            print(IoU)

            print('Frequency Weighted IU')
            print(frequency_weighted_IU)

            all_accuracy.append(accuracy)
            all_IoU.append(IoU)
            all_frequency_weighted_IU.append(frequency_weighted_IU)

            if not os.path.exists("../models_results/"):
                os.makedirs("../models_results/")
            
            path = File.make_path("../models_results/" + file + ".txt")
            path_img = File.make_path("../models_results/" + file + "_seg1.tif")
            path_img2 = File.make_path("../models_results/" + file + "_seg2.tif")

            img = Image.fromarray(self._image)
            img.save(path_img)
            img = Image.fromarray(self.class_color)
            img.save(path_img2)
            
            f=open(path,'ab')
            np.savetxt(f, ['Matriz de confusao'], fmt='%s')
            np.savetxt(f, confusion_matrix, fmt='%.5f')
            np.savetxt(f, ['\nAcuracia'], fmt='%s')
            np.savetxt(f, accuracy, fmt='%.5f')
            np.savetxt(f, ['\nInterseccao sobre uniao'], fmt='%s')
            np.savetxt(f, IoU, fmt='%.5f')
            np.savetxt(f, ['\nInterseccao sobre uniao com peso'], fmt='%s')
            np.savetxt(f, [frequency_weighted_IU], fmt='%.5f')
            f.close()


        path = File.make_path("../models_results/all_metrics.txt")
        f=open(path,'ab')
        np.savetxt(f, ['All Acuracia'], fmt='%s')
        np.savetxt(f, all_accuracy, fmt='%.5f')
        np.savetxt(f, ['\nAll IoU'], fmt='%s')
        np.savetxt(f, all_IoU, fmt='%.5f')
        np.savetxt(f, ['\nAll Frequency Weighted IU'], fmt='%s')
        np.savetxt(f, all_frequency_weighted_IU, fmt='%.5f')
        f.close()


    def run_grafic_confusion_matrix(self):
        '''
        Generate a a graphical confusion matrix where images are classified and according to classification go to the wrong or right folder.
        Just Available to WekaClassifier and CNNKeras.
        '''
        from classification import WekaClassifiers, CNNKeras
        
        is_weka = isinstance(self.classifier, WekaClassifiers)
        is_keras = isinstance(self.classifier, CNNKeras)
        if not (is_weka or is_keras):
            message='Only available to Weka and CNN Keras classifiers.'
            raise IException(message)

        
        if not self.has_trained:
            message='Dataset Must Be Trained.'
            raise IException(message)
        
        from os.path import abspath, isdir

        folder = self.tk.utils.ask_directory()
        if not folder:
            message = 'No selected directory.'
            raise IException(message)
            return
            
        folder = abspath(folder)
        dataset = abspath(self.dataset)
        if folder == self.dataset:
            title = 'Same Dataset'
            message = 'The dataset selected is the same of the trained. Are you sure that is right?'
            option=self.tk.ask_ok_cancel(title, message)
            if not option:
                return
                
        from os import listdir, mkdir
        listdirs=listdir(folder)
        size_dirs = reduce(lambda a,b: a+b, [0]+[len(listdir(folder+'/'+d)) for d in listdirs if isdir(folder+'/'+d)])
        if not size_dirs:
            message = 'Dataset has no content or the subfolder has no content.'
            raise IException(message)
            
        from shutil import rmtree
        from os import symlink
        
        def create_folder_struct(matrix_path, class_names, human, computer):
            try:
                rmtree(matrix_path)
            except Exception as e:
                pass

            mkdir(matrix_path,0o777)
            for class_ in class_names:
                real=matrix_path+human+class_+'/'
                mkdir(real, 0o777)
                for _class in class_names:
                    mkdir(real+computer+_class,0o777)




        header_output = 'Starting Graphical Confusion Matrix\n\n'
        index=folder[-2::-1].index('/')
        matrix_path=folder[:-(index+1)]+'folder_confusion_matrix'
        class_names, classes=listdir(folder), {}
        
        for i in range(len(class_names)-1,-1,-1):
            if isdir(dataset+'/'+class_names[i]):
                if class_names[i][0] != '.':
                    continue
            del class_names[i]
        for i, name in enumerate(class_names):
            classes[name], classes[i]=i, name
        images=[]
        
        for classe in class_names:
            image_names=listdir(folder+'/'+classe)
            for i in range(len(image_names)):
                image_names[i]=folder+'/',classe ,'/'+image_names[i]
            images.extend(image_names)
        
        human, computer = '/human_', '/computer_'
        create_folder_struct(matrix_path, class_names, human, computer)
        
        header_output_middle = header_output + 'Dataset selected: ' + folder + '\n\n'
        self.tk.write_log(header_output_middle + 'Initializing...')
        
        total = str(len(images))
        # internal function in method for create threads, cannot change for Process(Have a problem with JVM Instances)
        total = str(len(images))
        print("Waiting  finish classification!")
        for i, image_path in enumerate(images):
            original_name = reduce(lambda a, b: a + b, image_path)
            real_class_path = matrix_path + human + image_path[1]
            predicted = self.classifier.single_classify(original_name, folder, self.extractors, classes)
            message = header_output_middle + str(i + 1) + ' of ' + total + ' images classifield.'
            self.tk.write_log(message)
            predicted_class_path = real_class_path + computer + predicted
            predicted_name = predicted_class_path + image_path[2]
            symlink(original_name, predicted_name)
                


        message = header_output + 'Saved in ' + matrix_path
        self.tk.write_log(message)

