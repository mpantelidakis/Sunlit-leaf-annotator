#!/usr/bin/python
# -*- coding: utf-8 -*-
#

import numpy as np
from xdg.Config import language
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics
from sklearn.cluster import KMeans
import os
from os import listdir
from PIL import Image
import cv2 
from glob import glob


class BoVW_SYNTACTIC(object):
   
    def __init__(self, x_train, y_train):
        self.ktestable  = 2
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = None
        self.y_test = None
        self.inference = []
        
    def set_test(self, x_test, y_test): 
        self.x_test = x_test 
        self.y_test = y_test
          
    def __define_class(self,values):
        output = []
        seen = set()
        for value in values:
            if value not in seen:
                output.append(value)
                seen.add(value)
        return output
    

    def __group_values_class(self,x_t, y_t):
        indexs = []
        y_t_len = len(y_t)
        for i in range(y_t_len):
            if (i != (y_t_len-1)):
                if y_t[i] != y_t[i+1]:
                    indexs.append(i+1)
        return np.split(x_t, indexs)
    
    
    def __create_sequences_to_classes(self, grupo_of_class):
        
        sequence_temp = []
        for j in grupo_of_class:
            temp = '-'.join(str(x) for x in j)  
            #temp = ''.join(str(x) for x in j)      
            sequence_temp.append(temp)
        return sequence_temp
    
    
    def __define_languages(self, x_t, y_t):
        array_values_classes = self.__group_values_class(x_t, y_t)    
        languages = []
        
        for i in array_values_classes:    
            languages.append(self.__create_sequences_to_classes(i))   
        
        return languages
    
    
    def __recognize(self, test_sentence):
        """ 
        This method recognizes a single image 
        It can be utilized individually as well.
        param test_img: representation of an image using visual words. Example: [ 40.  16.  16.  13.  13.].
        """ 
        class_prediction = -1
        preditions = []
        for language in self.inference:
            for key, kt in language.items():
                errors, detect = kt.detect(test_sentence)
                preditions.append({'class': key, 'errors': errors})
                    
        min_errors = min(preditions, key=lambda x:x['errors'])
        class_prediction = min_errors['class']
                                      
        return class_prediction             
        
    def testModelCrossValidation(self, n_splits, test_size, alphabet_size):
        
        X_train = np.concatenate((self.x_train , self.x_test))
        y_train = np.concatenate((self.y_train , self.y_test))
        
        kf = ShuffleSplit(n_splits, test_size, random_state=0)

        predictions = []
        classes = []
        accuracy_list = []
           
        for train_index, test_index in kf.split(X_train):        
            self.x_train, self.x_test = X_train[train_index], X_train[test_index]        
            self.y_train, self.y_test = y_train[train_index], y_train[test_index] 
            self.trainModel()
            pred, cl = self.testModel()  
            predictions.extend(pred)
            classes.extend(cl) 
            accuracy_list.append(metrics.accuracy_score(cl, pred)) 
            
        return accuracy_list, classes, predictions
    
    
    def trainModel(self):
        """
        Uses k-testable inference classifier 
        """        
        languages = (self.__define_languages(self.x_train, self.y_train))
        classes = self.__define_class(self.y_train)
       
        for language, cl in zip(languages, classes):
            self.inference.append({cl:KTestable.build(self.ktestable, language)})
    
           
    def testModel(self):
        
        languages = (self.__define_languages(self.x_test, self.y_test))
        classes = self.__define_class(self.y_test)
        predictions = []
        cls = []
        
        for sentences, c in zip(languages, classes):
            for sentence in sentences:
                cl = self.__recognize(sentence)
                predictions.append(cl)
                cls.append(c)            
        
        return (np.asarray([int(i) for i in predictions]), np.asarray([int(i) for i in cls]))      
 

# ============================================================================
# PUBLIC
# ============================================================================

class DFA(object):
    '''
    (Simple) Deterministic Finite Automata.
    '''
    def __init__(self, alphabet, accepts, start, states, trans):
        '''
        The constructor.
        :param alphabet: Alphabet of DFA
        :param accepts: Final states
        :param start: Start state
        :param states: States
        :param trans: Transitions (delta)
        '''
        self.alphabet = alphabet
        self.accepts = accepts
        self.current_state = start
        self.start = start
        self.states = states
        self.trans = trans
        self.count_failure = 0

    def reset(self):
        '''
        Reset DFA: set the current state to the start state.
        '''
        self.current_state = self.start

    def status(self):
        '''
        Get the status of the DFA.
        :return: True if the current state is in the set of the final states,
            otherwise False.
        '''
        return self.current_state in self.accepts


class KDFA(DFA):
    '''
    DFA built from K-Testable-Machine
    >>> dfa = DFA.build(ktmachine)
    >>> dfa.detect('00011100')
    '''
    STATE_FAILURE = 'FAILURE'

    def _detect_char(self, c):
        '''
        Detect the character: find the transition for the given character.
        :param c: Character to be detected
        '''
        if c in self.alphabet:
            # new state     
            state = self.current_state
            self.current_state = self.trans[self.current_state][c]
            if self.current_state == 'FAILURE':
                self.current_state = state
                self.count_failure += 1
        else:
            # unknown character
            self.current_state = self.STATE_FAILURE

    def _detect_string(self, s):
        '''
        Detect the string: the final state is crucial.
        :param s: String to be detected.
        '''
        for c in s:
            self._detect_char(c)

    def detect(self, s):
        '''
        Detect the string (string should belong to the language).
        :param s: String to be analyzed
        :return: True if the string is detected and belongs to the language,
            otherwise False.
        '''
        self.reset()    
        self.count_failure = 0    
        self._detect_string(s)
        return self.count_failure, self.status()

    @staticmethod
    def build(ktmachine):
        '''
        Build the DFA from the K-Testable-Machine.
        -# merge prefixes and short strings :: I&C
        -# create states as prefixes from I&C
        -# create states from valid strings T as suff(k-1) and pref(k-1)
        -# initialize transitions
        -# transitions for I&C
        -# transitions for set of valid strings T
        -# add state from suffixes to accepted states
        -# add state from short string to accepted states
        :param ktmachine: K-Testable-Machine
        :return: DFA from K-Testable-Machine
        '''
        accepts = []
        start = ''
        failure = KDFA.STATE_FAILURE
        states = [start, failure]
        trans = {}
        # merge I&C and create states as prefixes from I&C
        for x in set(ktmachine.prefixes) | set(ktmachine.shortstr):
            for i in range(1, len(x)+1):
                state = x[:i]
                if state not in states:
                    states.append(state)
        # create states from valid strings T as suff(k-1) and pref(k-1)
        for x in ktmachine.validstr:
            if len(x) > 1:
                tpref = x[:len(x)-1]
                tsuff = x[1:]
                if tpref not in states:
                    states.append(tpref)
                if tsuff not in states:
                    states.append(tsuff)
        # initialize transitions
        for x in states:
            trans[x] = {}
            for y in ktmachine.alphabet:
                trans[x][y] = failure
        # transitions for set of I and C
        for x in set(ktmachine.prefixes) | set(ktmachine.shortstr):
            if not x or x == failure:
                continue
            for i in range(0, len(x)):
                char = x[i]
                if i == 0:
                    source = ''
                else:
                    source = x[:i]
                dest = x[:i+1]
                trans[source][char] = dest
        # transitions for set of valid strings T
        for x in ktmachine.validstr:
            if len(x) < 2:
                # suffix and prefix are required
                continue
            source = x[:len(x)-1]
            dest = x[1:]
            char = x[-1]
            trans[source][char] = dest
        # add state from suffixes to accepted states
        for x in ktmachine.suffixes:
            if x not in accepts:
                accepts.append(x)
        # add state from short string to accepted states
        for x in ktmachine.shortstr:
            if x not in accepts:
                accepts.append(x)
        # DFA
        return KDFA(ktmachine.alphabet, accepts, start, states, trans)


class KTMachine(object):
    '''
    K-Testable-Machine.
    The machine can be built from the language and the K-value.
    Usage:
    >>> KTMachine.build(k, language)
    '''
    def __init__(self, k, alphabet, prefixes, shortstr, suffixes, validstr):
        '''
        The constructor of the machine
        for K, M=(alphabet, prefixes, shortstr, suffixes, validstr).
        :param k: K value
        :param alphabet: Alphabet
        :param prefixes: Prefixes (k-1)
        :param shortstr: short strings (<k)
        :param suffixes: suffixes (k-1)
        :param validstr: allowed/valid strings (k)
        '''
        self.k = k
        self.alphabet = alphabet
        self.prefixes = prefixes
        self.shortstr = shortstr
        self.suffixes = suffixes
        self.validstr = validstr

    @staticmethod
    def build(k, language):
        '''
        Build the machine from a language.
        -# get all short strings (size < k)
        -# get all prefixes and suffixes (k-1)
        -# extract allowed/valid strings (size == k)
        -# build alphabet
        :param k: K value
        :param language: The language as a list of language's words
        :return: The K-Testable-Machine
        '''
        alphabet = []
        prefixes = []
        shortstr = []
        suffixes = []
        validstr = []
        # build
        for word in language:
            # get all short strings (size < k)
            if len(word) < k:
                if word not in shortstr:
                    shortstr.append(word)
            # get all prefixes and suffixes (k-1)
            if len(word) >= (k-1):
                p = word[:k-1]
                s = word[len(word)-k+1:]
                if p not in prefixes:
                    prefixes.append(p)
                if s not in suffixes:
                    suffixes.append(s)
            # extract allowed strings (size == k)
            if len(word) >= k:
                for i in range(0, len(word)-k+1):
                    tword = word[i:i+k]
                    if tword not in validstr:
                        validstr.append(tword)
            # build alphabet by each character
            for c in word:
                if c not in alphabet:
                    alphabet.append(c)
        # done
        return KTMachine(k, alphabet, prefixes, shortstr, suffixes, validstr)


# ============================================================================
# PUBLIC
# ============================================================================
class KTestable(object):
    '''
    K-Testable public facade.
    '''
    def __init__(self, kdfa):
        '''
        The constructor with DFA
        :param kdfa: K-testable DFA
        '''
        self.kdfa = kdfa

    def detect(self, s):
        '''
        Detect the string (string should belong to the language).
        :param s: String to be analyzed
        :return: True if the string is detected and belongs to the language,
            otherwise False.
        '''
        return self.kdfa.detect(s)

    @staticmethod
    def build(k, language):
        '''
        Build the machine from a language.
        :param k: K value
        :param language: The language as a list of language's words
        :return: The K-Testable
        '''
        return KTestable(KDFA.build(KTMachine.build(k, language)))

# ============================================================================
# PUBLIC
# ============================================================================      


class BoVW(object):
    def __init__(self, alphabet_size):
        """
        This is method creates an object SIFT to extract alphabet_size key points of the image.
        n_clusters is equal alphabet_size/classes and it is type integer.
        """
        self.alphabet_size = alphabet_size
        self.n_clusters = self.alphabet_size
        self.kmeans_obj = KMeans(n_clusters = self.n_clusters)
        self.kmeans_ret = None
        self.mega_histogram = None
        self.descriptor_list = None 
        self.kp_list = None
        self.kmeans_ret = None
        self.n_images = 0
        
        self.file_helper = FileHelpers() 
        self.sift = SIFT()
        self.train_labels = None
        self.name_dict = None

    def __cluster(self):
        """    
        cluster using KMeans algorithm
        """
            
        #restructures list into vstack array of shape
        #M samples x N features for sklearn.cluster        
        descriptor_vstack = np.array(self.descriptor_list[0])
        for remaining in self.descriptor_list[1:]:
            descriptor_vstack = np.vstack((descriptor_vstack, remaining)) 
            
        self.kmeans_ret = self.kmeans_obj.fit_predict(descriptor_vstack)        
    
    def __remove_zeros(self):
        
        mega_histogram_temp = []
        for line in range(len(self.descriptor_list)):
            indices=[]
            for i in range(len(self.descriptor_list[line]),len(self.mega_histogram[line]),1):
                indices.append(i)
            mega_histogram_temp.append([i for j, i in enumerate(self.mega_histogram[line]) if j not in indices])
        
       
        self.mega_histogram = np.asarray(mega_histogram_temp) 
      
    
    def __develop_vocabulary_natural_form(self):
        
        """
        Each cluster denotes a particular visual word. 
        Every image can be represented as a combination of multiple visual words. 
        This is method represents each image as a list of multiple visual words. 
        """
        
        self.mega_histogram = np.array([np.zeros(self.alphabet_size, dtype=int) for i in range(self.n_images)])
        old_count = 0
        for i in range(self.n_images):
            l = len(self.descriptor_list[i])
            if l > self.alphabet_size: # if key points of the image > alphabet 
                l = self.alphabet_size
            for j in range(l):
                idx = self.kmeans_ret[old_count+j]
                self.mega_histogram[i][j] = idx
            old_count += l
         
         
               
        #print ("List of visual word generate to each image")
            

    def __develop_vocabulary_frequency_form(self):
        
        """
        Each cluster denotes a particular visual word. 
        Every image can be represented as a combination of multiple visual words. 
        The best method is to generate a sparse histogram that contains the frequency of 
        occurence of each visual word. Thus the vocabulary comprises of a set of histograms 
        of encompassing all descriptions for all images. 
        This is method represents each image as a list that contains the frequency of each visual word in the image. 
        """
        
        self.mega_histogram = np.array([np.zeros(self.alphabet_size, dtype=int) for i in range(self.n_images)])
        old_count = 0
        for i in range(self.n_images):
            l = len(self.descriptor_list[i])
            if l > self.alphabet_size: # if key points of the image > alphabet
                l = self.alphabet_size
            for j in range(l):
                idx = self.kmeans_ret[old_count+j]                    
                self.mega_histogram[i][idx] += 1
            old_count += l
        
        #print ("List that contains the frequency of each visual word in the image") 
    
             
    def load_data(self, train_path, histogram, train):
        """
        Represent each image as a character sequence.
        :param train_path: path of the image to training.
        :param test_path: path of the image to test.
        :param histogram: if true, create a histogram.
        
        :return: 
            (x_train, y_train): set of data for training. 
            Each i position of x_train represents the visual words of an image. 
            Each i position of y_train represents the class of the i position of x_train.
            Example: 
            x_train [[ 40  16  16  13  13], [ 49  60   8  53  18]]
            y_train [[0],[1]]
            The image of x_train[0] belongs to class y_train [0]
            The image of x_train[1] belongs to class y_train [1]
            The same applies to the (x_test, y_test), using for testing.
            
            self.name_dict - a dictionary of all classes 
            having key => label class  
            objectname => class name
            Example: {0:"daninhas", 1:"soja"}  
        """
        
        # training data
        self.__load_data_set(train_path, histogram, train)
        self.__remove_zeros()
        x_train = self.mega_histogram
        y_train = self.train_labels
       
          
        return ((x_train, y_train), self.name_dict)
    
    
    def __load_data_set(self, path, histogram, train):
        
        # read file. prepare file lists.
        images = None
        if train:
            images, self.n_images = self.file_helper.getFiles(path)
        else:
            images, self.n_images = self.file_helper.getFilesImageTest(path)
         
        # extract SIFT Features from each image
        label_count = 0
        self.descriptor_list = []
        self.name_dict = {}
        self.train_labels = np.array([])
        self.kp_list = []
                 
        for word, imlist in images.items():
            
            #self.name_dict[str(label_count)] = word
            self.name_dict[word] = str(label_count)
            # Computing Features for each word
            for im in imlist:
                self.train_labels = np.append(self.train_labels, label_count)                
                kp, des = self.sift.features(im)
                self.descriptor_list.append(des)
                self.kp_list.append(kp)

            label_count += 1
         
        self.__cluster()
        
        if histogram: 
            self.__develop_vocabulary_frequency_form()           
        else:
            self.__develop_vocabulary_natural_form()
            




# ============================================================================
# PUBLIC
# ============================================================================


class SIFT(object):
    
    def __init__(self):
        """
        This is method create an object SIFT to extract alphabet_size key points of the image.
        alphabet_size is type integer.
        """
        self.sift_object = cv2.xfeatures2d.SIFT_create()

    def gray(self, image):
        """
        This is method renders grayscale image to the SIFT use.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    def features(self, image):
        keypoints, descriptors = self.sift_object.detectAndCompute(image, None)
        return [keypoints, descriptors]


class FileHelpers(object):

    def __init__(self):
        pass
    
    '''
    Retorna os arquivos encontrados no path
    '''   
    def getFilesImageTest(self, path): 
        imlist = {}
        word = 'test'
        imlist[word] = []
        count = 0 
        
        for infile in listdir(path):
            file, ext = os.path.splitext(infile)
            try:
                im = Image.open(path+file+ext)
                im.save(path+file+'.jpg', "JPEG")
                im = cv2.imread(path+file+'.jpg', 0)
                imlist[word].append(im)
                count +=1 
            except IOError:
                print('')
             
        return [imlist, count]
       
       
    def getFiles(self, path):
        """
        - returns a dictionary of all files 
        having key => value as  objectname => image path

        - returns total number of files.
        """
        imlist = {}
        count = 0
        for each in glob(path + "*"):
            if os.path.isfile(each):
                continue
            word = each.split("/")[-1]
            #print (" #### Reading image category ", word, " ##### ")
            imlist[word] = []
            for imagefile in glob(path+word+"/*"):
                #print ("Reading file ", imagefile)
                im = cv2.imread(imagefile, 0)
                imlist[word].append(im)
                count +=1 

        return [imlist, count]

   
