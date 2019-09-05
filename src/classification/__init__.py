from .classifier import Classifier

try:
    from .weka_classifiers import WekaClassifiers
except Exception as e: 
    WekaClassifiers = None
    print(e)


try:
    from .cnn_caffe import CNNCaffe
except Exception as e: 
    CNNCaffe = None
    print(e)
    
try:
    from .syntactic import Syntactic
except:
    Syntactic = None

try:
    from .cnn_keras import CNNKeras
except Exception as e: 
    CNNKeras = None
    print(e)

try:
    from .cnn_pseudo_label_keras import CNNPseudoLabel
except Exception as e: 
    CNNPseudoLabel = None
    print(e)


try:
    from .segnet_keras import SEGNETKeras
except Exception as e: 
    SEGNETKeras = None
    print(e)


__all__ = ["classifier", 
           "cnn_caffe",
           "cnn_keras",
           "cnn_pseudo_label_keras",
           "segnet_keras",
           "weka_classifiers",
           "syntactic"
           ]


from collections import OrderedDict
from util.config import Config


_classifier_list = OrderedDict( [ 
                            ["cnn_caffe", Config("Invalid" if CNNCaffe is None else CNNCaffe.__name__,
                                WekaClassifiers is None and CNNCaffe is not None, bool, meta=CNNCaffe, hidden=CNNCaffe is None)],
                            ["cnn_keras", Config("Invalid" if CNNKeras is None else CNNKeras.__name__,
                                CNNKeras is not None, bool, meta=CNNKeras, hidden=CNNKeras is None)],
                            ["cnn_pseudo_label_keras", Config("Invalid" if CNNPseudoLabel is None else CNNPseudoLabel.__name__,
                                CNNPseudoLabel is not None, bool, meta=CNNPseudoLabel, hidden=CNNPseudoLabel is None)],
                            ["segnet_keras", Config("Invalid" if SEGNETKeras is None else SEGNETKeras.__name__,
                                SEGNETKeras is not None, bool, meta=SEGNETKeras, hidden=SEGNETKeras is None)],                            
                            ["weka_classifiers", Config("Invalid" if WekaClassifiers is None else WekaClassifiers.__name__,
                                WekaClassifiers is not None, bool, meta=WekaClassifiers, hidden=WekaClassifiers is None)],
                            ["syntactic", Config("Invalid" if Syntactic is None else Syntactic.__name__,
                                Syntactic is not None, bool, meta=Syntactic, hidden=Syntactic is None)]
                        ] )

def get_classifier_config():
    return _classifier_list

def set_classifier_config(configs):
    _classifier_list["cnn_caffe"] = Config.nvl_config(configs["cnn_caffe"], _classifier_list["cnn_caffe"])
    _classifier_list["cnn_keras"] = Config.nvl_config(configs["cnn_keras"], _classifier_list["cnn_keras"])
    _classifier_list["cnn_pseudo_label_keras"] = Config.nvl_config(configs["cnn_pseudo_label_keras"], _classifier_list["cnn_pseudo_label_keras"])
    _classifier_list["segnet_keras"] = Config.nvl_config(configs["segnet_keras"], _classifier_list["segnet_keras"])
    _classifier_list["weka_classifiers"] = Config.nvl_config(configs["weka_classifiers"], _classifier_list["weka_classifiers"])
    _classifier_list["syntactic"] = Config.nvl_config(configs["syntactic"], _classifier_list["syntactic"])
