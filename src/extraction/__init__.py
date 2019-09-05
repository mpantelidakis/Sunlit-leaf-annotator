from .extractor import Extractor
from .feature_extraction import FeatureExtractor
from .color_summarizer import ColorStats
from .glcm import GLCM
from .hog import HOG
from .image_moments import RawCentralMoments, HuMoments
from .lbp import LBP
from .gabor import GABOR
from .kcurvature import KCURVATURE


__all__ = ["extractor",
            "color_summarizer",
            "glcm",
            "hog",
            "image_moments",
            "lbp",
            "gabor",
           "kcurvature"]


from collections import OrderedDict
from util.config import Config


_extractor_list = OrderedDict( [ 
                            ["color_summarizer", Config("Color Statistics", True, bool, meta=ColorStats)],
                            ["glcm", Config("Gray-Level Co-Occurrence Matrix", True, bool, meta=GLCM)],
                            ["hog", Config("Histogram of Oriented Gradients", True, bool, meta=HOG)],
                            ["hu_moments", Config("Hu Image Moments", True, bool, meta=HuMoments)],
                            ["rc_moments", Config("Image Moments (Raw/Central)", True, bool, meta=RawCentralMoments)],
                            ["lbp", Config("Local Binary Patterns", True, bool, meta=LBP)],
                            ["gabor", Config("Gabor Filter Bank", True, bool, meta=GABOR)],
                            ["kcurvature", Config("K-Curvature Angles", True, bool, meta=KCURVATURE)]
                        ] )

def get_extractor_config():
    return _extractor_list

def set_extractor_config(configs):
        _extractor_list["color_summarizer"] = Config.nvl_config(configs["color_summarizer"], _extractor_list["color_summarizer"])
        _extractor_list["glcm"] = Config.nvl_config(configs["glcm"], _extractor_list["glcm"])
        _extractor_list["hog"] = Config.nvl_config(configs["hog"], _extractor_list["hog"])
        _extractor_list["hu_moments"] = Config.nvl_config(configs["hu_moments"], _extractor_list["hu_moments"])
        _extractor_list["rc_moments"] = Config.nvl_config(configs["rc_moments"], _extractor_list["rc_moments"])
        _extractor_list["lbp"] = Config.nvl_config(configs["lbp"], _extractor_list["lbp"])
        _extractor_list["gabor"] = Config.nvl_config(configs["gabor"], _extractor_list["gabor"])
        _extractor_list["kcurvature"] = Config.nvl_config(configs["kcurvature"], _extractor_list["kcurvature"])
