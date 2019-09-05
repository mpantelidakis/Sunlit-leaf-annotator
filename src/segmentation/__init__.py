from .segmenter import Segmenter
from .felzenszwalb import Felzenszwalb
from .quickshift import Quickshift
from .slic import Slic

__all__ = ["segmenter",
           "felzenszwalb",
           "quickshift",
            "slic"]


from collections import OrderedDict
from util.config import Config


_segmenter_list = OrderedDict( [ 
                            ["felzenszwalb", Config(Felzenszwalb.__name__, False, bool, meta=Felzenszwalb)],
                            ["quickshift", Config(Quickshift.__name__, False, bool, meta=Quickshift)],
                            ["slic", Config(Slic.__name__, True, bool, meta=Slic)],
                            ["invalid", Config("Invalid", False, bool, meta=None, hidden=True)]
                        ] )


def get_segmenter_config():
    return _segmenter_list

def set_segmenter_config(configs):
    _segmenter_list["felzenszwalb"] = Config.nvl_config(configs["felzenszwalb"], _segmenter_list["felzenszwalb"])
    _segmenter_list["quickshift"] = Config.nvl_config(configs["quickshift"], _segmenter_list["quickshift"])
    _segmenter_list["slic"] = Config.nvl_config(configs["slic"], _segmenter_list["slic"])
