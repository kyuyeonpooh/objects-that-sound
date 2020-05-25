# ======================================================================
# [Note] Please run this file in root directory, not in "utils" folder.
# [Example] $ python preprocess.py        (X)
#           $ python utils/preprocess.py  (O)
# ======================================================================

import configparser
import os

from easydict import EasyDict

from utils.extractor import Extractor


# convert string to boolean variable
def stob(bool_str, config_name):
    if bool_str == "True" or bool_str == "true":
        return True
    elif bool_str == "False" or bool_str == "false":
        return False
    else:
        raise ValueError(
            "Configuration {} will only accept one among True, true, False, and false.".format(config_name)
        )


# open configuration file
config_file = "config.ini"
config = configparser.ConfigParser()
config.read(config_file)

# extract configurations with EasyDict
config_dict = {section: dict(config.items(section)) for section in config.sections()}
config = EasyDict(config_dict)
kwargs = config.preprocess

# convert bool string to bool type argument
kwargs.run_vid = stob(kwargs.run_vid, "run_vid")
kwargs.run_aud = stob(kwargs.run_aud, "run_aud")
kwargs.delete_missing_pair = stob(kwargs.delete_missing_pair, "delete_missing_pair")
kwargs.delete_failure = stob(kwargs.delete_failure, "delete_failure")
kwargs.randomcrop = stob(kwargs.randomcrop, "randomcrop")
kwargs.logscale = stob(kwargs.logscale, "logscale")

# convert int string to int type argument
kwargs.ncpu = int(kwargs.ncpu)
kwargs.nseg = int(kwargs.nseg)
kwargs.sr = int(kwargs.sr)
kwargs.winsize = int(kwargs.winsize)
kwargs.nfft = int(kwargs.nfft)

# convert float string to float type argument
kwargs.start_pos = float(kwargs.start_pos)
kwargs.interval = float(kwargs.interval)
kwargs.overlap = float(kwargs.overlap)
kwargs.eps = float(kwargs.eps)

# preprocess video and audio into npz file
if __name__ == "__main__":
    extractor = Extractor(**kwargs)
    extractor.run(**kwargs)
