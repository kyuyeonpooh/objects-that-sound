# ======================================================================
# [Note] Please run this file in the root directory.
# [Example] ~/objects-that-sound$ python preprocess.py          (O)
#           ~/objects-that-sound/utils$ python ../process.py    (X)
# ======================================================================

import configparser
import os

from easydict import EasyDict

from utils.extractor import Extractor
from utils.util import stob


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
kwargs.remove_unpaired_raw = stob(kwargs.remove_unpaired_raw, "remove_unpaired_raw")
kwargs.remove_failure = stob(kwargs.remove_failure, "remove_failure")
kwargs.remove_unpaired_npz = stob(kwargs.remove_unpaired_npz, "remove_unpaired_npz")
kwargs.make_train_val_test_split = stob(kwargs.make_train_val_test_split, "make_train_val_test_split")
kwargs.randomcrop = stob(kwargs.randomcrop, "randomcrop")
kwargs.logscale = stob(kwargs.logscale, "logscale")

# convert int string to int type argument
kwargs.ncpu = int(kwargs.ncpu)
kwargs.total = int(kwargs.total)
kwargs.random_seed = int(kwargs.random_seed)
kwargs.nseg = int(kwargs.nseg)
kwargs.sr = int(kwargs.sr)
kwargs.winsize = int(kwargs.winsize)
kwargs.nfft = int(kwargs.nfft)

# convert float string to float type argument
kwargs.val_size = float(kwargs.val_size)
kwargs.test_size = float(kwargs.test_size)
kwargs.start_pos = float(kwargs.start_pos)
kwargs.interval = float(kwargs.interval)
kwargs.overlap = float(kwargs.overlap)
kwargs.eps = float(kwargs.eps)

if __name__ == "__main__":
    extractor = Extractor(**kwargs)
    # preprocessing video and audio files into npz format
    if kwargs.run_vid or kwargs.run_aud or kwargs.remove_unpaired:
        extractor.run(**kwargs)
    # remove redundant npz files
    if kwargs.remove_failure or kwargs.remove_unpaired_npz:
        extractor.remove_redundant(**kwargs)
    # make train, validation, and test data into different folders
    if kwargs.make_train_val_test_split:
        extractor.train_val_test_split(**kwargs)
