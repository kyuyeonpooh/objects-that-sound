import configparser
import os

from easydict import EasyDict

from util.extractor import Extractor

# open configuration file
config_file = "config.ini"
config = configparser.ConfigParser()
config.read(config_file)

# extract configurations with EasyDict
config_dict = {section: dict(config.items(section)) for section in config.sections()}
config = EasyDict(config_dict)

"""
please run this file in root directory not in "utils" folder
$ python preprocess.py          (X)
$ python utils/preprocess.py    (O)
"""
if __name__ == "__main__":
    # preprocess video and audio into npz file
    extractor = Extractor(src_vid_dir, src_aud_dir, out_vid_dir, out_aud_dir, vid_file_head, aud_file_head)
    extractor.run(ncpu=8)
