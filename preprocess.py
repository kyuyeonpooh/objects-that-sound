import os
import configparser
from util import extractor

# open configuration file
config_file = "config.ini"
config = configparser.ConfigParser()
config.read(config_file)

# extract configurations
data_dir = config["DATA_DIR"]
src_vid_dir = data_dir["src_video_dir"]
src_aud_dir = data_dir["src_audio_dir"]
out_vid_dir = data_dir["out_video_dir"]
out_aud_dir = data_dir["out_audio_dir"]

# raw data file naming conventions
data_file = config["DATA_FILE"]
vid_file_head = data_file["vid_file_head"]
aud_file_head = data_file["aud_file_head"]

if __name__ == "__main__":
    ext = extractor.Extractor(src_vid_dir, src_aud_dir, out_vid_dir, out_aud_dir, vid_file_head, aud_file_head)
    ext.run(ncpu=8)
