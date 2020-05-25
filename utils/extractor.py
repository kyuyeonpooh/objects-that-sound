import os
from functools import partial
from multiprocessing import Pool

import cv2
import numpy as np
from scipy import signal
from scipy.io import wavfile


class Extractor:
    def __init__(
        self,
        src_vid_dir,
        src_aud_dir,
        dst_vid_dir,
        dst_aud_dir,
        vid_fname_head="video_",
        aud_fname_head="audio_",
        vid_ext="mp4",
        aud_ext="wav",
        nseg=9,
        start_pos=0,
        interval=1,
        **kwargs
    ):
        self.src_vid_dir = src_vid_dir
        self.src_aud_dir = src_aud_dir
        self.dst_vid_dir = dst_vid_dir
        self.dst_aud_dir = dst_aud_dir
        self.vid_fname_head = vid_fname_head
        self.aud_fname_head = aud_fname_head
        self.vid_ext = vid_ext
        self.aud_ext = aud_ext
        self.nseg = nseg
        self.start_pos = start_pos
        self.interval = interval

    def extract_frame(self, vid_file, randomcrop=True, **kwargs):
        # parse video ID from video file path
        vid_path = os.path.join(self.src_vid_dir, vid_file)
        vid_id = os.path.splitext(vid_file)[0][len(self.vid_fname_head) :]

        # video capture initialization with validity check
        try:
            cap = cv2.VideoCapture(vid_path)
            if not cap.isOpened():
                print("Video seems to be corrupted, vid_id: {}".format(vid_id))
                return False
            total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or self.start_pos + self.interval * self.nseg > total_frame / fps:
                print("Error in video FPS or in method arguments, vid_id: {}".format(vid_id))
                return False
        except cv2.error:
            print("Error occurs within cv2, vid_id: {}".format(vid_id))
            return False
        except:
            print("Error occurs when opening and parsing video file, vid_id: {}".format(vid_id))
            return False

        # extract frames
        frame_dict = dict()
        seg_count = 0
        start = self.start_pos
        end = start + self.interval
        try:
            while seg_count < self.nseg:
                cap.set(cv2.CAP_PROP_POS_MSEC, (start + end) / 2 * 1000)  # extract middle point of start and end point
                success, frame = cap.read()
                if not success or frame is None:
                    print("Video capture is unsuccessful, vid_id: {}".format(vid_id))
                    return False
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # BGR to RGB (cv2 image is BGR)
                frame = cv2.resize(frame, (256, 256) if randomcrop else (224, 224))  # image resize
                frame_dict[str(seg_count)] = frame
                # update interval pointers
                start += self.interval
                end += self.interval
                seg_count += 1
        except:
            print("Error occurs when extracting a frame from video, vid_id: {}".format(vid_id))
            return False

        # save into npz file
        np.savez_compressed(os.path.join(self.dst_vid_dir, vid_id + ".npz"), **frame_dict)
        return True

    def extract_spectrogram(
        self, aud_file, sr=48000, winsize=480, overlap=0.5, nfft=512, logscale=True, eps=1e-7, **kwargs
    ):
        # parse audio ID from audio file path
        aud_path = os.path.join(self.src_aud_dir, aud_file)
        aud_id = os.path.splitext(aud_file)[0][len(self.aud_fname_head) :]

        # audio file reading with validity check on arguments
        try:
            rate, sample = wavfile.read(aud_path)
        except:
            print("Failed to open wav file, aud_id: {}".format(aud_id))
            return False
        if rate != sr:
            print("Given sampling rate does not match, aud_id: {}".format(aud_id))
            return False
        duration = len(sample) / sr
        if self.start_pos + self.interval * self.nseg > duration:
            print("Error in audio file or in method arguments, aud_id: {}".format(aud_id))
            return False

        # extract spectrograms
        spec_dict = dict()
        seg_count = 0
        start = self.start_pos
        end = start + self.interval
        try:
            while seg_count < self.nseg:
                cur_sample = sample[start * sr : end * sr]
                freq, time, spectrogram = signal.spectrogram(
                    cur_sample, fs=sr, nperseg=winsize, noverlap=winsize * overlap, nfft=nfft
                )
                # convert into log-scale spectrogram (magnitude to decibel)
                if logscale:
                    spectrogram = 10 * np.log10(spectrogram + eps)
                # update interval pointers
                spec_dict[str(seg_count)] = spectrogram
                start += self.interval
                end += self.interval
                seg_count += 1
        except:
            print("Error occurs when extracting a spectrogram from audio, aud_id: {}".format(aud_id))
            return False

        # save into npz file
        np.savez_compressed(os.path.join(self.dst_aud_dir, aud_id + ".npz"), **spec_dict)
        return True

    def run(
        self,
        ncpu=8,
        run_vid=True,
        run_aud=True,
        delete_missing_pair=True,
        delete_failure=True,
        fail_fname="csv/preprocess_fail.csv",
        **kwargs
    ):
        # get cpu count
        if ncpu == 0:
            ncpu = mp.cpu_count()
        print("Using {} processes for extraction.".format(ncpu))

        # make video and audio directory
        if not os.path.exists(self.dst_vid_dir):
            os.makedirs(self.dst_vid_dir)
        if not os.path.exists(self.dst_aud_dir):
            os.makedirs(self.dst_aud_dir)

        # prepare video and audio list
        vid_list = os.listdir(self.src_vid_dir)
        aud_list = os.listdir(self.src_aud_dir)
        vid_list.sort()
        aud_list.sort()
        if len(vid_list) == 0 or len(aud_list) == 0:
            print("Video or audio folder is empty.")
            return
        print("Found {} videos.".format(len(vid_list)))
        print("Found {} audios.".format(len(aud_list)))

        # validity check on file names
        vid_valid = list()
        aud_valid = list()
        for vid_file, aud_file in zip(vid_list, aud_list):
            if vid_file.startswith(self.vid_fname_head) and vid_file.endswith(self.vid_ext):
                vid_valid.append(vid_file)
            if aud_file.startswith(self.aud_fname_head) and aud_file.endswith(self.aud_ext):
                aud_valid.append(aud_file)
        print(
            "{} video files and {} audio files are rejected.".format(
                len(vid_list) - len(vid_valid), len(aud_list) - len(aud_valid)
            )
        )
        vid_list = vid_valid
        aud_list = aud_valid

        # check video and audio both exists for paritcular ID
        print("Checking if video and audio both exist.")
        vid_set = set([vid_file[len(self.vid_fname_head) : -len(self.vid_ext)] for vid_file in vid_list])
        aud_set = set([aud_file[len(self.aud_fname_head) : -len(self.aud_ext)] for aud_file in aud_list])
        vid_miss = vid_set - aud_set
        aud_miss = aud_set - vid_set
        print("{} video files do not have corresponding audio file.".format(len(vid_miss)))
        print("{} audio files do not have corresponding video file.".format(len(aud_miss)))
        print("Check finished.")

        # delete missing files in src directory whose pair does not exist
        if delete_missing_pair:
            print("Deleting unpaired files.")
            for vid_file in vid_miss:
                os.remove(os.path.join(self.src_vid_dir, vid_file + self.vid_ext))
            for aud_file in aud_miss:
                os.remove(os.path.join(self.src_aud_dir, aud_file + self.aud_ext))
            # reassign updated file list
            vid_list = os.listdir(self.src_vid_dir)
            aud_list = os.listdir(self.src_aud_dir)
            vid_list.sort()
            aud_list.sort()
            print("Deleted {} video and {} audio files.".format(len(vid_miss), len(aud_miss)))

        # failure list
        vid_fail = list()
        aud_fail = list()

        # multiprocessing with pool
        with Pool(ncpu) as pool:
            # extract video frames
            if run_vid:
                print("Starting video frame extraction.")
                for i, success in enumerate(pool.imap(partial(self.extract_frame, **kwargs), vid_list)):
                    if not success:
                        vid_fail.append(vid_list[i])
                    print("Video preprocessing progress: {} / {}\r".format(i + 1, len(vid_list)), end="")
                print("Video preprocessing finished.")

            # extract spectrograms
            if run_aud:
                print("Starting spectrogram extraction.")
                for i, success in enumerate(pool.imap(partial(self.extract_spectrogram, **kwargs), aud_list)):
                    if not success:
                        aud_fail.append(aud_list[i])
                    print("Audio preprocessing progress: {} / }\r".format(i + 1, len(aud_list)), end="")
                print("Audio preprocessing finished.")

        # dump failed id into csv file
        print("{} videos and {} audios failed to preprocess.".format(len(vid_fail), len(aud_fail)))
        print("Writing video or audio id failed to preprocess.")
        vid_fail_id = set([vid_file[len(self.vid_fname_head) : -len(self.vid_ext)] for vid_file in vid_fail])
        aud_fail_id = set([aud_file[len(self.aud_fname_head) : -len(self.aud_ext)] for aud_file in aud_fail])
        fail_list = list(vid_fail_id | aud_fail_id)
        fail_list.sort()
        fail_list = np.array(fail_list)
        np.savetxt(fail_fname, fail_list, fmt="%s")
        print("Saved failed id list in '{}'.".format(fail_fname))

        # delete files failed on preprocessing
        if delete_failure:
            vid_rm_count = 0
            aud_rm_count = 0
            fail_id_list = np.genfromtxt(fail_fname, delimiter=",", dtype=str).tolist()
            for fail_id in fail_id_list:
                vid_fail_file = os.path.join(self.dst_vid_dir, self.vid_fname_head + fail_id + self.vid_ext)
                if os.path.exists(vid_fail_file):
                    os.remove(vid_fail_file)
                    vid_rm_count += 1
                aud_fail_file = os.path.join(self.dst_aud_dir, self.aud_fname_head + fail_id + self.aud_ext)
                if os.path.exists(aud_fail_file):
                    os.remove(aud_fail_file)
                    aud_rm_count += 1
            print("Removed {} video files and {} audio files failed to preprocess.".format(vid_rm_count, aud_rm_count))

        print("Preprocessing all finished.")
