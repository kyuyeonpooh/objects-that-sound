from multiprocessing import Pool
import os

import cv2
import numpy as np
from scipy import signal
from scipy.io import wavfile


class Extractor:
    def __init__(
        self,
        src_vid_dir,
        src_aud_dir,
        out_vid_dir,
        out_aud_dir,
        vid_file_head="video_",
        aud_file_head="audio_",
        max_extract=9,
        start_sec=0,
        interval_sec=1,
    ):
        self.src_vid_dir = src_vid_dir
        self.src_aud_dir = src_aud_dir
        self.out_vid_dir = out_vid_dir
        self.out_aud_dir = out_aud_dir
        self.vid_file_head = vid_file_head
        self.aud_file_head = aud_file_head
        self.max_extract = max_extract
        self.start_sec = start_sec
        self.interval_sec = interval_sec

    def extract_frame(self, vid_file):
        # parse video file information
        vid_path = os.path.join(self.src_vid_dir, vid_file)
        vid_id = os.path.splitext(vid_file)[0][len(self.vid_file_head) :]
        # out_vid_dir = os.path.join(self.out_vid_dir, vid_id)

        # video capture initialization with validity check on arguments
        try:
            cap = cv2.VideoCapture(vid_path)
            if not cap.isOpened():
                print("Video seems to be corrupted, vid_id: {}".format(vid_id))
                return False
            total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or self.start_sec + self.interval_sec * self.max_extract > total_frame / fps:
                print("Error in video FPS or in method arguments, vid_id: {}".format(vid_id))
                return False
        except cv2.error:
            print("Error occurs within cv2, vid_id: {}".format(vid_id))
        except:
            print("Error occurs when opening and parsing video file, vid_id: {}".format(vid_id))

        # extract frames
        frame_dict = dict()
        extract_cnt = 0
        start = self.start_sec
        end = start + self.interval_sec
        try:
            while extract_cnt < self.max_extract:
                # extract middle point of start and end point
                cap.set(cv2.CAP_PROP_POS_MSEC, (start + end) / 2 * 1000)
                success, frame = cap.read()
                if not success or frame is None:
                    print("Video capture is unsuccessful, vid_id: {}".format(vid_id))
                    return False
                frame_dict[str(extract_cnt)] = frame
                # update interval pointers
                start += self.interval_sec
                end += self.interval_sec
                extract_cnt += 1
        except:
            print("Error occurs when extracting a frame from video, vid_id: {}".format(vid_id))
            return False
        # save into npz file
        np.savez_compressed(os.path.join(self.out_vid_dir, vid_id + ".npz"), **frame_dict)
        return True

    def extract_spectrogram(
        self,
        aud_file,
        sr=48000,
        win_size=480,
        overlap=240,
        nfft=512,
        logscale=True,
        normalize=True,
        threshold=80,
        eps=1e-7,
    ):
        # parse audio file information
        aud_path = os.path.join(self.src_aud_dir, aud_file)
        aud_id = os.path.splitext(aud_file)[0][len(self.aud_file_head) :]

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
        if self.start_sec + self.interval_sec * self.max_extract > duration:
            print("Error in audio file or in method arguments, aud_id: {}".format(aud_id))
            return False

        # extract spectrograms
        spec_dict = dict()
        extract_cnt = 0
        start = self.start_sec
        end = start + self.interval_sec
        try:
            while extract_cnt < self.max_extract:
                cur_sample = sample[start * sr : end * sr]
                freq, time, spectrogram = signal.spectrogram(
                    cur_sample, fs=sr, nperseg=win_size, noverlap=overlap, nfft=nfft
                )
                # convert into log-scale spectrogram (magnitude to decibel)
                if logscale:
                    spectrogram = 10 * np.log10(spectrogram + eps)
                # normalize spectrogram from max decibel down to tolerance range
                """
                if normalize:
                    spectrogram = np.clip(spectrogram, np.max(spectrogram) - threshold, a_max=None)
                """
                # update interval pointers
                spec_dict[str(extract_cnt)] = spectrogram
                start += self.interval_sec
                end += self.interval_sec
                extract_cnt += 1
        except:
            print("Error occurs when extracting a spectrogram from audio, aud_id: {}".format(aud_id))
            return False
        # save into npz file
        np.savez_compressed(os.path.join(self.out_aud_dir, aud_id + ".npz"), **spec_dict)
        return True

    def run(self, ncpu=None, fail_fname="fail.csv"):
        # get cpu count
        if ncpu is None:
            ncpu = mp.cpu_count()
        print("Using {} processes for extraction.".format(ncpu))

        # make video and audio directory
        if not os.path.exists(self.out_vid_dir):
            os.makedirs(self.out_vid_dir)
        if not os.path.exists(self.out_aud_dir):
            os.makedirs(self.out_aud_dir)

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

        # check video and audio both exists for paritcular id
        print("Checking if video and audio both exist.")
        vid_set = set([v[len(self.vid_file_head) : -len(os.path.splitext(v)[1])] for v in vid_list])
        aud_set = set([a[len(self.aud_file_head) : -len(os.path.splitext(a)[1])] for a in aud_list])
        vid_miss = vid_set - aud_set
        aud_miss = aud_set - vid_set
        for vid_id in vid_miss:
            print("Warning: audio file is missing for video, vid_id: {}".format(vid_id))
        for aud_id in aud_miss:
            print("Warning: video file is missing for audio, aud_id: {}".format(aud_id))
        print("Check finished.")

        # failure list
        vid_fail = list()
        aud_fail = list()

        # multiprocessing
        with Pool(ncpu) as pool:
            # extract video frame
            percent = 1
            print("Starting video frame extraction.")
            for i, success in enumerate(pool.imap(self.extract_frame, vid_list)):
                if not success:
                    vid_fail.append(i)
                if i + 1 == int(0.0001 * percent * len(vid_list)):
                    print("Video preprocessing progress: {:.2f}%\r".format(percent / 100), end="")
                    percent += 1
            print("Video preprocessing finished.")

            # extract spectrogram
            percent = 1
            print("Starting spectrogram extraction.")
            for i, success in enumerate(pool.imap(self.extract_spectrogram, aud_list)):
                if not success:
                    aud_fail.append(i)
                if i + 1 == int(0.0001 * percent * len(aud_list)):
                    print("Audio preprocessing progress: {:.2f}%\r".format(percent / 100), end="")
                    percent += 1
            print("Audio preprocessing finished.")

        # dump failed id into csv file
        print("{} videos and {} audios failed to preprocess.".format(len(vid_fail), len(aud_fail)))
        print("Writing video or audio id failed to preprocess.")
        vid_fail = [vid_list[i] for i in vid_fail]
        aud_fail = [aud_list[i] for i in aud_fail]
        vid_fail_id = set([v[len(self.vid_file_head) : -len(os.path.splitext(v)[1])] for v in vid_fail])
        aud_fail_id = set([a[len(self.vid_file_head) : -len(os.path.splitext(a)[1])] for a in aud_fail])
        fail_id = list(vid_fail_id | aud_fail_id)
        fail_id.sort()
        fail_id = np.array(fail_id)
        np.savetxt(fail_fname, fail_id, fmt="%s")
        print("Saved failed id list in '{}'.".format(fail_fname))

        print("Preprocessing finished.")
