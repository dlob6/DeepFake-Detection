#! pip install ffmpeg-python
#! pip install facenet-pytorch
#! pip install mmcv
#! apt install -y ffmpeg

import mmcv
import ffmpeg
from facenet_pytorch import MTCNN
import numpy as np
import os
import json
import glob
import time

import PIL
from scipy import signal
from scipy.io import wavfile
import cv2

LABEL_FILE = "metadata.json"

PATH_ROOT_VID = "/media/dlo/New Volume/DeepFake/"#"../input/deepfake-detection-challenge/"
PATH_OUT = PATH_ROOT_VID #""

VIDEO_EXT = '.mp4'
AUDIO_EXT = '.wav'

def write_faces_and_stfts(curr_dir):
    """
    Process a directory of videos for kaggle's deepfake challenge.
    For each video of the directory:
    Compute a stft of the audio portion of the file, save it to disk
    Detect the faces present in the file, save them as .jpg to disk
    """
    
    path_curr_vid = f"{PATH_ROOT_VID}{curr_dir}/"
    path_write_faces = f"{PATH_OUT}{curr_dir}/"
    
    print(f"Reading videos from {path_curr_vid}")
    video_names = []
    for filename in glob.iglob(path_curr_vid + '*.mp4', recursive=True):
        video_names.append(filename.split("/")[-1])
    if not os.path.isdir(path_write_faces):
        os.mkdir(path_write_faces)
        
    print(f"Writing .jpg of audio's STFTs in {path_write_faces}")
    face_detector = MTCNN(device = 'cuda')
    start_time = time.time()
    for ix, video_name in enumerate(video_names):

        output_path = f"{path_write_faces}{video_name.split('.')[0]}/"
        invid = f'{path_curr_vid}{video_name}'
        try:
            write_face_samples(face_detector, output_path, invid)
        except:
            print(f"Couldnt detect faces in {invid}")
        try:
            write_stfts(video_name, path_curr_vid, output_path)
        except:
            print(f"Couldnt stft {invid}")
        print(f"Average time per video {round((time.time() - start_time)/(ix + 1),2)} s")

def write_face_samples(model, output_path, invid):
    """
    Writes to disk a series of faces detected in a video sample"""
    
    if not os.path.isdir(output_path) :
        os.mkdir(output_path)
    
    video = mmcv.VideoReader(invid)
    for frame_ix, frame in enumerate(video):
        frame_name = f"{output_path}webcam_{frame_ix}_0.jpg"
        if os.path.isfile(frame_name): continue
            
        frame_img = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        _ = model(frame_img,frame_name)

def write_stfts(video_name, path_curr_vid, path_current_frames):
    """Creates .jpg files of the an audio signal's STFT.
    A video 'video_name' is taken as input, 
    a temporary .wav file containing its uncompressed audio is created,
    .jpg files containing the dB of its STFT are written to disk, 300 of them
    (one per video frame, to match the extracted faces).
    the temporary .wav file is then deleted.
    """
    
    video_file, audio_file = video_audio_files(video_name, path_curr_vid, path_current_frames)
    
    files_exist = {f : os.path.isfile(f.replace('\'','')) 
                   for f in [video_file, audio_file]}
    
    if not os.path.isdir(path_current_frames):
        os.mkdir(path_current_frames)
    # Create audio_file .wav file from video
    if not files_exist[audio_file] :
        (ffmpeg
        .input(video_file)
        .output(audio_file)
        .run())
    
    raw_samples = numpy_from_audio(audio_file)
    _, _, samples = get_stft_db(*raw_samples)
    
    drop_samples = -(samples.shape[1] % 300)
    samples = samples[:,:drop_samples]
    
    samples -= np.min(samples)
    samples /= np.max(samples)
    
    chunk_size = samples.shape[1] // 300
    for chunk_idx in range(300):
        fname = f"{path_current_frames}audio_{chunk_idx}.jpg"
        if not os.path.isfile(fname):
            PIL.Image.fromarray((samples[:,chunk_idx*chunk_size:(chunk_idx+1)*chunk_size] * 255)
                            .astype(np.uint8)).save(fname)
            
    delete_command = f"rm '{audio_file}'"
    os.system(delete_command)

def video_audio_files(video_name, path_curr_vid, path_current_frames):
    """Returns path names from a video name"""
    video_file = f"{path_curr_vid}{video_name}"
    audio_file = f"{path_current_frames}{video_name.replace(VIDEO_EXT,AUDIO_EXT)}"
    return video_file, audio_file

def numpy_from_audio(audio_file, downsample_factor = None):
    """
    Reads an audio .wav file and returns its samples in a numpy array, 
    and the sampling rate [Hz]"""
    sample_rate, samples = wavfile.read(audio_file.replace('\'',''))
    if downsample_factor is not None:
        samples = signal.resample(samples, len(samples) // downsample_factor)
        sample_rate //= downsample_factor
    drop_samples = -(len(samples) % sample_rate)
    return samples[:drop_samples], sample_rate

def get_stft_db(samples, sample_rate):
    """Reads in 'audio_file', takes its STFT with a window size of 128, 
    takes the magnitude of it, and returns its dB values.
    """
    f, t, Zxx = signal.stft(samples, fs = sample_rate, nperseg = 128)
    return f, t, np.log(np.abs(Zxx))

if __name__ == "__main__":
    # Pass a list of directories as input
    [write_faces_and_stfts(d) for d in sys.argv[1:]]
