import pandas as pd
import numpy as np
import glob
import json
import PIL
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

PATH_ROOT_DIR = "/media/dlo/New Volume/DeepFake/"
AUDIO_EXTENSION = ".jpg"
LABEL_FILE = "metadata.json"

# Data preparation
class DeepFakeDF():
    """ Indexes the faces & STFTs into a dataframe, to be used
    by the pytorch dataset. processing is a bit long"""
    def __init__(self, data_dirs, test = False):
        self.data_dirs = [f"{PATH_ROOT_DIR}{d}" for d in data_dirs]
        self.frame_dirs = [f"{d}" for d in self.data_dirs]
        audio = []
        video = []
        for frame_dir in self.frame_dirs:
            audio.extend(glob.glob(f"{frame_dir}*/audio*"))
            video.extend(glob.glob(f"{frame_dir}*/webcam*"))
            
        if not test:
            self.labels = self._get_labels()
        else:
            self.labels = {}
            
        self.video = video
        self.df = self._prep_df_audio(audio)
        self.video_dicts = self._prep_video_dicts(video)
        self._merge_audio_video()
        
    def get_df(self):
        return self.df
        
    def _get_labels(self):
        labels = {}
        for d in self.data_dirs:
            with open(f"{d}{LABEL_FILE}", "r") as f:
                labels.update({f"{k.split('.mp4')[0]}": v['label'] 
                             for k, v in json.load(f).items()})
                
        return labels
    
    def _merge_audio_video(self):
        """
        Audio has 1 sample per frame in any case. But the face extractor may have 
        missed some faces. This function attempts to provide a face for each audio sample.
        """
        self.df['dir'] = self.df['audio'].str.split("/").str[-3]
        
        # Actor 0:
        # Flagging frames for which actor 0 was detected
        self.df['actor_0'] = [self.video_dicts[0].get(tuple(o),np.nan) 
                              for o in self.df[['video_name', 'sample']].values.tolist()]
        
        # Creating path variables for frames in which actor 0 was detected
        act0 = self.df.loc[~self.df['actor_0'].isna()].copy()
        act0['actor_0'] = (PATH_ROOT_DIR + act0['dir'] + "/" + act0['video_name'] 
                           + "/" + "webcam_" + act0['sample'].astype(str) + "_0" 
                           + ".jpg")
        self.df.loc[~self.df['actor_0'].isna(), 'actor_0'] = act0
        
        # Actor 1:
        # Flagging frames for which actor 1 was detected
        self.df['actor_1'] = [self.video_dicts[1].get(tuple(o),np.nan) 
                              for o in self.df[['video_name', 'sample']].values.tolist()]
        # Creating path variables for frames in which actor 1 was detected
        act1 = self.df.loc[~self.df['actor_1'].isna()].copy()
        act1['actor_1'] = (PATH_ROOT_DIR + act1['dir'] + "/" + act1['video_name'] 
                           + "/" + "webcam_" + act1['sample'].astype(str) + "_1" 
                           + ".jpg")
        self.df.loc[~self.df['actor_1'].isna(), 'actor_1'] = act1
        
        # Filling NaNs. Forward fill per video name, so that missing faces are replaced
        # by the previous detected face.
        for vid in self.df['video_name'].unique():
            cond = (self.df['video_name'] == vid)
            
            self.df.loc[cond,'actor_0'] = (self.df.loc[cond,'actor_0']
                                           .fillna(method = 'ffill')
                                           .fillna(method = 'bfill'))
            
            self.df.loc[cond,'actor_1'] = (self.df.loc[cond,'actor_1']
                                           .fillna(method = 'ffill')
                                           .fillna(method = 'bfill'))
        
        # As not all videos have two actors, for now, simply copying the 1st actor into the 2nd
        # actor field when there is only 1 actor.
        self.df.loc[self.df['actor_1'].isna(), 'actor_1'] = self.df['actor_0']
        
        for col in ['audio', 'actor_0', 'actor_1']:
            self.df[col] = self.df[col].str.replace(PATH_ROOT_DIR,"")
        
    
    def _prep_df_audio(self, audio):
        """Returns a dataframe indexed on frames of videos.
        Contains the path to each .jpg of STFTs of video frames"""
        df = pd.DataFrame(audio, columns = ['audio'])
        df['video_name'] = df['audio'].str.split("/").str[-2]
        df['sample'] = df['audio'].str.split("/").str[-1].str.split(".").str[0].str.split("_").str[-1].astype(int)
        df['label'] = df['video_name'].apply(lambda x: self.labels.get(x,""))
        df.sort_values(by=['video_name','sample'], inplace = True)
        df['actor_0'] = ""
        df['actor_1'] = ""
        return df
    
    def _prep_video_dicts(self, video):
        """Returns dicts, one that tell if a face was detected in frames of videos,
        and one that tells if a second face was detected in frames of videos."""
        video_name, frame_name = zip(*[o.split("/")[-2:] for o in video])
        samples, actors = zip(*[o.replace(".jpg","").split("_")[-2:] for o in frame_name])
        samples = [int(o) for o in samples]
        actors = [int(o) for o in actors] 
        #samples = [int(o) for o in actors]
        #actors = [0 for o in samples]
        actor_0_present = {(v, s) : a for v, s, a in zip(video_name, samples, actors) if a == 0}
        actor_1_present = {(v, s) : a for v, s, a in zip(video_name, samples, actors) if a == 1}
        return actor_0_present, actor_1_present

    
# Custom pytorch dataset
class DeepFakeJPGDataset(Dataset):
    """DeepFakeJPGDataset. Opens .jpgs of either faces or STFTs for each frame of video.
    Returns tensors of a concatenatetion of the video's frames."""

    def __init__(self, df, col_name, transform = None, downsample_factor = 1):
        """df[col_name] has to contain paths to .jpg files,
        either of faces or of STFTs.
        transform: resize images - all cropped faces don't have the same
        shape, they won't fit together in a batch. Need to resize them,
        use transforms.Resize((150,100)) for example. 
        downsample_factor: use > 1 to not use all the frames of a video
        """
        self.x = df[col_name]
        self.y = df['label'].astype('category').cat.codes.astype(int)
        self.transform = transform
        self.downsample_factor = int(downsample_factor)
        self.n_images = 300 // self.downsample_factor
        self.col_name = col_name

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        list_image_paths = self.x.iloc[idx]
        # Opening one every 'downsample_factor' image
        images = [PIL.Image.open(PATH_ROOT_DIR + im) 
                  for im in list_image_paths[::self.downsample_factor]] #::self.downsample_factor
        target = self.y.iloc[idx]
        
        # Resizing
        if self.transform:
            images = [self.transform(im) for im in images]
            
        # Normalizing the .jpgs to [-0.5, 0.5] both for 
        # faces and sound STFTs.
        # TODO: Normalize faces with imagenet stats, as the 
        # model taking faces as input is pretrained on imagenet
        images = [(np.array(im) / 255.0) - 0.5 for im in images]
        
        # Adding channel dimension to STFT images (grayscale)
        if len(images[0].shape) == 2: 
            images = [im[...,None] for im in images]
            
        # (n_frames, channels, height, width)    
        return torch.Tensor(images).permute(0,3,1,2)

# Merging audio and video datasets  
class DeepFakeDetectionDataset(Dataset):
    """DeepFakeDetectionDataset. Merges faces & STFT datasets."""
    def __init__(self, x1, x2, y):
        self.x1,self.x2,self.y = x1,x2,y
    def __len__(self): 
        return len(self.y)
    def __getitem__(self, i): 
        return (self.x1[i], self.x2[i]), self.y[i]
