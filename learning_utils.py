import fastai
from fastai.text.models.transformer import MultiHeadAttention
import torch
import torch.nn as nn
import torch.nn.functional as F

# The DeepFakeDetector model consists of 4 subnetworks.
#
# 1. Face model: takes a face detected from a video frame as input 
#    and outputs a face embedding.
# 2. Audio model: takes the STFT of the audio of a video frame and outputs an audio embedding.
# 3. Frame model: takes a face embedding and the corresponding audio embedding (concatenated)
#    as inputs, and outputs a frame embedding.
# 4. Video model: takes as input the concatenation of the frame embeddings of a video, 
#    and outputs the TRUE/FAKE label of the video.
#
# The 4 subnetworks need to be passed as arguments to the DeepFakeDetector constructor.
#
# The subnetworks can be of any kind. But the dataloaders provided in 'data_utils.py'
# suggest using Convnet2Ds as Face and Audio models. The second part of this file provides
# sample code for the Frame and Video models.
#
# Example of usage:
#
# n_frames = 30
# face_embedding_size = 512
# stft_embedding_size = 64
# frame_embedding_size = 64
#
# model_faces = fastai.vision.learner.create_body(fastai.vision.models.resnet18) #output of size 512
#
# model_stfts = fastai.vision.learner.simple_cnn(actns = [1,8,16,32,stft_embedding_size],
#                                               strides = [(2,1),(2,2),(2,2),(1,1)],
#                                               bn = True)
#
# merge_layers = [face_embedding_size + stft_embedding_size, frame_embedding_size]
# model_frame = learning_utils.fully_connected(merge_layers, dropout = [0.1]*len(merge_layers))
#
# model_video = learning_utils.VideoModel(frame_embedding_size, n_frames)
#
# model = learning_utils.DeepFakeDetector(model_faces, model_stfts, model_frame, 
#                                         model_video, n_frames = n_frames)


##########################
# DeepFakeDetector Model #
##########################
class DeepFakeDetector(torch.nn.Module):
    """DeepFakeDetectionModel."""
    
    def __init__(self, model_face, model_stft, model_frame, model_video, n_frames): 
        super().__init__()
        self.n_frames = n_frames
        self.model_face = model_face
        self.model_stft = model_stft
        self.model_frame = model_frame
        self.model_video = model_video

        # Need to avgpool and flatten the output of the convnet used to analyze faces 
        self.poolflat = fastai.layers.PoolFlatten()

        # used to provide different learning rates to subnetworks
        self.layer_groups = [self.model_face, self.model_stft,
                             self.model_frame, self.model_video]

    def forward(self, *x):
        x_faces = x[0]
        x_stfts = x[1]
        
        frame_embeddings = []
        for frame in range(self.n_frames):
            
            x_face = self.model_face(x_faces[:,frame,:,:,:])
            x_face = self.poolflat(x_face)
            x_stft = self.model_stft(x_stfts[:,frame,:,:,:])
            x = torch.cat([x_face, x_stft], dim=1)
            x = self.model_frame(x)
            frame_embeddings.append(x[:,None,:])
        
        x = torch.cat(frame_embeddings, dim = 1)
        x = self.model_video(x)
        return F.log_softmax(x, dim = -1)
    
###############################
# HELPER CODE FOR SUBNETWORKS #
###############################

# Example of a Frame model, a stack of Linear layers (+ BatchNorm and ReLU)
def fully_connected(layers, dropout, bn = True):
    """Returns a series of [BatchNorm1d, Dropout, Linear]*len(layers)
    The size of the linear layers is given by 'layers'. """
    model_layers = [] 
    activations = [nn.ReLU(inplace=True)] * (len(layers)-1)
    for n_in, n_out, p, actn in zip(layers[:-1], layers[1:], dropout, activations):
        model_layers += fastai.layers.bn_drop_lin(n_in, n_out, p = p, actn = actn, bn = bn)
    return nn.Sequential(*model_layers)

# Example of a Video model. The model consists of two parts:
# 1. A Transformer part that does the Self Attention magic over frames,
# input and output shape of the Transformer: (batch_size, n_frames, frame_embedding_size).
# 2. A classifier that flattens the output of the Transformer and predicts TRUE/FAKE
class VideoModel(nn.Module):
    def __init__(self, frame_embedding_size, n_frames, n_heads = 5, n_layers = 2): 
        super().__init__()
        d_head = frame_embedding_size // n_heads
        d_inner = int(1.5*frame_embedding_size)
        """self.transformer = Transformer(n_layers = n_layers, n_heads = n_heads,
                                       d_model = frame_embedding_size,
                                       d_head = d_head, d_inner = d_inner)"""
        self.pos_enc = fastai.text.models.transformer.PositionalEncoding(frame_embedding_size)
        self.mha = MultiHeadAttention(n_heads = n_heads, d_model = frame_embedding_size, d_head = d_head)
        self.classifier = nn.Linear(frame_embedding_size*n_frames,2)
    
    def forward(self, x):
        bs, x_len, _ = x.size()
        pos = torch.arange(0, x_len, device=x.device, dtype=x.dtype)
        x = self.mha(x + self.pos_enc(pos)[None])
        x = x.view(x.size(0), -1) 
        x = self.classifier(x)
        return x

# That's it for the DeepFakeDetector model and utils. Below comes some code for the Transformer
# used in the Video model, mostly copy pasted from fastai's git repo, with some adjustments to
# handle the transformer's input being frame embeddings and not int encoding of word
# embeddings.
def feed_forward(d_model:int, d_ff:int, ff_p:float=0., act=nn.ReLU(inplace=True), 
                 double_drop:bool=True):
    # Mostly taken from https://github.com/fastai/fastai/blob/master/fastai/text/models/transformer.py
    layers = [nn.Linear(d_model, d_ff), act]
    if double_drop: layers.append(nn.Dropout(ff_p))
    return fastai.layers.SequentialEx(*layers, nn.Linear(d_ff, d_model), 
                                      nn.Dropout(ff_p), fastai.layers.MergeLayer(),
                                      nn.LayerNorm(d_model))

class DecoderLayer(nn.Module):
    # Mostly taken from https://github.com/fastai/fastai/blob/master/fastai/text/models/transformer.py
    def __init__(self, n_heads:int, d_model:int, d_head:int, d_inner:int, 
                 resid_p:float=0., attn_p:float=0., ff_p:float=0.,
                 bias:bool=True, scale:bool=True, act=nn.ReLU(inplace=True), 
                 double_drop:bool=True,attn_cls = MultiHeadAttention):
        super().__init__()
        self.mhra = attn_cls(n_heads, d_model, d_head, 
                             resid_p=resid_p, attn_p=attn_p, 
                             bias=bias, scale=scale)
        self.ff   = feed_forward(d_model, d_inner, ff_p=ff_p,
                                 act=act, double_drop=double_drop)

    def forward(self, x, mask=None, **kwargs): 
        return self.ff(self.mhra(x, mask=mask, **kwargs))

class Transformer(nn.Module):
    # Mostly taken from https://github.com/fastai/fastai/blob/master/fastai/text/models/transformer.py
    def __init__(self, n_layers:int, n_heads:int, d_model:int, d_head:int, d_inner:int,
                 resid_p:float=0.1, attn_p:float=0.1, ff_p:float=0.1, 
                 embed_p:float=0.1, bias:bool=True, scale:bool=True,
                 double_drop:bool=True, attn_cls=MultiHeadAttention):
        super().__init__()
        self.pos_enc = fastai.text.models.transformer.PositionalEncoding(d_model)
        self.drop_emb = nn.Dropout(embed_p)
        self.layers = nn.ModuleList([DecoderLayer(n_heads, d_model, d_head,
                                                  d_inner, resid_p=resid_p, attn_p=attn_p,
                                                  ff_p=ff_p, bias=bias, scale=scale, 
                                                  act=nn.ReLU(inplace=True), double_drop=double_drop,
                                                  attn_cls=attn_cls) 
                                     for k in range(n_layers)])

    def forward(self, x):
        bs, x_len, _ = x.size()
        pos = torch.arange(0, x_len, device=x.device, dtype=x.dtype)
        inp = self.drop_emb(x + self.pos_enc(pos)[None]) 
        for layer in self.layers: inp = layer(inp, mask=None)
        return inp 
