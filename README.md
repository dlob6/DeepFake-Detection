# DeepFake-Detection
DeepFake detection from videos and audio.
<br> <br>
Preprocessing: <br>
The notebook Extract faces and sound.ipynb preprocesses the video from the kaggle DeepFake dataset. It takes videos as input, and outputs detected faces and STFTs of audio content as .jpg files.
<br> <br>
Deep Fake Detection: <br>
The notebook Model training.ipynb contains code to train the DeepFakeDetector model using pytorch and fastai. <br>
The DeepFakeDetector is composed of several subnetworks. Three subnetworks analyze the video frame by frame:
1. A face analyzer, that takes a face detected in a frame as input and outputs an embedding of it.
2. A STFT analyzer, that takes the STFT of the sound present in the frame as input, and outputs an embedding of it. 
3. A frame analyzer, that takes a concatenation of the face and STFT embeddings, and outputs a joint embedding for the frame, representing the audio and face.

The frame embeddings are concatenated and passed to the last subnetwork of the DeepFakeDetector, that must predict if the video is True or Fake, given the embeddings of the video frames. Prediction errors are backpropagated to each subnetwork.
<br><br>
The DeepFakeDetector model takes the subnetworks as parameters so that different combinations can easily be tested.<br><br>
For illustration purposes the Model training.ipynb gives an example that uses:
* A resnet18 pretrained on Imagenet as face analyzer.
* A simple convnet 2d as STFT analyzer.
* A fully connected network as frame analyzer.
* A fully connected network as video analyzer.
