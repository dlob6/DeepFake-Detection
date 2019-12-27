#!/bin/bash

# make '/weights' directory if it does not exist and cd into it
mkdir -p weights && cd weights

# copy darknet weight files, continue '-c' if partially downloaded
wget -c https://drive.google.com/open?id=1JpkINv9ISpkydSsHE3osL71ed7g8D_Py
