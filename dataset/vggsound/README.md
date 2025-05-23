# Downloading the VGGSound Dataset

This folder contains the scripts for downloading the VGGSound dataset. The CSV file is downloaded from the original [repository](https://www.robots.ox.ac.uk/~vgg/data/vggsound/).

## Prerequisites

Shuffle and split the CSV file as follows.
Put `vggsound.csv` into your data directory, e.g. `dataset/vggsound`.

Install packeages
```sh
pip install youtube_dl tqdm pafy
```



## Download the dataset

Run the following script over all the CSV files.

```sh
python download_ffmpeg.py -e -s -i data/vggsound/vggsound-shuf-00.csv -o data/vggsound/video/00/
```

Extract audio from videos
```sh
python extract_audio.py -i data/vggsound/video -o data/vggsound/audio -s -e
```
Extract image frames from videos
```sh
python extract_frames.py -i data/vggsound/video -o data/vggsound/frames -s -e
```
Resize and crop images
```sh
python preprocess.py -i data/vggsound/frames -o data/vggsound/preprocessed -s -e
```