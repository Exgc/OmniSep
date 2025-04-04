import csv

import h5py
import numpy as np
import os
from tqdm import tqdm

def get_filelist(csv_path, feature_mode='imagebind'):
    samples = []
    split=csv_path.split('/')[-1].replace('.csv','.h5')
    with h5py.File(f'/nfs/chengxize.cxz/data/VGGSOUND/{feature_mode}/h5/{split}', 'w') as hf:
        for row in tqdm(csv.reader(open(csv_path, "r"), delimiter=",")):
            filename_audio, filename_dir, total_frames, _ = row
            idx = '_'.join(filename_dir.split('/')[-3:])
            obj = hf.create_group(idx)
            json_obj={}
            audio = np.array(np.load(filename_audio.replace('audio_16000', f'{feature_mode}/audio').replace('.wav', '.npy')))
            obj.create_dataset('audio', data=audio)
            text = np.array(np.load(filename_audio.replace('audio_16000', f'{feature_mode}/text').replace('.wav', '.npy')))
            obj.create_dataset('text', data=text)
            for frame_path in os.listdir(filename_dir):
                filename_frame = os.path.join(filename_dir, frame_path)
                frame = np.array(np.load(filename_frame.replace('frames', f'{feature_mode}/frames').replace('.jpg', '.npy')))
                obj.create_dataset(frame_path,data=frame)

if __name__ == '__main__':
    root='/nfs/chengxize.cxz/projects/clipsep/clipsep/data/vggsound-v1'
    feature_mode='Molecule-0.5-0.8'
    for csv_file in os.listdir(root):
        csv_path = os.path.join(root, csv_file)
        get_filelist(csv_path,feature_mode=feature_mode)
