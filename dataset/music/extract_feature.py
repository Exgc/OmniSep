import os
import csv
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pathlib
import sys
import utils
import argparse


@utils.resolve_paths
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Extract frames from videos.")
    parser.add_argument(
        "-i", "--in_dir", type=pathlib.Path, help="input directory"
    )

    return parser.parse_args(args=args, namespace=namespace)


def get_model(device='cpu'):
    from imagebind.models import imagebind_model
    # Instantiate model
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    return model


def get_feature(model,obj_list,modality='text',device='cpu',model_name='imagebind'):
    if model_name=='imagebind':
        from imagebind.models.imagebind_model import ModalityType
        from imagebind import data
        if modality=='text':
            key=ModalityType.TEXT
            value=data.load_and_transform_text(obj_list, device)
        elif modality=='image':
            key=ModalityType.VISION
            value=data.load_and_transform_vision_data(obj_list, device)
        elif modality=='audio':
            key=ModalityType.AUDIO
            value=data.load_and_transform_audio_data(obj_list, device)
        inputs={key:value}
        with torch.no_grad():
            embeddings = model(inputs)[key]
    return embeddings
    

if __name__ == "__main__":
    args = parse_args()

    device = "cuda:0"

    model_name = "imagebind"
    batch_size = 100
    audio_files = []
    total_files = 0
    model = get_model(model_name, device)

    # # # check feature exists
    # # # audio
    total_files = 0
    text_list=[]
    for file in os.listdir(args.in_dir):
        csv_path = os.path.join(args.in_dir, file)
        with open(csv_path) as f:
            for item in tqdm(csv.reader(f)):
                if os.path.exists(item[0].replace('audio',f'{model_name}/audio').replace('.wav','.npy')):
                    continue
                text_list.append(item[-1])
                audio_files.append(item[0])
                total_files += 1
    print(total_files)
    
    ## extract audio features
    for idx in tqdm(range(0,total_files,batch_size)):
        file_paths=audio_files[idx:min(total_files,idx+batch_size)]
        audio_embeddings = get_feature(model, file_paths, modality='audio', device=device,model_name=model_name)
        for audio_embedding,file_path in zip(audio_embeddings,file_paths):
            os.makedirs(os.path.dirname(file_path.replace('audio',f'{model_name}/audio').replace('.wav','.npy')),exist_ok=True)
            if os.path.exists(file_path.replace('audio',f'{model_name}/audio').replace('.wav','.npy')):
                continue
            np.save(file_path.replace('audio',f'{model_name}/audio').replace('.wav','.npy'),audio_embedding.cpu().numpy())


    total_files = 0
    text_list=[]
    for file in os.listdir(args.in_dir):
        csv_path = os.path.join(args.in_dir, file)
        with open(csv_path) as f:
            for item in tqdm(csv.reader(f)):
                if os.path.exists(item[0].replace('audio',f'{model_name}/text').replace('.wav','.npy')):
                    continue
                text_list.append(item[-1])
                audio_files.append(item[0])
                total_files += 1
    print(total_files)

    # # extract text feature
    for idx in tqdm(range(0,total_files,batch_size)):
        text=text_list[idx:min(total_files,idx+batch_size)]
        file_paths=audio_files[idx:min(total_files,idx+batch_size)]
        text_embeddings = get_feature(model, text, modality='text', device=device,model_name=model_name)
        for text_embedding,file_path in zip(text_embeddings,file_paths):
            os.makedirs(os.path.dirname(file_path.replace('audio',f'{model_name}/text').replace('.wav','.npy')),exist_ok=True)
            if os.path.exists(file_path.replace('audio',f'{model_name}/text').replace('.wav','.npy')):
                continue
            np.save(file_path.replace('audio',f'{model_name}/text').replace('.wav','.npy'),text_embedding.cpu().numpy())


    # image
    total_files = 0
    frames_files=[]
    for file in os.listdir(args.in_dir):
        csv_path = os.path.join(args.in_dir, file)
        with open(csv_path) as f:
            for item in tqdm(csv.reader(f)):
                for frame_path in os.listdir(item[1]):
                    if os.path.exists(os.path.join(item[1],frame_path).replace('frames',f'{model_name}/frames').replace('.jpg','.npy')):
                        continue
                    frames_files.append(os.path.join(item[1],frame_path))
                    total_files += 1
    print(total_files)

    # # extract image feature
    for idx in tqdm(range(0,total_files,batch_size)):

        file_paths=frames_files[idx:min(total_files,idx+batch_size)]
        frame_embeddings = get_feature(model, file_paths, modality='image', device=device,model_name=model_name)
        for frame_embedding,file_path in zip(frame_embeddings,file_paths):
            os.makedirs(os.path.dirname(file_path.replace('frames',f'{model_name}/frames').replace('.jpg','.npy')),exist_ok=True)
            if os.path.exists(file_path.replace('frames',f'{model_name}/frames').replace('.jpg','.npy')):
                continue
            np.save(file_path.replace('frames',f'{model_name}/frames').replace('.jpg','.npy'),frame_embedding.cpu().numpy())



