import os
import csv
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pathlib
import sys


# sys.path.append('/nfs/chengxize.cxz/projects/Molecule/MoleculeSpace')
# from models.paths import *
# from models.uni_spaces import Uni_Spaces, DrvtFTPP_uni, IBPP_uni
def get_model(model_name="imagebind", device='cpu'):
    if model_name == "imagebind":
        from imagebind.models import imagebind_model
        # Instantiate model
        model = imagebind_model.imagebind_huge(pretrained=True)
        model.eval()
        model.to(device)
    elif 'Molecule' in model_name:
        import torch

        model = DrvtFTPP_uni()
        model.eval()
        model.to(device)
    return model


def get_feature(model, obj_list, modality='text', device='cpu', model_name='imagebind'):
    if model_name == 'imagebind':
        from imagebind.models.imagebind_model import ModalityType
        from imagebind import data
        if modality == 'text':
            key = ModalityType.TEXT
            value = data.load_and_transform_text(obj_list, device)
        elif modality == 'image':
            key = ModalityType.VISION
            value = data.load_and_transform_vision_data(obj_list, device)
        elif modality == 'audio':
            key = ModalityType.AUDIO
            value = data.load_and_transform_audio_data(obj_list, device)
        inputs = {key: value}
        with torch.no_grad():
            embeddings = model(inputs)[key]
    else:
        model.text_factor = 0.5
        model.audio_factor = 0.8
        if modality == 'text':
            embeddings = model.emb_texts(obj_list)
        elif modality == 'image':
            embeddings = model.emb_images(obj_list)
        elif modality == 'audio':
            embeddings = model.emb_audios(obj_list)
    return embeddings


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    csv_path = '/nfs/chengxize.cxz/projects/clipsep/clipsep/data/MUSIC-v1/solo/train.csv'
    model_name = "imagebind"
    # model_name = 'Molecule-0.5-0.8'
    batch_size = 100
    model = get_model(model_name, device)

    files = {}
    with open(csv_path) as f:
        for item in tqdm(csv.reader(f)):
            files.setdefault(item[-1], [])
            if len(files[item[-1]]) < 10:
                files[item[-1]].append(item[0])
    audio_files = []
    text_files = []
    for key in files:
        text_files.append(key)
        if len(files[key]) < 10:
            print(len(files[key]), key)
        for audio_path in files[key]:
            audio_files.append(audio_path)


    audio_query = {}
    for key in files:
        file_paths = files[key]
        audio_embedding = get_feature(model, file_paths, modality='audio', device=device, model_name=model_name)
        audio_query.setdefault(key, audio_embedding.cpu())
    np.save('MUSIC-aq.npy', audio_query)

    # total_files = 0
    # text_list=[]
    # for file in os.listdir(root):
    #     csv_path = os.path.join(root, file)
    #     with open(csv_path) as f:
    #         for item in tqdm(csv.reader(f)):
    #             if os.path.exists(item[0].replace('audio',f'{model_name}/text').replace('.wav','.npy')):
    #                 continue
    #             text_list.append(item[-1])
    #             audio_files.append(item[0])
    #             total_files += 1
    # print(total_files)

    # # # text 提取特征
    # for idx in tqdm(range(0,total_files,batch_size)):
    #     text=text_list[idx:min(total_files,idx+batch_size)]
    #     file_paths=audio_files[idx:min(total_files,idx+batch_size)]
    #     text_embeddings = get_feature(model, text, modality='text', device=device,model_name=model_name)
    #     for text_embedding,file_path in zip(text_embeddings,file_paths):
    #         os.makedirs(os.path.dirname(file_path.replace('audio',f'{model_name}/text').replace('.wav','.npy')),exist_ok=True)
    #         if os.path.exists(file_path.replace('audio',f'{model_name}/text').replace('.wav','.npy')):
    #             continue
    #         np.save(file_path.replace('audio',f'{model_name}/text').replace('.wav','.npy'),text_embedding.cpu().numpy())
