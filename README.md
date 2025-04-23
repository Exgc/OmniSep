# **[ICLR 2025] OmniSep: Unified Omni-Modality Sound Separation with Query-Mixup**

**[Xize Cheng](https://exgc.github.io)**,  Siqi Zheng,  Zehan Wang,  Minghui Fang,  Ziang Zhang,  Rongjie Huang,  Ziyang Ma,  Shengpeng Ji,  Jialong Zuo,  Tao Jin,  Zhou Zhao  

[arXiv](https://arxiv.org/abs/2410.21269) ｜[Demo](https://omnisep.github.io/)｜[OpenReviewer](https://openreview.net/forum?id=DkzZ1ooc7q)

---

✅ TODO

- [x] Release the omnisep training codes.
- [x] Release the code for data preprocess.
- [ ] Release the inference codes.
- [ ] Release the checkpoints.
- [ ] Implementation based on CLAPSep.

---

## 📦 Data Preparation

### 🎵 MUSIC Dataset  
Please refer to the script under [`dataset/music`](dataset/music).

### 🔊 VGGSound Dataset  
Please refer to the script under [`dataset/vggsound`](dataset/vggsound).

## 🚀 Installation

Clone the repository and set up the environment:

```bash
git clone https://github.com/Exgc/OmniSep.git
cd OmniSep/omnisep

conda create -n omnisep python=3.10
conda activate omnisep

pip install -r requirements.txt
```


## Training

```bash
python train.py \
    -o exp/vggsound/omnisep \
    -t data/vggsound/train.csv \
    -v data/vggsound/val.csv \
    --batch_size 128 \
    --workers 20 \
    --emb_dim 1024 \
    --train_mode image text audio \
    --is_feature \
    --feature_mode imagebind
```


## Evaluate 

Download the eval set of [MUSIC]() + [VGGSound](https://huggingface.co/datasets/Exgc/OmniSep_VGGSOUND_eval).

Evaluate on MUSIC and VGGSound.
```bash
OMP_NUM_THREADS=1 python evaluate.py -o exp/vggsound/omnisep/ -l exp/vggsound/omnisep/eval_MUISC_VGGS.txt -t data/MUSIC/solo/test.csv -t2 data/vggsound/test-good-no-music.csv --no-pit --prompt_ens
```

Evaluate on VGGSoundClean + VGGSound. 

```bash
OMP_NUM_THREADS=1 python evaluate.py -o exp/vggsound/omnisep -l exp/vggsound/omnisep/eval_VGGS_VGGSN.txt -t data/vggsound/test-good.csv -t2 data/vggsound/test-no-music.csv --no-pit --prompt_ens --audio_source ./VGGSOUND-aq.npy
```

## Inference

```bash
OMP_NUM_THREADS=1 python infer.py -o exp/vggsound/clipsep_nit/  -i "demo/audio/hvCj8Dk0Su4.wav" --text_query "playing bagpipes" -f "exp/vggsound/clipsep_nit/hvCj8Dk0Su4/playing bagpipes.wav"
```

## 🏃 Training and Inference

Training and inference scripts are provided in the [`omnisep`](omnisep) directory.

---

## 📄 Citation

If you find this work useful for your research, please consider citing:

```bibtex
@article{cheng2024omnisep,
  title={OmniSep: Unified Omni-Modality Sound Separation with Query-Mixup},
  author={Cheng, Xize and Zheng, Siqi and Wang, Zehan and Fang, Minghui and Zhang, Ziang and Huang, Rongjie and Ma, Ziyang and Ji, Shengpeng and Zuo, Jialong and Jin, Tao and others},
  journal={arXiv preprint arXiv:2410.21269},
  year={2024}
}
```
