# OmniSep: Unified Omni-Modality Sound Separation with Query-Mixup

[Xize Cheng](https://salu133445.github.io/), 
[Siqi Zheng]()*, 
[Zehan Wang](), 
Minghui Fang, 
Ziang Zhang, 
Rongjie Huang, 
Ziyang Ma, 
Shengpeng Ji,
Jialong Zuo, 
Tao Jin, 
Zhou Zhao <br> <br>
To appear at ICLR 2025. <br>

[[arXiv]](https://arxiv.org/abs/2410.21269) [[Demo]](https://sony.github.io/CLIPSep)<br>


## Installation

Clone repository

```bash
git clone https://github.com/Exgc/OmniSep.git
cd OmniSep/omnisep

conda create -n omnisep python==3.10
pip install -r requeriments.txt

git clone https://github.com/facebookresearch/ImageBind.git
cd ImageBind
pip install .

```

## Datasets and pre-trained model 
We provide a script to download datasets used in our paper and the pre-trained networks. The datasets and network checkpoints will be downloaded and stored in the `CLIPSep/clipsep/data` and `CLIPSep/clipsep/exp/vggsound` directories, respectively.

### MUSIC dataset
Please use the script in `OmniSep/music` directory.
### VGGSound dataset
Please use the script in `OmniSep/vggsound` directory.


## Inference

```bash
OMP_NUM_THREADS=1 python infer.py -o exp/vggsound/clipsep_nit/  -i "demo/audio/hvCj8Dk0Su4.wav" --text_query "playing bagpipes" -f "exp/vggsound/clipsep_nit/hvCj8Dk0Su4/playing bagpipes.wav"
```

## Evaluate 

Evaluate on MUSIC + VGGSound  

```bash
OMP_NUM_THREADS=1 python evaluate.py -o exp/vggsound/clipsep_nit/ -l exp/vggsound/clipsep_nit/eval_woPIT_MUISC_VGGS.txt -t data/MUSIC-v1/solo/test.csv -t2 data/vggsound-v1/test-good-no-music.csv --no-pit --prompt_ens
```

Evaluate on VGGSoundClean + VGGSound  

```bash
OMP_NUM_THREADS=1 python evaluate.py -o exp/vggsound/clipsep_nit/ -l exp/vggsound/clipsep_nit/eval_woPIT_VGGS_VGGSN.txt -t data/vggsound-v1/test-good.csv -t2 data/vggsound-v1/test-no-music.csv --no-pit --prompt_ens
```
  
## Citation
If you find this work useful for your research, please cite our paper:

```
@article{cheng2024omnisep,
  title={OmniSep: Unified Omni-Modality Sound Separation with Query-Mixup},
  author={Cheng, Xize and Zheng, Siqi and Wang, Zehan and Fang, Minghui and Zhang, Ziang and Huang, Rongjie and Ma, Ziyang and Ji, Shengpeng and Zuo, Jialong and Jin, Tao and others},
  journal={arXiv preprint arXiv:2410.21269},
  year={2024}
}
```
