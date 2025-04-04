# **OmniSep: Unified Omni-Modality Sound Separation with Query-Mixup**

**[Xize Cheng](https://salu133445.github.io/)**,  Siqi Zheng,  Zehan Wang,  Minghui Fang,  Ziang Zhang,  Rongjie Huang,  Ziyang Ma,  Shengpeng Ji,  Jialong Zuo,  Tao Jin,  Zhou Zhao  
<br><br>
*To appear at ICLR 2025*  
[[arXiv]](https://arxiv.org/abs/2410.21269) • [[Demo]](https://sony.github.io/CLIPSep)  
<br>

---

✅ TODO

- [x] Release the omnisep training codes.
- [x] Release the code for data preprocess.
- [ ] Release the inference codes.
- [ ] Release the checkpoints.

---

## 🚀 Installation

Clone the repository and set up the environment:

```bash
git clone https://github.com/Exgc/OmniSep.git
cd OmniSep/omnisep

conda create -n omnisep python=3.10
conda activate omnisep

pip install -r requirements.txt
```

---

## 📦 Datasets and Pre-trained Models


### 🎵 MUSIC Dataset  
Please refer to the script under [`dataset/music`](dataset/music).

### 🔊 VGGSound Dataset  
Please refer to the script under [`dataset/vggsound`](dataset/vggsound).

---

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
