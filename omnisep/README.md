# OmniSep PyTroch implementation



## Train
Train CLIPSep-NIT model

> {training modality} 

```bash
python train.py \
    -o exp/vggsound/omnisep \
    -t data/vggsound/train.csv \
    -v data/vggsound/val.csv \
    --batch_size 128 \
    --workers 20 \
    --emb_dim 1024 \
    --image_model imagebind \
    --fusion late \
    --train_mode image text audio \
    --is_feature \
    --feature_mode imagebind \
    --feat_func mixup
```

## Inference

Download the dataset from huggingface. 

[VGGSOUND eval](https://huggingface.co/datasets/Exgc/OmniSep_VGGSOUND_eval) 

[MUSIC eval]().

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
OMP_NUM_THREADS=1 python evaluate.py -o exp/vggsound/imagebindsep_late_hybrid_mixup_3 -l exp/vggsound/clipsep_nit/eval_woPIT_VGGS_VGGSN.txt -t data/vggsound-v1/test-good.csv -t2 data/vggsound-v1/test-no-music.csv --no-pit --prompt_ens --audio_source ./VGGSOUND-aq.npy
