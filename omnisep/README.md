# OmniSep PyTroch implementation



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