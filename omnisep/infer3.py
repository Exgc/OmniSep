"""Infer from a trained model."""
import argparse
import logging
import pathlib
import pprint
import random
import sys
import types

import clip
import librosa
import numpy as np
import scipy.io.wavfile
import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision

import omnisep
import utils
from pydub import AudioSegment

from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from torch.nn import functional as F


@utils.resolve_paths
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--in_filename", type=str, help="input filename"
    )
    parser.add_argument(
        "-o", "--out_dir", type=pathlib.Path, help="output directory"
    )
    parser.add_argument(
        "-m", "--mix_filename", type=pathlib.Path, help="mix filename"
    )
    parser.add_argument("--text_query", help="text query")
    parser.add_argument("--image_query", help="image query", default=None)
    parser.add_argument("--audio_query", help="audio query", default=None)
    parser.add_argument("--neg_query", help="text query", default=None)
    parser.add_argument(
        "-f", "--out_filename", type=pathlib.Path, help="output filename"
    )
    parser.add_argument(
        "--model_steps",
        type=int,
        help="step of the trained model to load (default to the best model)",
    )
    parser.add_argument(
        "-l", "--log_filename", type=pathlib.Path, help="log filename"
    )
    parser.add_argument(
        "-a", "--audio_len", default=None, type=int, help="audio length"
    )
    parser.add_argument(
        "--binary",
        action=argparse.BooleanOptionalAction,
        help="whether to binarize the masks",
    )
    parser.add_argument(
        "--threshold", default=0.5, type=float, help="binarization threshold"
    )
    parser.add_argument(
        "--prompt_ens",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="whether to ensemble prompts",
    )

    parser.add_argument(
        "--emb_dim",
        default=512,
        type=int,
        help="audio length (in samples)",
    )

    # Others
    parser.add_argument("--seed", default=1234, type=int, help="manual seed")
    parser.add_argument(
        "--gpus", default=1, type=int, help="number of gpus to use"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="show warnings only"
    )

    return parser.parse_args(args=args, namespace=namespace)


def get_text_prompt(label):
    """Get the text prompt for a label."""
    return f"a photo of {label}"


def get_text_prompts(label):
    """Get the text prompt for a label."""
    return [f"a photo of {label}.", f"a photo of the small {label}.", f"a low resolution photo of a {label}.",
            f"a photo of many {label}."]


def count_parameters(net):
    """Return the number of parameters in a model."""
    return sum(p.numel() for p in net.parameters())


def recover_wav(
        mag_mix,
        phase_mix,
        pred_mask,
        n_fft=1024,
        hop_len=256,
        win_len=1024,
        use_log_freq=True,
        use_binary_mask=True,
):
    """Recover the waveform."""
    # Unwarp log scale
    B = mag_mix.size(0)
    if use_log_freq:
        grid_unwarp = torch.from_numpy(
            utils.warpgrid(B, n_fft // 2 + 1, pred_mask.size(3), warp=False)
        ).to(pred_mask.device)
        pred_mask_linear = F.grid_sample(
            pred_mask, grid_unwarp, align_corners=True
        )
    else:
        pred_mask_linear = pred_mask[0]

    # Convert into numpy arrays
    mag_mix = mag_mix.detach().cpu().numpy()
    phase_mix = phase_mix.detach().cpu().numpy()
    pred_mask = pred_mask.detach().cpu().numpy()
    pred_mask_linear = pred_mask_linear.detach().cpu().numpy()

    # Apply the threshold
    if use_binary_mask:
        pred_mask = (pred_mask > 0.5).astype(np.float32)
        pred_mask_linear = (pred_mask_linear > 0.5).astype(np.float32)

    # Recover predicted audio
    pred_mag = mag_mix[0, 0] * pred_mask_linear[0, 0]
    pred_wav = utils.istft_reconstruction(
        pred_mag,
        phase_mix[0, 0],
        hop_len=hop_len,
        win_len=win_len,
    )

    return pred_wav


def mix_audios(audio_paths, target_sample_rate=16000):
    mixed_audio = None
    for audio_path in audio_paths:
        # 加载音频文件并将采样率调整为目标采样率
        y, sr = librosa.load(audio_path, sr=target_sample_rate)
        # 初始化混合音频
        if mixed_audio is None:
            mixed_audio = np.zeros_like(y[:150000])
        # 混合音频
        mixed_audio += y[:len(mixed_audio)]

    return mixed_audio


def new_clip_forward(self, image=None, text=None):
    """A CLIP forward function that automatically chooses the mode."""
    if image is None and text is None:
        raise ValueError("Either `image` or `text` must be given.")
    if image is None:
        return self.encode_text(text)
    if text is None:
        return self.encode_image(image)
    return self.old_forward(image, text)


def load_data(filename, args):
    # Load the audio
    print(filename)
    print('\n')
    audio_raw, rate = librosa.load(filename, sr=args.audio_rate, mono=True)
    # audio_raw = torch.tensor(audio_raw)

    # Initialize an empty audio array
    audio_len = 65535 * (audio_raw.shape[0] // 65535 + 1) if args.audio_len is None else args.audio_len
    audio = np.zeros(args.audio_len, dtype=np.float32)

    # Repeat if audio is too short
    audio_sec = 1.0 * audio_len / args.audio_rate
    out_audio_len = min(audio_raw.shape[0], audio_len)
    if audio_raw.shape[0] < rate * audio_sec:
        repeats = int(rate * audio_sec / audio_raw.shape[0]) + 1
        audio_raw = np.tile(audio_raw, repeats)

    # Crop N seconds
    # len_raw = audio_raw.shape[0]
    # center = args.audio_len // 2
    # start = max(0, center - args.audio_len // 2)
    # end = min(len_raw, center + args.audio_len // 2)
    # audio[
    #     (args.audio_len // 2 - (center - start)) : (
    #         args.audio_len // 2 + (end - center)
    #     )
    # ] = audio_raw[start:end]
    audio = audio_raw[:audio_len]

    # Clip the audio to [-1, 1]
    audio = np.clip(audio, -1, 1)

    # Compute STFT
    spec_mix = librosa.stft(
        audio,
        n_fft=args.n_fft,
        hop_length=args.hop_len,
        win_length=args.win_len,
    )

    # Compute magnitude and phase mixture
    mag_mix = torch.tensor(np.abs(spec_mix)).unsqueeze(0).unsqueeze(0)
    phase_mix = torch.tensor(np.angle(spec_mix)).unsqueeze(0).unsqueeze(0)

    return mag_mix, phase_mix, out_audio_len


def main(args):
    """Main function."""
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Get the device
    device = torch.device("cuda")
    # print(device)
    # Create the model
    logging.info(f"Creating the model...")

    model = omnisep.OmniSep(
        args.n_mix,
        args.layers,
        args.channels,
        use_log_freq=args.log_freq,
        use_weighted_loss=args.weighted_loss,
        use_binary_mask=args.binary_mask,
        emb_dim=args.emb_dim
    )
    model = torch.nn.DataParallel(model, device_ids=range(args.gpus))
    model.to(device)
    model.eval()
    # Create the image model
    imagebind_net = imagebind_model.imagebind_huge(pretrained=True)
    imagebind_net = torch.nn.DataParallel(imagebind_net, device_ids=range(args.gpus))
    imagebind_net.to(device)
    imagebind_net.eval()

    # Load the checkpoint
    checkpoint_dir = args.out_dir / "checkpoints"
    # checkpoint_dir = args.out_dir 
    if args.model_steps is None:
        checkpoint_filename = checkpoint_dir / "best_model.pt"
    else:
        checkpoint_filename = checkpoint_dir / f"model_{args.model_steps}.pt"
    model.load_state_dict(torch.load(checkpoint_filename, map_location=device))
    logging.info(f"Loaded the model weights from: {checkpoint_filename}")
    model.eval()

    # Star visualization
    logging.info("Inferring...")

    # Start evaluation

    with torch.no_grad():
        mag_mix, phase_mix, out_audio_len = load_data(args.in_filename, args)
        mag_mix = mag_mix.to(device)
        phase_mix = phase_mix.to(device)

        # Compute image embedding
        img_emb = []

        inputs = {}
        if args.image_query is not None and args.text_query is not None and args.audio_query is not None:
            inputs.setdefault(ModalityType.TEXT, data.load_and_transform_text([args.text_query], device))
            inputs.setdefault(ModalityType.VISION, data.load_and_transform_vision_data([args.image_query], device))
            inputs.setdefault(ModalityType.AUDIO, data.load_and_transform_audio_data([args.audio_query], device))
            embeddings = F.normalize((imagebind_net(inputs)[ModalityType.VISION] + imagebind_net(inputs)[
                ModalityType.TEXT] * 0 + 3 * imagebind_net(inputs)[ModalityType.AUDIO]) / 54)
        elif args.text_query is not None:
            inputs.setdefault(ModalityType.TEXT, data.load_and_transform_text([args.text_query], device))
            embeddings = F.normalize(imagebind_net(inputs)[ModalityType.TEXT])
        elif args.audio_query is not None:
            inputs.setdefault(ModalityType.AUDIO, data.load_and_transform_audio_data([args.audio_query], device))
            embeddings = F.normalize(imagebind_net(inputs)[ModalityType.AUDIO])
        else:
            inputs.setdefault(ModalityType.VISION, data.load_and_transform_vision_data([args.image_query], device))
            embeddings = F.normalize(imagebind_net(inputs)[ModalityType.VISION])

        if args.neg_query is not None:
            inputs = {}
            inputs.setdefault(ModalityType.AUDIO, data.load_and_transform_audio_data([args.neg_query], device))
            neg_embeddings = F.normalize(imagebind_net(inputs)[ModalityType.AUDIO])
            embeddings = 1.5 * embeddings - 0.5 * neg_embeddings

        img_emb.append(embeddings.type(mag_mix.dtype))
        # if "clip" in args.image_model and args.text_query is not None:
        #     text_inputs = torch.cat([clip.tokenize(args.text_query)])
        #     img_emb.append(clip_net(text=text_inputs).type(mag_mix.dtype))
        # else:
        #     raise RuntimeError("Not implemented!")

        # Forward pass
        if args.binary is not None:
            use_binary_mask = args.binary
        else:
            use_binary_mask = args.binary_mask
        pred_mask = model.module.infer(mag_mix, img_emb)[0]

        # Recover the waveform
        pred_wav = recover_wav(
            mag_mix,
            phase_mix,
            pred_mask,
            n_fft=args.n_fft,
            hop_len=args.hop_len,
            win_len=args.win_len,
            use_log_freq=args.log_freq,
            use_binary_mask=use_binary_mask,
        )

        # Save the audio
        pathlib.Path(args.out_filename).parent.mkdir(exist_ok=True, parents=True)
        scipy.io.wavfile.write(args.out_filename, args.audio_rate, pred_wav[:out_audio_len])
        logging.info(f"Saved the output audio to {args.out_filename}.")


if __name__ == "__main__":
    # Parse command-lind arguments
    args = parse_args()

    # Make sure the output directory exists
    args.out_dir.mkdir(exist_ok=True)

    # Set up a console logger
    if args.log_filename is None:
        args.log_filename = args.out_dir / "infer.log"
    logging.basicConfig(
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(args.log_filename, "w"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Log command called
    logging.info(f"Running command: python {' '.join(sys.argv)}")

    # Log arguments
    logging.info(f"Using arguments:\n{pprint.pformat(vars(args))}")

    # Save command-line arguments
    utils.save_args(args.out_dir / "infer-args.json", args)
    logging.info(f"Saved arguments to {args.out_dir / 'infer-args.json'}")

    # Load training configurations
    logging.info(
        f"Loading training arguments from: {args.out_dir / 'train-args.json'}"
    )
    train_args = utils.load_json(args.out_dir / "train-args.json")
    logging.info(f"Using loaded arguments:\n{pprint.pformat(train_args)}")
    for key in (
            "audio_rate",
            "n_fft",
            "hop_len",
            "win_len",
            "img_size",
            "fps",
            "n_mix",
            "fusion",
            "channels",
            "layers",
            "frames",
            "stride_frames",
            "binary_mask",
            "loss",
            "weighted_loss",
            "log_freq",
    ):
        setattr(args, key, train_args[key])

    # Handle backward compatibility
    args.image_model = train_args.get("image_model", "clip")
    args.train_mode = train_args.get("train_mode", "image")
    # args.audio_only = train_args.get("audio_only", False)
    args.n_labels = train_args.get("n_labels")
    args.label_map_filename = train_args.get("label_map_filename")
    args.reg_coef = train_args.get("reg_coef", 0)
    args.reg_epsilon = train_args.get("reg_epsilon", 0.1)
    args.reg2_coef = train_args.get("reg2_coef", 0)
    args.reg2_epsilon = train_args.get("reg2_epsilon", 0.5)
    args.emb_dim = train_args.get("emb_dim", 512)
    # Run the main program
    main(args)

"""
OMP_NUM_THREADS=1 python infer2.py -o exp/vggsound/imagebindsep_late_hybrid_mixup/  -i "" "" --text_query "/nfs/chengxize.cxz/data/VGGSOUND/frames/trai" -f "exp/vggsound/infer2/-query.wav"
OMP_NUM_THREADS=1 python infer2.py -o exp/vggsound/imagebindsep_late_hybrid_mixup/  -i "" "" --image_query "/nfs/chengxize.cxz/data/VGGSOUND/frames/train/playing_clarinet/dfA32yXUIHE_000020/000002.jpg" -f "exp/vggsound/infer2/image-query.wav"

OMP_NUM_THREADS=1 python infer3.py -o exp/vggsound/imagebindsep_late_hybrid_mixup_3/ \
    -i "/nfs/chengxize.cxz/data/AUDIOSET/audio/unbalanced_train_segments_part00/Y-7kGYLofPgM.wav" \
    --image_query "/nfs/chengxize.cxz/data/AUDIOSET/frames/unbalanced_train_segments_part00/Y-7kGYLofPgM/000001.jpg" \
    -f "exp/vggsound/infer2/image-query.wav" --neg_query "male speech man speaking"


OMP_NUM_THREADS=1 python infer3.py -o exp/vggsound/imagebindsep_late_hybrid_mixup_3/ \
    -i "/nfs/chengxize.cxz/data/AUDIOSET/audio/unbalanced_train_segments_part00/Y-7kGYLofPgM.wav" \
    --image_query "/nfs/chengxize.cxz/data/AUDIOSET/frames/unbalanced_train_segments_part00/Y-7kGYLofPgM/000001.jpg" \
    -f "exp/vggsound/infer2/image-query.wav" --neg_query "male speech man speaking"

OMP_NUM_THREADS=1 python infer3.py -o exp/vggsound/imagebindsep_late_hybrid_mixup_3/ \
-i "/nfs/chengxize.cxz/projects/clipsep/clipsep/real_sample/sample1/ori.wav" \
--audio_query "/nfs/chengxize.cxz/data/VGGSOUND/audio/test/skateboarding/6kXUG1Zo6VA_000000.wav" \
--text_query "skateboarding" \
--image_query "/nfs/chengxize.cxz/projects/clipsep/clipsep/real_sample/sample1/frames/000005.jpg" \


OMP_NUM_THREADS=1 python infer3.py -o exp/vggsound/imagebindsep_late_hybrid_mixup_3/ \
-i "/nfs/chengxize.cxz/projects/clipsep/clipsep/real_sample/sample1/ori.wav" \
--text_query "skateboarding" \
-f "/nfs/chengxize.cxz/projects/clipsep/clipsep/real_sample/sample1/neg_omnisep_2.wav" \
--neg_query "people cheering"



OMP_NUM_THREADS=1 python infer3.py -o exp/vggsound/imagebindsep_late_hybrid_mixup_3/ \
-i "/nfs/chengxize.cxz/projects/clipsep/clipsep/real_sample/sample2/ori.wav" \
--text_query "male speech man speaking" \
-f "/nfs/chengxize.cxz/projects/clipsep/clipsep/real_sample/sample2/neg_omnisep.wav" \
--neg_query "/nfs/chengxize.cxz/data/VGGSOUND/audio/test/people_cheering/eOwKvuuMl74_000070.wav"

--image_query "/nfs/chengxize.cxz/projects/clipsep/clipsep/real_sample/sample2/frames/000005.jpg" \

--text_query "waterfall burbling" \
--image_query "/nfs/chengxize.cxz/projects/clipsep/clipsep/real_sample/sample2/frames/000005.jpg" \

--audio_query "/nfs/chengxize.cxz/data/VGGSOUND/audio/test/skateboarding/6kXUG1Zo6VA_000000.wav" \


--neg_query "people_clapping"
"ocean burbling"

cat|speech|animal/nfs/chengxize.cxz/data/AUDIOSET/audio/unbalanced_train_segments_part00/Y-7kGYLofPgM.wav,/nfs/chengxize.cxz/data/AUDIOSET/frames/unbalanced_train_segments_part00/Y-7kGYLofPgM,12,/m/01yrx|/m/09x0r|/m/0jbk



OMP_NUM_THREADS=1 python infer3.py -o exp/vggsound/imagebindsep_late_hybrid_mixup_3/ \
-i "/nfs/chengxize.cxz/projects/clipsep/clipsep/real_sample/sample2/ori.wav" \
--text_query "male speech man speaking" \
-f "/nfs/chengxize.cxz/projects/clipsep/clipsep/real_sample/sample4/neg_omnisep.wav" \
--neg_query "/nfs/chengxize.cxz/data/VGGSOUND/audio/test/people_cheering/eOwKvuuMl74_000070.wav"


OMP_NUM_THREADS=1 python infer3.py -o exp/vggsound/imagebindsep_late_hybrid_mixup_3/ \
-i "/nfs/chengxize.cxz/data/VGGSOUND/audio/train/playing_didgeridoo/e2rM750cFGc_000242.wav" \
--image_query "/nfs/chengxize.cxz/data/VGGSOUND/frames/train/playing_didgeridoo/e2rM750cFGc_000242/000005.jpg" \
--neg_query "people_clapping" \
-f "/nfs/chengxize.cxz/projects/clipsep/clipsep/real_sample/sample4/clipsep.wav"


"""
