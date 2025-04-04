import os
import csv

import argparse
import pathlib
import random
from tqdm import tqdm
import utils


def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--root_audio",
        type=pathlib.Path,
        help="root for extracted audio files",
    )
    parser.add_argument(
        "-f",
        "--root_frame",
        type=pathlib.Path,
        help="root for extracted video frames",
    )
    parser.add_argument(
        "-c", "--csv_filename", type=pathlib.Path, default='/nfs/chengxize.cxz/VGGSound/vggsound.csv', help="input csv filename",
    )
    parser.add_argument(
        "-o", "--out_dir", type=pathlib.Path, help="output directory"
    )
    parser.add_argument(
        "--fps", default=1, type=int, help="fps of video frames"
    )
    parser.add_argument(
        "--ratio",
        default=0.1,
        type=float,
        help="percentage of the validation set",
    )
    parser.add_argument("--seed", default=1234, type=int, help="manual seed")
    return parser.parse_args(args=args, namespace=namespace)


def main():
    """Main function."""
    # Parse command-line arguments
    args = parse_args()

    # Set random seed
    random.seed(args.seed)

    # Load the input CSV file
    data = utils.load_csv_text(args.csv_filename, True)

    # Construct the label map
    label_map = {youtube_id: text for youtube_id, _, text, _ in data}

    # Find all audio/frames pairs
    infos = {}
    filenames = list(args.root_audio.rglob("*.wav"))
    for filename in tqdm(filenames):
        suffix = filename.with_suffix("").relative_to(args.root_audio)
        frame_dir = args.root_frame / suffix
        # n_frames = len(list(frame_dir.rglob("*.jpg")))
        n_frames = len(os.listdir(frame_dir))
        if n_frames > args.fps * 8:
            youtube_id = filename.stem[:11]
            label = label_map[youtube_id].replace(", ", " ")
            infos.setdefault(youtube_id,(f"{filename},{frame_dir},{n_frames},{label}"))
    print(f"{len(infos)} audio/frames pairs found.")

    root='/nfs/chengxize.cxz/projects/CLIPSep/clipsep/data/vggsound'
    for split_file in os.listdir(root):
        split_name=split_file[:-4]
        file_path=os.path.join(root,split_file)
        file_list=get_youtube_id(file_path)

        subset=[]
        print(split_file)
        for youtube_id in tqdm(file_list):
            if youtube_id in infos:
                subset.append(infos[youtube_id])
                filename = args.out_dir / f"{split_file}"
                with open(filename, "w") as f:
                    for item in subset:
                        f.write(item + "\n")
                # print(f"{len(subset)} items saved to {filename}.")
            else:
                print(youtube_id)

    print("Done!")

def get_youtube_id(file_path):
    id_list=[]
    with open(file_path) as f:
        for item in csv.reader(f):
            youtube_id=item[0][-15:-4]
            id_list.append(youtube_id)
    return id_list


if __name__ == "__main__":
    main()


    """
    python form_filelist.py \
        -a /nfs/chengxize.cxz/data/VGGSOUND/audio \
        -f /nfs/chengxize.cxz/data/VGGSOUND/frames \
        -o /nfs/chengxize.cxz/projects/CLIPSep/clipsep/data/vggsound-v1

    """
