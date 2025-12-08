# TODO: Review tools.py, mask.py and remove this file

from mask import Mask
from tools import is_left
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import json

def prediction(mask_a: Mask, mask_b: Mask) -> str:
    return "left" if is_left(mask_a, mask_b) else "right"

def main():
    parser = ArgumentParser()
    parser.add_argument('--annotations_file', type=Path, required=True)
    args = parser.parse_args()

    with open(args.annotations_file, 'r') as f:
        full_annotations = json.load(f)

    lr_annotations = [annotation for annotation in full_annotations if annotation["category"] == "left_right"]
    num_correct = 0

    for annotation in tqdm(lr_annotations):
        mask_a = Mask(annotation["rle"][0])
        mask_b = Mask(annotation["rle"][1])
        if prediction(mask_a, mask_b) == annotation["normalized_answer"]:
            num_correct += 1

    print(f"Correct: {num_correct}/{len(lr_annotations)}")

if __name__ == "__main__":
    main()
