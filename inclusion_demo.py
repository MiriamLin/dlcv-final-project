# TODO: Review tools.py, mask.py and remove this file

from mask import Mask
from tools import inclusion
from argparse import ArgumentParser
from pathlib import Path
import json

def main():
    parser = ArgumentParser()
    parser.add_argument('--annotations_file', type=Path, required=True)
    parser.add_argument('--id', type=str, required=True)
    args = parser.parse_args()

    with open(args.annotations_file, 'r') as f:
        full_annotations = json.load(f)

    annotation = [annotation for annotation in full_annotations if annotation["id"] == args.id][0]

    rle = annotation["rle"]
    l = len(rle)

    for i in range(l):
        for j in range(l):
            if i == j: continue
            r = inclusion(Mask(rle[j]), Mask(rle[i]))
            if r > 0:
                print(f"{j} is {r:.2f} inside {i}.")

if __name__ == "__main__":
    main()
