from argparse import ArgumentParser
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np

import tools
from mask import Mask

STR_DIST = "distance"
STR_MCQ = "mcq"
STR_COUNT = "count"
STR_LEFT_RIGHT = "left_right"

STR_RLE = "rle"
STR_CONVERSATIONS = "conversations"
STR_VALUE = "value"
STR_ID = "id"
STR_NORMALIZED_ANSWER = "normalized_answer"

STR_LEFT = "left"
STR_RIGHT = "right"

STR_MIDDLE = "middle"
STR_CLOSE = "close"

STR_BUFFER = "buffer"
STR_PALLET = "pallet"
STR_SHELF = "shelf"

def pred_cat(question: str) -> str:
    low_question = question.lower()

    KEY_WORDS = [
        (STR_MCQ, {"which"}),
        (STR_COUNT, {"many", "count", "number"}),
        (STR_DIST, {"distance", "distant", "far", "close"}),
    ]
    
    for cat, keys in KEY_WORDS:
        for key in keys:
            if key in low_question:
                return cat
    
    return STR_LEFT_RIGHT

def pred_lr(item: dict) -> str:
    masks = [Mask(i) for i in item[STR_RLE]]
    assert len(masks) == 2
    return STR_LEFT if masks[0] < masks[1] else STR_RIGHT

def categorize_masks(question: str, rle: list) -> dict:
    words = question.split()
    cats = [STR_BUFFER, STR_PALLET, STR_SHELF]
    
    answer = {cat: [] for cat in cats}
    cat_stack: list = []
    was_mask: bool = False
    is_mask: bool = False
    index: int = 0
    for word in words:
        was_mask = is_mask
        if "<mask>" in word:
            is_mask = True
            answer[cat_stack[-1]].append(rle[index])
            index += 1
            continue
        
        is_mask = False
        if was_mask:
            cat_stack.pop()

        for cat in cats:
            if cat in word:
                cat_stack.append(cat)

    return answer

def select_buffer(question: str, buffer_masks: list[Mask], shelf_masks: list[Mask]) -> Mask:
    assert len(buffer_masks) > 0, question
    
    def select_shelf(question: str, shelf_masks: list[Mask]) -> Mask:
        if len(shelf_masks) == 1:
            return shelf_masks[0]
        if STR_LEFT in question:
            return sorted(shelf_masks)[0]
        assert STR_RIGHT in question
        return sorted(shelf_masks)[-1]
    
    if STR_CLOSE in question:
        shelf_mask = select_shelf(question, shelf_masks)
        return sorted(buffer_masks, key=lambda x: tools.proxy_dist(x, shelf_mask))[0]
    
    if STR_LEFT in question:
        return sorted(buffer_masks)[0]

    if STR_RIGHT in question:
        return sorted(buffer_masks)[-1]

    assert STR_MIDDLE in question
    return sorted(buffer_masks)[(len(buffer_masks) + 1) // 2]

def pred_count(item: dict) -> int:
    question: str = item[STR_CONVERSATIONS][0][STR_VALUE]
    cat_masks = categorize_masks(question, [Mask(i) for i in item[STR_RLE]])
    assert len(cat_masks[STR_PALLET]) > 0
    buffer_mask = select_buffer(question, cat_masks[STR_BUFFER], cat_masks[STR_SHELF])
    return np.count_nonzero([tools.inclusion(pallet_mask, buffer_mask) > 0.2 for pallet_mask in cat_masks[STR_PALLET]])

def pred(item: dict) -> str | int | float:
    cat = pred_cat(item[STR_CONVERSATIONS][0][STR_VALUE])
    if cat == STR_LEFT_RIGHT:
        return pred_lr(item)
    if cat == STR_COUNT:
        return pred_count(item)
    if cat == STR_MCQ:
        # TODO: Finish multiple choice questions
        return 1
    assert cat == STR_DIST
    # TODO: Attach distance model
    return 5.0

def main():
    parser = ArgumentParser()
    parser.add_argument("--json_file", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--prediction_file", type=Path, required=True)
    args = parser.parse_args()

    json_file = args.json_file
    with open(json_file, "r") as f:
        data = json.load(f)

    predictions = []
    for item in tqdm(data):
        prediction = pred(item)
        predictions.append({STR_ID: item[STR_ID], STR_NORMALIZED_ANSWER: prediction})

    prediction_file = args.prediction_file
    with open(prediction_file, "w") as f:
        json.dump(predictions, f, indent=4)
    
main()