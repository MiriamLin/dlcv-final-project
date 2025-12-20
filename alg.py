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
STR_IMAGE = "image"

STR_LEFT = "left"
STR_RIGHT = "right"

STR_MIDDLE = "middle"
STR_CLOSE = "close"

STR_BUFFER = "buffer"
STR_BUFFERS = "buffers"
STR_PALLET = "pallet"
STR_PALLETS = "pallets"
STR_SHELF = "shelf"
STR_SHELVES = "shelves"
STR_TRANSPORTER = "transporter"
STR_TRANSPORTERS = "transporters"

STR_UNKNOWN = "unknown"

CATS = [STR_BUFFER, STR_PALLET, STR_SHELF, STR_TRANSPORTER]

def cat_plural(cat: str) -> str:
    if cat == STR_BUFFER: return STR_BUFFERS
    if cat == STR_PALLET: return STR_PALLETS
    if cat == STR_SHELF: return STR_SHELVES
    assert cat == STR_TRANSPORTER
    return STR_TRANSPORTERS

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

def categorize_masks(question: str, masks: list[Mask]) -> dict:
    words = question.split()
    
    answer = {cat: [] for cat in CATS + [STR_UNKNOWN]}
    cat_stack: list = []
    was_mask: bool = False
    is_mask: bool = False
    index: int = 0
    for word in words:
        was_mask = is_mask
        if "<mask>" in word:
            is_mask = True
            answer[STR_UNKNOWN if len(cat_stack) == 0 else cat_stack[-1]].append(masks[index])
            index += 1
            continue
        
        is_mask = False
        if was_mask and len(cat_stack) > 0:
            cat_stack.pop()

        for cat in CATS:
            if cat in word or cat_plural(cat) in word:
                cat_stack.append(cat)

    return answer

def select_shelf(question: str, shelf_masks: list[Mask]) -> Mask:
    if len(shelf_masks) == 1:
        return shelf_masks[0]
    if STR_LEFT in question:
        return sorted(shelf_masks)[0]
    assert STR_RIGHT in question
    return sorted(shelf_masks)[-1]

def select_buffer(question: str, buffer_masks: list[Mask], shelf_masks: list[Mask]) -> Mask:
    assert len(buffer_masks) > 0, question
    
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

def select_target(question: str, cat_masks: dict[str, list[Mask]]) -> Mask:
    filtered_cat_masks = {k: v for k, v in cat_masks.items() if len(v) > 0}
    if min([len(x) for x in filtered_cat_masks.values()]) == 1:
        return sorted(filtered_cat_masks.values(), key=lambda l: len(l))[0][0]
    if len(cat_masks[STR_BUFFER]) > 0 and len(cat_masks[STR_SHELF]) > 0:
        return select_shelf(question, cat_masks[STR_SHELF])
    assert len(cat_masks[STR_PALLET]) > 0 and len(cat_masks[STR_TRANSPORTER]) > 0
    return sorted(cat_masks[STR_TRANSPORTER], key=lambda transporter: sum([tools.inclusion(pallet, transporter) for pallet in cat_masks[STR_PALLET]]))[0]

def select_mcq_answer(question: str, masks: list[Mask], dist_model: tools.DistModel, image_name: Path | None, data_dir: Path | None) -> Mask:
    cat_masks = categorize_masks(question, masks)
    filtered_cat_masks = {k: v for k, v in cat_masks.items() if len(v) > 0}
    num_cats = len(filtered_cat_masks)
    if num_cats == 1:
        return select_shelf(question, masks)
    
    assert num_cats == 2
    target = select_target(question, cat_masks)
    candidates = sorted(filtered_cat_masks.values(), key=lambda l: len(l))[1]
    return sorted(candidates, key=lambda x: dist_model.predict(x, target, image_name, data_dir))[0]

def pred_mcq(item: dict, dist_model: tools.DistModel, data_dir: Path | None = None) -> int:
    masks = [Mask(i) for i in item[STR_RLE]]
    question = item[STR_CONVERSATIONS][0][STR_VALUE]
    return masks.index(select_mcq_answer(question, masks, dist_model, item[STR_IMAGE], data_dir))

def pred_dist(item: dict, dist_model: tools.DistModel, data_dir: Path | None = None) -> float:
    image_name = item[STR_IMAGE]
    masks = [Mask(i) for i in item[STR_RLE]]
    if len(masks) == 2:
        return dist_model.predict(
            masks[0],
            masks[1],
            image_name,
            data_dir
        )
    
    question = item[STR_CONVERSATIONS][0][STR_VALUE]
    cat_masks = categorize_masks(question, masks)

    targets: list[Mask] = []
    for cat in CATS:
        if "leftmost " + cat in question or "left most " + cat in question or cat + " on the left" in question:
            targets.append(sorted(cat_masks[cat])[0])
        if "rightmost " + cat in question or "right most " +  cat in question or cat + " on the right" in question:
            targets.append(sorted(cat_masks[cat])[-1])
        if "middle " + cat in question:
            targets.append(sorted(cat_masks[cat])[(len(cat_masks[cat]) + 1) // 2])
    
    if len(targets) == 2:
        return dist_model.predict(
            targets[0],
            targets[1],
            image_name,
            data_dir
        )

    targets += cat_masks[STR_SHELF] + cat_masks[STR_TRANSPORTER]

    assert len(targets) == 2, question

    return dist_model.predict(
        targets[0],
        targets[1],
        image_name,
        data_dir
    )

def pred(item: dict, dist_model: tools.DistModel, data_dir: Path | None = None) -> str | int | float:
    cat = pred_cat(item[STR_CONVERSATIONS][0][STR_VALUE])
    if cat == STR_LEFT_RIGHT:
        return pred_lr(item)
    if cat == STR_COUNT:
        return pred_count(item)
    if cat == STR_MCQ:
        return pred_mcq(item, dist_model, data_dir)
    assert cat == STR_DIST
    return pred_dist(item, dist_model, data_dir)

def main():
    parser = ArgumentParser()
    parser.add_argument("--json_file", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--prediction_file", type=Path, required=True)
    parser.add_argument("--config_path", type=Path, required=True)
    args = parser.parse_args()

    json_file = args.json_file
    with open(json_file, "r") as f:
        data = json.load(f)

    config_path = args.config_path
    dist_model = tools.DistModel(
        config_path
    )

    data_dir = args.data_dir

    predictions = []
    for item in tqdm(data):
        prediction = pred(item, dist_model, data_dir)
        predictions.append({STR_ID: item[STR_ID], STR_NORMALIZED_ANSWER: prediction})

    prediction_file = args.prediction_file
    with open(prediction_file, "w") as f:
        json.dump(predictions, f, indent=4)
    
if __name__ == "__main__":
    main()