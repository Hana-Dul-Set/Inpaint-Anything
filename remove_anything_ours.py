import torch
import sys
import argparse
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

from sam_segment import predict_masks_with_sam
from lama_inpaint import inpaint_img_with_lama
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points, get_clicked_point

import os
import json
from tqdm import tqdm

OUTPUT_ERROR_PATH = "./exception_log.txt"
OUTPUT_SKIPPED_PATH = "./skipped_log.txt"

def read_json(file_path):
    with open(file_path) as file:
        data = json.load(file)
        return data


def get_file_names_in_dir(dir_path):
    file_names = os.listdir(dir_path)
    return [file_name for file_name in file_names if os.path.isfile(os.path.join(dir_path, file_name))]


def run_impaint(input_img, point_coords, dilate_kernel_size, output_dir):
    point_labels = [1]
    sam_model_type = "vit_h"
    sam_ckpt = "pretrained_models/sam_vit_h_4b8939.pth"
    lama_config = "lama/configs/prediction/default.yaml"
    lama_ckpt = "pretrained_models/big-lama"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    latest_coords = point_coords
    img = load_img_to_array(input_img)

    masks, _, _ = predict_masks_with_sam(
        img,
        [latest_coords],
        point_labels,
        model_type=sam_model_type,
        ckpt_p=sam_ckpt,
        device=device,
    )
    masks = masks.astype(np.uint8) * 255

    # dilate mask to avoid unmasked edge effect
    if dilate_kernel_size is not None:
        masks = [dilate_mask(mask, dilate_kernel_size) for mask in masks]

    # visualize the segmentation results
    img_stem = Path(input_img).stem
    out_dir = Path(output_dir) / img_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, mask in enumerate(masks):
        # path to the results
        mask_p = out_dir / f"mask_{idx}.png"
        img_points_p = out_dir / f"with_points.png"
        img_mask_p = out_dir / f"with_{Path(mask_p).name}"

        # save the mask
        save_array_to_img(mask, mask_p)

    # inpaint the masked image
    for idx, mask in enumerate(masks):
        mask_p = out_dir / f"mask_{idx}.png"
        img_inpainted_p = out_dir / f"inpainted_with_{Path(mask_p).name}"
        img_inpainted = inpaint_img_with_lama(
            img, mask, lama_config, lama_ckpt, device=device)
        save_array_to_img(img_inpainted, img_inpainted_p)



all_image_names = get_file_names_in_dir('../all_image_files')


file_path = "../pose_dataset_0614.json" 
json_data = read_json(file_path)

def find_dict(dict_list, key, value):
    results = []
    for dictionary in dict_list:
        if key in dictionary and dictionary[key] == value:
            results.append(dictionary)
    return results

no_peoples = []
skipped = []

already_done = os.listdir("../inpaint_results/")
for image_data in tqdm(json_data):
    '''if os.path.splitext(image_name)[0] in already_done:
        skipped.append(image_name)
        continue'''
    image_name = image_data['name']

    keypoints = image_data["keypoints"]

    face = [0, 1, 2, 18] #코, 눈, 눈, 목  
    leftarm = [6, 8, 10]
    rightarm = [5, 7, 9]
    leftleg = [12, 14, 16]
    rightleg = [11, 13, 15]

    point_coords = [keypoints[18][0], keypoints[18][1]]
    point_coords[0] *= int(image_data['size'][0])
    point_coords[1] *= int(image_data['size'][1])
    input_img = f"../all_image_files/{image_name}"
    try:
        run_impaint(input_img, point_coords, 15, "../inpaint_results")
    except Exception as e:
        with open(OUTPUT_ERROR_PATH, mode='a', encoding='utf-8', newline='') as file:
            file.write(input_img + '\n')

print(len(no_peoples))
print(len(skipped))
with open('../no_people.txt','w',encoding='utf-8') as f:
    f.write(str(no_peoples))
with open('../skipped_log.txt','w',encoding='utf-8') as f:
    f.write(str(skipped))