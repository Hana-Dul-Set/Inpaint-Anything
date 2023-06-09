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

        # save the pointed and masked image
        dpi = plt.rcParams['figure.dpi']
        height, width = img.shape[:2]
        plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
        plt.imshow(img)
        plt.axis('off')
        show_points(plt.gca(), [latest_coords], point_labels,
                    size=(width*0.04)**2)
        plt.savefig(img_points_p, bbox_inches='tight', pad_inches=0)
        show_mask(plt.gca(), mask, random_color=False)
        plt.savefig(img_mask_p, bbox_inches='tight', pad_inches=0)
        plt.close()

    # inpaint the masked image
    for idx, mask in enumerate(masks):
        mask_p = out_dir / f"mask_{idx}.png"
        img_inpainted_p = out_dir / f"inpainted_with_{Path(mask_p).name}"
        img_inpainted = inpaint_img_with_lama(
            img, mask, lama_config, lama_ckpt, device=device)
        save_array_to_img(img_inpainted, img_inpainted_p)



all_image_names = get_file_names_in_dir('../all_image_files')
print(all_image_names)


file_path = "../all_keypoints_0609.json" 
json_data = read_json(file_path)

def find_dict(dict_list, key, value):
    results = []
    for dictionary in dict_list:
        if key in dictionary and dictionary[key] == value:
            results.append(dictionary)
    return results

no_peoples = []

for image_name in tqdm(all_image_names):
    people_datas = find_dict(json_data, "image_id", image_name)

    if len(people_datas) == 0:
        #print(f"{image_name} has no people")
        no_peoples.append(image_name)
        continue

    scores = []
    for i, data in enumerate(people_datas):
        scores.append((i, data["score"]))
    scores.sort(key=lambda x: x[1], reverse = True)
    main_person = people_datas[scores[0][0]]

    keypoints = main_person["keypoints"]

    face = [0, 1, 2, 18] #코, 눈, 눈, 목  
    leftarm = [6, 8, 10]
    rightarm = [5, 7, 9]
    leftleg = [12, 14, 16]
    rightleg = [11, 13, 15]

    is_valid_photo = True
    arm_count = 0 
    leg_count = 0
    for i in range(0, 26*3, 3):
        x, y = keypoints[i], keypoints[i+1]
        score = keypoints[i+2]

        index = i // 3
        if index in face and score < 0.01:
            is_valid_photo = False
            break
        if index in leftarm+rightarm and score >=0.01:
            arm_count += 1
        if index in leftleg+rightleg and score >=0.01:
            leg_count += 1
    if arm_count < 3 :
        is_valid_photo = False

    if not is_valid_photo:
        print(image_name +" IS NOT VALID")
        continue

    point_coords = [keypoints[18*3], keypoints[18*3+1]]

    input_img = f"../all_image_files/{image_name}"
    run_impaint(input_img, point_coords, 15, "../inpaint_results")

print(len(no_peoples))
with open('../no_people.txt','w',encoding='utf-8') as f:
    f.write(no_peoples)