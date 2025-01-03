import sys
sys.path.append("..")
sys.path.append(".")
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import json
from utils import rleToMask, save_image, extract_random_blocks, center_square_crop, crop_to_square_and_save, interpolate_tensor, interpolate_numpy, warp_image_affine, warp_image_twist, random_based_on_time, random_rotate, resize_and_crop_or_pad, resize_single_channel_image, expand_mask, augment_mask
from pycocotools.coco import COCO
from pycocotools import mask
import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F
import cv2
from transformers import AutoImageProcessor, AutoModel
import torch.nn as nn
import copy
import time
import copy
from transformers import CLIPTextModel, CLIPTokenizer
import random

COCO_CLASSNAMES = {
1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorbike', 5: 'aeroplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'trafficlight', 11: 'firehydrant', 12: 'streetsign', 13: 'stopsign', 14: 'parkingmeter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 26: 'hat', 27: 'backpack', 28: 'umbrella', 29: 'shoe', 30: 'eyeglasses', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sportsball', 38: 'kite', 39: 'baseballbat', 40: 'baseballglove', 41: 'skateboard', 42: 'surfboard', 43: 'tennisracket', 44: 'bottle', 45: 'plate', 46: 'wineglass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hotdog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'sofa', 64: 'pottedplant', 65: 'bed', 66: 'mirror', 67: 'diningtable', 68: 'window', 69: 'desk', 70: 'toilet', 71: 'door', 72: 'tvmonitor', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cellphone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'subrigerator', 83: 'blender', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddybear', 89: 'hairdrier', 90: 'toothbrush', 91: 'hairbrush'
}

def pad_to_square(image):
    H, W, _ = image.shape
    new_size = max(W, H)
    # 创建一个白色背景的图片
    new_image = np.ones((new_size, new_size, 3), dtype=np.uint8) * 255
    
    x_start = (new_size - W) // 2
    y_start = (new_size - H) // 2
    
    new_image[y_start:y_start+H, x_start:x_start+W, :] = image
    
    return new_image

def resize_bboxes(bboxes, orig_size, target_size):
    # Calculate scaling factors for width and height
    scale_w = target_size[1] / orig_size[1]
    scale_h = target_size[0] / orig_size[0]

    # Resize bounding boxes
    resized_bboxes = []
    for bbox in bboxes:
        x, y, w, h = bbox
        resized_bbox = [
            x * scale_w,  # Scale x position
            y * scale_h,  # Scale y position
            w * scale_w,  # Scale width
            h * scale_h   # Scale height
        ]
        resized_bboxes.append(resized_bbox)

    return resized_bboxes


class DatasetCOCO(Dataset):
    def __init__(self, data_paths, transform=None, max_len=4, tokenizer=None, train_split='val', args=None, subset_size=None):
        """_summary_
        Args:
            data_paths (_type_): _description_
            transform (_type_, optional): _description_. Defaults to None.
            
            COCO folder structure:
                input data_paths
                - train2014
                    -- COCO_train2014_000000XXXXXX.jpg
                    ...
                - val2014
                    -- COCO_val2014_000000XXXXXX.jpg
                    ...
                - annotations
                    -- captions_train2014.json
                    -- captions_val2014.json
                    -- instances_train2014.json
                    -- instances_val0214.json
                    ... and keypoints
        """
        self.train_split = train_split
        self.data_paths = data_paths
        self.transform = transform
        self.max_len = max_len
        self.data = self.load_data()
        self.tokenizer = tokenizer
        self.args = args
        self.subset_size = subset_size
        
        
    def load_data(self):
        """
        Load coco caption pairs.
        return pair dict
        pairs[pairs_count] = [img_path, [height, width], segmaps, bboxes]
        """
        pairs = {}
        pairs_count = 0
        
        if 'train' in self.train_split:
            print("== COCO train 2014, using training set! ==")
            train_coco_cap = COCO(os.path.join(self.data_paths, "annotations/captions_train2014.json"))
            train_coco_ins = COCO(os.path.join(self.data_paths, "annotations/instances_train2014.json"))
            
            for key in train_coco_cap.imgs.keys():
                cap_anns_ids = train_coco_cap.getAnnIds(imgIds=train_coco_cap.imgs[key]['id'],iscrowd=None)
                inst_anns_ids = train_coco_ins.getAnnIds(imgIds=train_coco_cap.imgs[key]['id'],iscrowd=None)
                    
                cap_anns = train_coco_cap.loadAnns(cap_anns_ids)
                ins_anns = train_coco_ins.loadAnns(inst_anns_ids)
                    
                # get caption for current image
                caption = ""
                for cap in cap_anns:
                    if len(cap['caption']) > len(caption):
                        caption = cap['caption']
                    
                    
                # get path for current image
                img_path = os.path.join(self.data_paths, "train2014", train_coco_cap.imgs[key]['file_name'])
                    
                # get image size
                height = train_coco_cap.imgs[key]['height']
                width = train_coco_cap.imgs[key]['width']
                    
                # get image seg map
                segmaps = [ins_ann['segmentation'] for ins_ann in ins_anns]
                    
                # get image boxes
                bboxes = [ins_ann['bbox'] for ins_ann in ins_anns]
                    
                # get suberence categories
                categories = [COCO_CLASSNAMES[ins_ann["category_id"]] for ins_ann in ins_anns]
                    
                area = [ins_ann['area'] for ins_ann in ins_anns]
                    
                pairs[pairs_count] = [img_path, [height, width], segmaps, bboxes, categories, area, caption]
                pairs_count += 1
            
            del train_coco_cap
            del train_coco_ins
        
        if 'val' in self.train_split:
            print("== COCO train 2014, using validation set! ==")
            val_coco_cap = COCO(os.path.join(self.data_paths, "annotations/captions_val2014.json"))
            val_coco_ins = COCO(os.path.join(self.data_paths, "annotations/instances_val2014.json"))
            for key in val_coco_cap.imgs.keys():
                cap_anns_ids = val_coco_cap.getAnnIds(imgIds=val_coco_cap.imgs[key]['id'],iscrowd=None)
                inst_anns_ids = val_coco_ins.getAnnIds(imgIds=val_coco_cap.imgs[key]['id'],iscrowd=None)
                    
                cap_anns = val_coco_cap.loadAnns(cap_anns_ids)
                ins_anns = val_coco_ins.loadAnns(inst_anns_ids)
                    
                # get caption for current image
                caption = ""
                for cap in cap_anns:
                    if len(cap['caption']) > len(caption):
                        caption = cap['caption']
                            
                    
                # get path for current image
                img_path = os.path.join(self.data_paths, "val2014", val_coco_cap.imgs[key]['file_name'])
                    
                # get image size
                height = val_coco_cap.imgs[key]['height']
                width = val_coco_cap.imgs[key]['width']
                    
                # get image seg map
                segmaps = [ins_ann['segmentation'] for ins_ann in ins_anns]
                    
                # get image boxes
                bboxes = [ins_ann['bbox'] for ins_ann in ins_anns]
                    
                # get suberence categories
                categories = [COCO_CLASSNAMES[ins_ann["category_id"]] for ins_ann in ins_anns]
                    
                area = [ins_ann['area'] for ins_ann in ins_anns]
                    
                pairs[pairs_count] = [img_path, [height, width], segmaps, bboxes, categories, area, caption]
                pairs_count += 1
                
            del val_coco_ins
            del val_coco_cap
        
        return pairs
    
    
    def tokenize_text(self, text):
        inputs = self.tokenizer(
            text, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids
    
    
    
    def create_train_data(self, image, bboxes, categories, areas, rle_masks, h, w, top_n_size = 5):
        sub_imgs = []
        sub_cates = []
        selected_areas = []
        decoded_masks = []
        H, W, C = image.shape
        
        bboxes = resize_bboxes(bboxes, (h, w), (self.args.resolution, self.args.resolution))
        for seg, cate, area, bbox in zip(rle_masks, categories, areas, bboxes):
            if ((len(areas) > self.max_len) and (area not in sorted(areas)[-int(len(areas)//2):]) or (area < 1000)):
                continue
            try:
                rle = mask.merge(mask.frPyObjects(seg, h, w))
                decoded_mask = mask.decode(rle)
            except:
                continue
                
            bbox_x, bbox_y, bbox_w, bbox_h = [int(i) for i in bbox]
            white_image = copy.copy(image)
            white_image[decoded_mask == 0] = np.array([255, 255, 255])
            white_image = cv2.resize(white_image, (self.args.resolution, self.args.resolution))
            masked_sub = white_image[bbox_y:bbox_y + bbox_h, bbox_x:bbox_x + bbox_w, :]
            masked_sub = pad_to_square(masked_sub)
                
            if np.sum(masked_sub) != 0:
                sub_imgs.append(masked_sub)
                sub_cates.append(cate)
                selected_areas.append(area)
                decoded_masks.append(decoded_mask)
                
        return sub_imgs, sub_cates, selected_areas, decoded_masks
    
    

    def __len__(self):
        if self.subset_size is not None:
            return min(self.subset_size, len(self.data))
        
        return len(self.data)

    
    def __getitem__(self, idx):
        assert self.transform
        
        sample = self.data[idx]
        img_path, (h, w), rle_masks, bboxes, categories, areas, caption = sample
        img_pil = Image.open(img_path).convert('RGB')
        tar_image_np = np.array(img_pil)
        
        indexes = sorted(range(len(areas)), key=lambda i: areas[i], reverse=True)
        bboxes = [bboxes[i] for i in indexes]
        categories = [categories[i] for i in indexes]
        areas = [areas[i] for i in indexes]
        rle_masks = [rle_masks[i] for i in indexes]
        
        # # interpolate numpy
        sub_images, sub_cates, selected_area, decoded_masks = self.create_train_data(tar_image_np, bboxes, categories, areas, rle_masks, h, w)
        
        tar_image_np = cv2.resize(tar_image_np, (self.args.resolution, self.args.resolution))
        
        # padding 
        if len(sub_images) == 0:
            # for empty data pairs
            sub_images = [np.zeros((100, 100, 3)) for i in range(self.max_len)]
            sub_cates = ["" for i in range(self.max_len)]
            decoded_masks = [np.zeros(shape=(100, 100)) for i in range(len(sub_images))]
        else:
            sub_images = sub_images[:self.max_len]
            sub_cates = sub_cates[:self.max_len]
            decoded_masks = decoded_masks[:self.max_len]
        
        padding_num = self.max_len - len(sub_images)
        # prepare text and target and suberence images
        subject_prompts = [f"a photo of a {sub_cate}." for sub_cate in sub_cates]
        try:
            if self.transform:
                sub_imgs = [self.transform(Image.fromarray(np.uint8(sub))) for sub in sub_images]
                target = self.transform(Image.fromarray(np.uint8(tar_image_np)))
        except:
            print("Data suberence error!!!")
            if self.transform:
                sub_images = [np.zeros((100, 100, 3)) for i in range(self.max_len)]
                sub_imgs = [self.transform(Image.fromarray(np.uint8(sub))) for sub in sub_images]
                target = self.transform(Image.fromarray(np.uint8(tar_image_np)))
                subject_prompts = ["" for i in range(self.max_len)]
        pil_sub_images = [Image.fromarray(np.uint8(sub)) for sub in sub_images]
        sub_imgs = torch.stack(sub_imgs)
        
        # prepare suberence mask
        try:
            for i in range(len(decoded_masks)):
                cur_mask = decoded_masks[i]
                cur_mask = resize_single_channel_image(cur_mask, (64, 64)) * 255
                decoded_masks[i] = cur_mask
        except:
            breakpoint()
        
        text_ids = self.tokenize_text(caption)
        sub_text_ids = self.tokenize_text(subject_prompts)
        
        # return sample
        sample = {
            "target_image": target,
            "input_ids": text_ids,
            "subject_input_ids": sub_text_ids,
            "subject_images": sub_imgs,
            "sub_masks": decoded_masks,
            "padding_num": padding_num,
            "img_per_batch": sub_imgs.shape[0],
            "dataset_name": "coco2014",
            "subject_prompt": subject_prompts,
            "target_prompt": caption,
            "pil_sub_images": pil_sub_images,
        }
        
        return sample


if __name__ == "__main__":
    np.random.seed(0)
    
    transform = T.Compose(
        [
            T.Resize((512, 512)),
            # T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ]
    )
    
    tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="tokenizer", local_files_only=True)
    
    coco_data = DatasetCOCO("data/coco2014", transform=transform, tokenizer=tokenizer)
    # coco_data.load_data()
    
    coco_data[10]
    
    
    print()
    pass