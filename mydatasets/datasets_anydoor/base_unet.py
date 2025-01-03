import sys
sys.path.append("..")
sys.path.append(".")
import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import cv2
from utils import * 
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import albumentations as A
import time
import copy
from utils import rleToMask, save_image, extract_random_blocks, center_square_crop, crop_to_square_and_save, interpolate_tensor, interpolate_numpy, warp_image_affine, warp_image_twist, random_based_on_time, random_rotate, resize_and_crop_or_pad, expand_mask, augment_mask

class BaseDataset_unet(Dataset):
    def __init__(self):
        image_mask_dict = {}
        self.data = []

    def __len__(self):
        # We adjust the ratio of different dataset by setting the length.
        pass

    
    def aug_data_back(self, image):
        transform = A.Compose([
            A.ColorJitter(p=0.5, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            A.ChannelShuffle()
            ])
        transformed = transform(image=image.astype(np.uint8))
        transformed_image = transformed["image"]
        return transformed_image
    
    def aug_data_mask(self, image, mask):
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            # A.RandomBrightnessContrast(p=0.5),
            #A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT,  value=(0,0,0)),
            ])

        transformed = transform(image=image.astype(np.uint8), mask = mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        return transformed_image, transformed_mask


    def check_region_size(self, image, yyxx, ratio, mode = 'max'):
        pass_flag = True
        H,W = image.shape[0], image.shape[1]
        H,W = H * ratio, W * ratio
        y1,y2,x1,x2 = yyxx
        h,w = y2-y1,x2-x1
        if mode == 'max':
            if h > H or w > W:
                pass_flag = False
        elif mode == 'min':
            if h < H or w < W:
                pass_flag = False
        return pass_flag


    def __getitem__(self, idx):
        while(True):
            try:
                idx = np.random.randint(0, len(self.data)-1)
                item = self.get_sample(idx)
                return item
            except:
                idx = np.random.randint(0, len(self.data)-1)
                
    def get_sample(self, idx):
        # Implemented for each specific dataset
        pass

    def sample_timestep(self, max_step =1000):
        if np.random.rand() < 0.3:
            step = np.random.randint(0,max_step)
            return np.array([step])

        if self.dynamic == 1:
            # coarse videos
            step_start = max_step // 2
            step_end = max_step
        elif self.dynamic == 0:
            # static images
            step_start = 0 
            step_end = max_step // 2
        else:
            # fine multi-view images/videos/3Ds
            step_start = 0
            step_end = max_step
        step = np.random.randint(step_start, step_end)
        return np.array([step])

    def check_mask_area(self, mask):
        H,W = mask.shape[0], mask.shape[1]
        ratio = mask.sum() / (H * W)
        if ratio > 0.8 * 0.8  or ratio < 0.1 * 0.1:
            return False
        else:
            return True 
    

    def process_pairs(self, sub_image, sub_mask, tar_image, tar_mask, max_ratio = 0.8, sub_size=224, transforms=None, args = None):
        assert mask_score(sub_mask) > 0.90
        assert self.check_mask_area(sub_mask) == True
        assert self.check_mask_area(tar_mask)  == True

        # ========= suberence ===========
        '''
        # similate the case that the mask for suberence object is coarse. Seems useless :(

        if np.random.uniform(0, 1) < 0.7: 
            sub_mask_clean = sub_mask.copy()
            sub_mask_clean = np.stack([sub_mask_clean,sub_mask_clean,sub_mask_clean],-1)
            sub_mask = perturb_mask(sub_mask, 0.6, 0.9)
            
            # select a fake bg to avoid the background leakage
            fake_target = tar_image.copy()
            h,w = sub_image.shape[0], sub_image.shape[1]
            fake_targe = cv2.resize(fake_target, (w,h))
            fake_back = np.fliplr(np.flipud(fake_target))
            fake_back = self.aug_data_back(fake_back)
            sub_image = sub_mask_clean * sub_image + (1-sub_mask_clean) * fake_back
        '''
        # Get the outline Box of the suberence image
        sub_box_yyxx = get_bbox_from_mask(sub_mask)
        assert self.check_region_size(sub_mask, sub_box_yyxx, ratio = 0.10, mode = 'min') == True
        # Filtering background for the suberence image
        sub_mask_3 = np.stack([sub_mask,sub_mask,sub_mask],-1)
        masked_sub_image = sub_image * sub_mask_3 + np.ones_like(sub_image) * 255 * (1-sub_mask_3)
        y1,y2,x1,x2 = sub_box_yyxx
        masked_sub_image = masked_sub_image[y1:y2,x1:x2,:]
        sub_mask = sub_mask[y1:y2,x1:x2]

        ratio = np.random.randint(11, 15) / 10 
        masked_sub_image, sub_mask = expand_image_mask(masked_sub_image, sub_mask, ratio=ratio)
        sub_mask_3 = np.stack([sub_mask,sub_mask,sub_mask],-1)
        # Padding suberence image to square and resize to 224
        masked_sub_image = pad_to_square(masked_sub_image, pad_value = 255, random = False)
        masked_sub_image = cv2.resize(masked_sub_image.astype(np.uint8), (sub_size,sub_size) ).astype(np.uint8)

        sub_mask_3 = pad_to_square(sub_mask_3 * 255, pad_value = 0, random = False)
        sub_mask_3 = cv2.resize(sub_mask_3.astype(np.uint8), (sub_size,sub_size) ).astype(np.uint8)
        sub_mask = sub_mask_3[:,:,0]
        # Augmenting suberence image
        
        # Getting for high-freqency map
        masked_sub_image_compose, sub_mask_compose =  self.aug_data_mask(masked_sub_image, sub_mask) 
        masked_sub_image_aug = masked_sub_image_compose.copy()

        sub_mask_3 = np.stack([sub_mask_compose,sub_mask_compose,sub_mask_compose],-1)
        sub_image_collage = sobel(masked_sub_image_compose, sub_mask_compose/255)
        
        # ========= Training Target ===========
        tar_box_yyxx = get_bbox_from_mask(tar_mask)
        tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=[1.1,1.2]) #1.1  1.3
        assert self.check_region_size(tar_mask, tar_box_yyxx, ratio = max_ratio, mode = 'max') == True
        
        # Cropping around the target object 
        tar_box_yyxx_crop =  expand_bbox(tar_image, tar_box_yyxx, ratio=[1.3, 3.0])   
        tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop) # crop box
        y1,y2,x1,x2 = tar_box_yyxx_crop
        cropped_target_image = tar_image[y1:y2,x1:x2,:]
        cropped_tar_mask = tar_mask[y1:y2,x1:x2]
        tar_box_yyxx = box_in_box(tar_box_yyxx, tar_box_yyxx_crop)
        y1,y2,x1,x2 = tar_box_yyxx

        # Prepairing collage image
        sub_image_collage = cv2.resize(sub_image_collage.astype(np.uint8), (x2-x1, y2-y1))
        sub_mask_compose = cv2.resize(sub_mask_compose.astype(np.uint8), (x2-x1, y2-y1))
        sub_mask_compose = (sub_mask_compose > 128).astype(np.uint8)

        collage = cropped_target_image.copy() 
        collage[y1:y2,x1:x2,:] = sub_image_collage

        collage_mask = cropped_target_image.copy() * 0.0
        collage_mask[y1:y2,x1:x2,:] = 1.0
        # if np.random.uniform(0, 1) < 0.7: 
        #     cropped_tar_mask = perturb_mask(cropped_tar_mask)
        #     collage_mask = np.stack([cropped_tar_mask,cropped_tar_mask,cropped_tar_mask],-1)

        H1, W1 = collage.shape[0], collage.shape[1]
        cropped_target_image_cropped = cropped_target_image[y1: y2, x1: x2, :]
        cropped_target_image = pad_to_square(cropped_target_image, pad_value = 0, random = False).astype(np.uint8)
        cropped_tar_mask = pad_to_square(np.repeat(cropped_tar_mask[:, :, None], 3, axis=-1), pad_value = 0, random = False).astype(np.uint8)
        collage = pad_to_square(collage, pad_value = 0, random = False).astype(np.uint8)
        collage_mask = pad_to_square(collage_mask, pad_value = 2, random = False).astype(np.uint8)
        
        H2, W2 = collage.shape[0], collage.shape[1]
        # resize everything
        cropped_target_image = cv2.resize(cropped_target_image.astype(np.uint8), (512,512)).astype(np.float32)
        collage = cv2.resize(collage.astype(np.uint8), (512,512)).astype(np.float32)
        collage_mask  = cv2.resize(collage_mask.astype(np.uint8), (512,512),  interpolation = cv2.INTER_NEAREST).astype(np.float32)
        cropped_tar_mask  = cv2.resize(cropped_tar_mask[:, :, 0].astype(np.uint8), (64,64),  interpolation = cv2.INTER_NEAREST).astype(np.uint8)
        
        collage_mask[collage_mask == 2] = -1
        
        expanded_collage_mask = collage_mask
        expanded_collage_mask[expanded_collage_mask == -1] = 0
        tar_bbox_on_padded_image = get_bbox_from_mask(expanded_collage_mask)
        
        masked_sub_image_aug_pil = Image.fromarray(masked_sub_image_aug.astype(np.uint8))
        if type(transforms) != str:
            masked_sub_image_aug = transforms(Image.fromarray(masked_sub_image_aug.astype(np.uint8)))
            cropped_target_image_cropped = transforms(Image.fromarray(cropped_target_image_cropped.astype(np.uint8)))
            cropped_target_image = transforms(Image.fromarray(cropped_target_image.astype(np.uint8)))
            cropped_tar_mask = [expand_mask(np.array(cropped_tar_mask) * 255, max_distance=0, foreground_value=255)]
            
        else:
            if sub_size == 512:
                masked_sub_image_aug = torch.tensor(masked_sub_image_aug  / 127.5 - 1.0)
            elif sub_size == 224:
                masked_sub_image_aug = torch.tensor(masked_sub_image_aug  / 255)
            cropped_target_image = torch.tensor(cropped_target_image / 127.5 - 1.0)
            cropped_target_image_cropped = torch.tensor(cropped_target_image_cropped / 127.5 - 1.0)
            collage = torch.tensor(collage / 127.5 - 1.0)
        
        item = dict(
                subject_images=copy.copy(masked_sub_image_aug.unsqueeze(0)), 
                target_image=copy.copy(cropped_target_image), 
                target_prompt_mask = copy.copy(cropped_tar_mask),
                target_prompt_cropped=copy.copy(cropped_target_image_cropped), 
                hint=copy.copy(collage), 
                extra_sizes=np.array([H1, W1, H2, W2]), 
                tar_box_yyxx_crop=np.array(tar_box_yyxx_crop),
                tar_box_yyxx=np.array(tar_bbox_on_padded_image),
                pil_sub_images=[masked_sub_image_aug_pil],
                ) 
        return item




