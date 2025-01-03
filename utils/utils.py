import torch
import numpy as np
import copy
from scipy.ndimage import zoom
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
import cv2
import datetime
import subprocess
import os
import math
import numpy as np
from PIL import Image
import math
import torch
import gc
import torch.nn.functional as F
from scipy.ndimage import zoom
from scipy.ndimage import binary_dilation
from diffusers.utils import deprecate
from diffusers.models.attention_processor import (
    Attention,
    AttnProcessor,
    AttnProcessor2_0,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0
)
import PIL
from diffusers import AutoencoderKL, DDPMScheduler
import PIL
from scipy.ndimage import distance_transform_edt
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
attn_maps = {}
import torch.nn as nn
import clip

import numpy as np
import torch 
import cv2


def mask_score(mask):
    '''Scoring the mask according to connectivity.'''
    mask = mask.astype(np.uint8)
    if mask.sum() < 10:
        return 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt_area = [cv2.contourArea(cnt) for cnt in contours]
    conc_score = np.max(cnt_area) / sum(cnt_area)
    return conc_score


def sobel(img, mask, thresh = 50):
    '''Calculating the high-frequency map.'''
    H,W = img.shape[0], img.shape[1]
    img = cv2.resize(img,(256,256))
    mask = (cv2.resize(mask,(256,256)) > 0.5).astype(np.uint8)
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.erode(mask, kernel, iterations = 2)
    
    Ksize = 3
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=Ksize)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=Ksize)
    sobel_X = cv2.convertScaleAbs(sobelx)
    sobel_Y = cv2.convertScaleAbs(sobely)
    scharr = cv2.addWeighted(sobel_X, 0.5, sobel_Y, 0.5, 0)
    scharr = np.max(scharr,-1) * mask    
    
    scharr[scharr < thresh] = 0.0
    scharr = np.stack([scharr,scharr,scharr],-1)
    scharr = (scharr.astype(np.float32)/255 * img.astype(np.float32) ).astype(np.uint8)
    scharr = cv2.resize(scharr,(W,H))
    return scharr


def resize_and_pad(image, box):
    '''Fitting an image to the box region while keeping the aspect ratio.'''
    y1,y2,x1,x2 = box
    H,W = y2-y1, x2-x1
    h,w =  image.shape[0], image.shape[1]
    r_box = W / H 
    r_image = w / h
    if r_box >= r_image:
        h_target = H
        w_target = int(w * H / h) 
        image = cv2.resize(image, (w_target, h_target))

        w1 = (W - w_target) // 2
        w2 = W - w_target - w1
        pad_param = ((0,0),(w1,w2),(0,0))
        image = np.pad(image, pad_param, 'constant', constant_values=255)
    else:
        w_target = W 
        h_target = int(h * W / w)
        image = cv2.resize(image, (w_target, h_target))

        h1 = (H-h_target) // 2 
        h2 = H - h_target - h1
        pad_param =((h1,h2),(0,0),(0,0))
        image = np.pad(image, pad_param, 'constant', constant_values=255)
    return image



def expand_image_mask(image, mask, ratio=1.4):
    h,w = image.shape[0], image.shape[1]
    H,W = int(h * ratio), int(w * ratio) 
    h1 = int((H - h) // 2)
    h2 = H - h - h1
    w1 = int((W -w) // 2)
    w2 = W -w - w1

    pad_param_image = ((h1,h2),(w1,w2),(0,0))
    pad_param_mask = ((h1,h2),(w1,w2))
    image = np.pad(image, pad_param_image, 'constant', constant_values=255)
    mask = np.pad(mask, pad_param_mask, 'constant', constant_values=0)
    return image, mask


def resize_box(yyxx, H,W,h,w):
    y1,y2,x1,x2 = yyxx
    y1,y2 = int(y1/H * h), int(y2/H * h)
    x1,x2 = int(x1/W * w), int(x2/W * w)
    y1,y2 = min(y1,h), min(y2,h)
    x1,x2 = min(x1,w), min(x2,w)
    return (y1,y2,x1,x2)


def get_bbox_from_mask(mask):
    h,w = mask.shape[0],mask.shape[1]

    if mask.sum() < 10:
        return 0,h,0,w
    rows = np.any(mask,axis=1)
    cols = np.any(mask,axis=0)
    y1,y2 = np.where(rows)[0][[0,-1]]
    x1,x2 = np.where(cols)[0][[0,-1]]
    return (y1,y2,x1,x2)


def expand_bbox(mask,yyxx,ratio=[1.2,2.0], min_crop=0):
    y1,y2,x1,x2 = yyxx
    ratio = np.random.randint( ratio[0] * 10,  ratio[1] * 10 ) / 10
    H,W = mask.shape[0], mask.shape[1]
    xc, yc = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    h = ratio * (y2-y1+1)
    w = ratio * (x2-x1+1)
    h = max(h,min_crop)
    w = max(w,min_crop)

    x1 = int(xc - w * 0.5)
    x2 = int(xc + w * 0.5)
    y1 = int(yc - h * 0.5)
    y2 = int(yc + h * 0.5)

    x1 = max(0,x1)
    x2 = min(W,x2)
    y1 = max(0,y1)
    y2 = min(H,y2)
    return (y1,y2,x1,x2)


def box2squre(image, box):
    H,W = image.shape[0], image.shape[1]
    y1,y2,x1,x2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    h,w = y2-y1, x2-x1

    if h >= w:
        x1 = cx - h//2
        x2 = cx + h//2
    else:
        y1 = cy - w//2
        y2 = cy + w//2
    x1 = max(0,x1)
    x2 = min(W,x2)
    y1 = max(0,y1)
    y2 = min(H,y2)
    return (y1,y2,x1,x2)


def pad_to_square(image, pad_value = 255, random = False):
    H,W = image.shape[0], image.shape[1]
    if H == W:
        return image

    padd = abs(H - W)
    if random:
        padd_1 = int(np.random.randint(0,padd))
    else:
        padd_1 = int(padd / 2)
    padd_2 = padd - padd_1

    if H > W:
        pad_param = ((0,0),(padd_1,padd_2),(0,0))
    else:
        pad_param = ((padd_1,padd_2),(0,0),(0,0))

    image = np.pad(image, pad_param, 'constant', constant_values=pad_value)
    return image



def box_in_box(small_box, big_box):
    y1,y2,x1,x2 = small_box
    y1_b, _, x1_b, _ = big_box
    y1,y2,x1,x2 = y1 - y1_b ,y2 - y1_b, x1 - x1_b ,x2 - x1_b
    return (y1,y2,x1,x2 )



def shuffle_image(image, N):
    height, width = image.shape[:2]
    
    block_height = height // N
    block_width = width // N
    blocks = []
    
    for i in range(N):
        for j in range(N):
            block = image[i*block_height:(i+1)*block_height, j*block_width:(j+1)*block_width]
            blocks.append(block)
    
    np.random.shuffle(blocks)
    shuffled_image = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(N):
        for j in range(N):
            shuffled_image[i*block_height:(i+1)*block_height, j*block_width:(j+1)*block_width] = blocks[i*N+j]
    return shuffled_image


def get_mosaic_mask(image, fg_mask, N=16, ratio = 0.5):
    ids = [i for i in range(N * N)]
    masked_number = int(N * N * ratio)
    masked_id = np.random.choice(ids, masked_number, replace=False)
    

    
    height, width = image.shape[:2]
    mask = np.ones((height, width))
    
    block_height = height // N
    block_width = width // N
    
    b_id = 0
    for i in range(N):
        for j in range(N):
            if b_id in masked_id:
                mask[i*block_height:(i+1)*block_height, j*block_width:(j+1)*block_width] = mask[i*block_height:(i+1)*block_height, j*block_width:(j+1)*block_width] * 0
            b_id += 1
    mask = mask * fg_mask
    mask3 = np.stack([mask,mask,mask],-1).copy().astype(np.uint8)
    noise = q_x(image)
    noise_mask = image * mask3 + noise * (1-mask3)
    return noise_mask

def extract_canney_noise(image, mask, dilate=True):
    h,w = image.shape[0],image.shape[1]
    mask = cv2.resize(mask.astype(np.uint8),(w,h)) > 0.5
    kernel = np.ones((8, 8), dtype=np.uint8)
    mask =  cv2.erode(mask.astype(np.uint8), kernel, 10)

    canny = cv2.Canny(image, 50,100) * mask
    kernel = np.ones((8, 8), dtype=np.uint8)
    mask = (cv2.dilate(canny, kernel, 5) > 128).astype(np.uint8)
    mask = np.stack([mask,mask,mask],-1)

    pure_noise = q_x(image, t=1) * 0 + 255
    canny_noise = mask * image + (1-mask) * pure_noise
    return canny_noise


def get_random_structure(size):
    choice = np.random.randint(1, 5)

    if choice == 1:
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    elif choice == 2:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    elif choice == 3:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size//2))
    elif choice == 4:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size//2, size))

def random_dilate(seg, min=3, max=10):
    size = np.random.randint(min, max)
    kernel = get_random_structure(size)
    seg = cv2.dilate(seg,kernel,iterations = 1)
    return seg

def random_erode(seg, min=3, max=10):
    size = np.random.randint(min, max)
    kernel = get_random_structure(size)
    seg = cv2.erode(seg,kernel,iterations = 1)
    return seg

def compute_iou(seg, gt):
    intersection = seg*gt
    union = seg+gt
    return (np.count_nonzero(intersection) + 1e-6) / (np.count_nonzero(union) + 1e-6)


def select_max_region(mask):
    nums, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    background = 0
    for row in range(stats.shape[0]):
        if stats[row, :][0] == 0 and stats[row, :][1] == 0:
            background = row
    stats_no_bg = np.delete(stats, background, axis=0)
    max_idx = stats_no_bg[:, 4].argmax()
    max_region = np.where(labels==max_idx+1, 1, 0)

    return max_region.astype(np.uint8)



def perturb_mask(gt, min_iou = 0.3,  max_iou = 0.99):
    iou_target = np.random.uniform(min_iou, max_iou)
    h, w = gt.shape
    gt = gt.astype(np.uint8)
    seg = gt.copy()
    
    # Rare case
    if h <= 2 or w <= 2:
        print('GT too small, returning original')
        return seg

    # Do a bunch of random operations
    for _ in range(250):
        for _ in range(4):
            lx, ly = np.random.randint(w), np.random.randint(h)
            lw, lh = np.random.randint(lx+1,w+1), np.random.randint(ly+1,h+1)

            # Randomly set one pixel to 1/0. With the following dilate/erode, we can create holes/external regions
            if np.random.rand() < 0.1:
                cx = int((lx + lw) / 2)
                cy = int((ly + lh) / 2)
                seg[cy, cx] = np.random.randint(2) * 255

            # Dilate/erode
            if np.random.rand() < 0.5:
                seg[ly:lh, lx:lw] = random_dilate(seg[ly:lh, lx:lw])
            else:
                seg[ly:lh, lx:lw] = random_erode(seg[ly:lh, lx:lw])
            
            seg = np.logical_or(seg, gt).astype(np.uint8)
            #seg = select_max_region(seg) 

        if compute_iou(seg, gt) < iou_target:
            break
    seg = select_max_region(seg.astype(np.uint8)) 
    return seg.astype(np.uint8)


def q_x(x_0,t=65):
    '''Adding noise for and given image.'''
    x_0 = torch.from_numpy(x_0).float() / 127.5 - 1
    num_steps = 100
    
    betas = torch.linspace(-6,6,num_steps)
    betas = torch.sigmoid(betas)*(0.5e-2 - 1e-5)+1e-5

    alphas = 1-betas
    alphas_prod = torch.cumprod(alphas,0)
    
    alphas_prod_p = torch.cat([torch.tensor([1]).float(),alphas_prod[:-1]],0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
    
    noise = torch.randn_like(x_0)
    alphas_t = alphas_bar_sqrt[t]
    alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
    return (alphas_t * x_0 + alphas_1_m_t * noise).numpy()  * 127.5 + 127.5 


def extract_target_boundary(img, target_mask):
    Ksize = 3
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=Ksize)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=Ksize)

    # sobel-x
    sobel_X = cv2.convertScaleAbs(sobelx)
    # sobel-y
    sobel_Y = cv2.convertScaleAbs(sobely)
    # sobel-xy
    scharr = cv2.addWeighted(sobel_X, 0.5, sobel_Y, 0.5, 0)
    scharr = np.max(scharr,-1).astype(np.float32)/255
    scharr = scharr *  target_mask.astype(np.float32)
    return scharr

def attn_call(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale=1.0,
    ):
    residual = hidden_states

    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)

    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )
    attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    query = attn.to_q(hidden_states)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

    key = attn.to_k(encoder_hidden_states)
    value = attn.to_v(encoder_hidden_states)

    query = attn.head_to_batch_dim(query)
    key = attn.head_to_batch_dim(key)
    value = attn.head_to_batch_dim(value)

    attention_probs = attn.get_attention_scores(query, key, attention_mask)
    ####################################################################################################
    # (20,4096,77) or (40,1024,77)
    if hasattr(self, "store_attn_map"):
        self.attn_map = attention_probs
    ####################################################################################################
    hidden_states = torch.bmm(attention_probs, value)
    hidden_states = attn.batch_to_head_dim(hidden_states)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if attn.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / attn.rescale_output_factor

    return hidden_states


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias.to(attn_weight.device)
    attn_weight = torch.softmax(attn_weight, dim=-1)

    return torch.dropout(attn_weight, dropout_p, train=True) @ value, attn_weight


def attn_call2_0(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale: float = 1.0,
    ):
    residual = hidden_states

    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)

    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )

    if attention_mask is not None:
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        # scaled_dot_product_attention expects attention_mask shape to be
        # (batch, heads, source_length, target_length)
        attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    query = attn.to_q(hidden_states, scale=scale)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

    key = attn.to_k(encoder_hidden_states, scale=scale)
    value = attn.to_v(encoder_hidden_states, scale=scale)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    # the output of sdp = (batch, num_heads, seq_len, head_dim)
    # TODO: add support for attn.scale when we move to Torch 2.1
    ####################################################################################################
    # if self.store_attn_map:
    if hasattr(self, "store_attn_map"):
        hidden_states, attn_map = scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        # (2,10,4096,77) or (2,20,1024,77)
        self.attn_map = attn_map
    else:
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
    ####################################################################################################

    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
    hidden_states = hidden_states.to(query.dtype)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states, scale=scale)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if attn.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / attn.rescale_output_factor

    return hidden_states


def lora_attn_call(self, attn: Attention, hidden_states, *args, **kwargs):
    self_cls_name = self.__class__.__name__
    deprecate(
        self_cls_name,
        "0.26.0",
        (
            f"Make sure use {self_cls_name[4:]} instead by setting"
            "LoRA layers to `self.{to_q,to_k,to_v,to_out[0]}.lora_layer` respectively. This will be done automatically when using"
            " `LoraLoaderMixin.load_lora_weights`"
        ),
    )
    attn.to_q.lora_layer = self.to_q_lora.to(hidden_states.device)
    attn.to_k.lora_layer = self.to_k_lora.to(hidden_states.device)
    attn.to_v.lora_layer = self.to_v_lora.to(hidden_states.device)
    attn.to_out[0].lora_layer = self.to_out_lora.to(hidden_states.device)

    attn._modules.pop("processor")
    attn.processor = AttnProcessor()

    if hasattr(self, "store_attn_map"):
        attn.processor.store_attn_map = True

    return attn.processor(attn, hidden_states, *args, **kwargs)


def lora_attn_call2_0(self, attn: Attention, hidden_states, *args, **kwargs):
    self_cls_name = self.__class__.__name__
    deprecate(
        self_cls_name,
        "0.26.0",
        (
            f"Make sure use {self_cls_name[4:]} instead by setting"
            "LoRA layers to `self.{to_q,to_k,to_v,to_out[0]}.lora_layer` respectively. This will be done automatically when using"
            " `LoraLoaderMixin.load_lora_weights`"
        ),
    )
    attn.to_q.lora_layer = self.to_q_lora.to(hidden_states.device)
    attn.to_k.lora_layer = self.to_k_lora.to(hidden_states.device)
    attn.to_v.lora_layer = self.to_v_lora.to(hidden_states.device)
    attn.to_out[0].lora_layer = self.to_out_lora.to(hidden_states.device)

    attn._modules.pop("processor")
    attn.processor = AttnProcessor2_0()

    if hasattr(self, "store_attn_map"):
        attn.processor.store_attn_map = True

    return attn.processor(attn, hidden_states, *args, **kwargs)


def cross_attn_ref():
    AttnProcessor.__call__ = attn_call
    AttnProcessor2_0.__call__ = attn_call # attn_call is faster
    # AttnProcessor2_0.__call__ = attn_call2_0
    LoRAAttnProcessor.__call__ = lora_attn_call
    # LoRAAttnProcessor2_0.__call__ = lora_attn_call2_0
    LoRAAttnProcessor2_0.__call__ = lora_attn_call


def reshape_attn_map(attn_map):
    attn_map = torch.mean(attn_map,dim=0) # mean by head dim: (20,4096,77) -> (4096,77)
    attn_map = attn_map.permute(1,0) # (4096,77) -> (77,4096)
    latent_size = int(math.sqrt(attn_map.shape[1]))
    latent_shape = (attn_map.shape[0],latent_size,-1)
    attn_map = attn_map.reshape(latent_shape) # (77,4096) -> (77,64,64)

    return attn_map # torch.sum(attn_map,dim=0) = [1,1,...,1]


def hook_fn(name):
    def forward_hook(module, input, output):
        if hasattr(module.processor, "attn_map"):
            attn_maps[name] = module.processor.attn_map
            del module.processor.attn_map

    return forward_hook

def register_cross_attention_hook(unet):
    for name, module in unet.named_modules():
        if not name.split('.')[-1].startswith('attn2'):
            continue

        if isinstance(module.processor, AttnProcessor):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, AttnProcessor2_0):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, LoRAAttnProcessor):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, LoRAAttnProcessor2_0):
            module.processor.store_attn_map = True

        hook = module.register_forward_hook(hook_fn(name))
    
    return unet


def prompt2tokens(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    tokens = []
    for text_input_id in text_input_ids[0]:
        token = tokenizer.decoder[text_input_id.item()]
        tokens.append(token)
    return tokens


# TODO: generalize for rectangle images
def upscale(attn_map, target_size):
    attn_map = torch.mean(attn_map, dim=0) # (10, 32*32, 77) -> (32*32, 77)
    attn_map = attn_map.permute(1,0) # (32*32, 77) -> (77, 32*32)

    if target_size[0]*target_size[1] != attn_map.shape[1]:
        temp_size = (target_size[0]//2, target_size[1]//2)
        attn_map = attn_map.view(attn_map.shape[0], *temp_size) # (77, 32,32)
        attn_map = attn_map.unsqueeze(0) # (77,32,32) -> (1,77,32,32)

        attn_map = F.interpolate(
            attn_map.to(dtype=torch.float32),
            size=target_size,
            mode='bilinear',
            align_corners=False
        ).squeeze() # (77,64,64)
    else:
        attn_map = attn_map.to(dtype=torch.float32) # (77,64,64)

    attn_map = torch.softmax(attn_map, dim=0)
    attn_map = attn_map.reshape(attn_map.shape[0],-1) # (77,64*64)
    return attn_map


def get_net_attn_map(image_size, batch_size=2, instance_or_negative=False, detach=True):
    target_size = (image_size[0]//16, image_size[1]//16)
    idx = 0 if instance_or_negative else 1
    net_attn_maps = []

    for name, attn_map in attn_maps.items():
        attn_map = attn_map.cpu() if detach else attn_map
        attn_map = torch.chunk(attn_map, batch_size)[idx] # (20, 32*32, 77) -> (10, 32*32, 77) # negative & positive CFG
        if len(attn_map.shape) == 4:
            attn_map = attn_map.squeeze()

        attn_map = upscale(attn_map, target_size) # (10,32*32,77) -> (77,64*64)
        net_attn_maps.append(attn_map) # (10,32*32,77) -> (77,64*64)

    net_attn_maps = torch.mean(torch.stack(net_attn_maps,dim=0),dim=0)
    net_attn_maps = net_attn_maps.reshape(net_attn_maps.shape[0], 64,64) # (77,64*64) -> (77,64,64)

    return net_attn_maps


def save_net_attn_map(net_attn_maps, dir_name, tokenizer, prompt):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    tokens = prompt2tokens(tokenizer, prompt)
    total_attn_scores = 0
    for i, (token, attn_map) in enumerate(zip(tokens, net_attn_maps)):
        attn_map_score = torch.sum(attn_map)
        attn_map = attn_map.cpu().numpy()
        h,w = attn_map.shape
        attn_map_total = h*w
        attn_map_score = attn_map_score / attn_map_total
        total_attn_scores += attn_map_score
        token = token.replace('</w>','')
        save_attn_map(
            attn_map,
            f'{token}:{attn_map_score:.2f}',
            f"{dir_name}/{i}_<{token}>:{int(attn_map_score*100)}.png"
        )
    print(f'total_attn_scores: {total_attn_scores}')


def generate_attn_masks_for_each_block(masks_origin, pure_cross=False, device="cuda"):
    """
    masks: list of binary masks, each with foreground from 0 to 255.
    """
    num_hard_masks = len(masks_origin)
    
    mask_for_each_block = {}
    latent_shapes = [4096, 1024, 256, 64]
    masks_origin = torch.tensor(np.array(masks_origin), dtype=torch.float32) / 255
    
    # ori_masks = masks.clone()
    n_token = 1024
    side_len = int(math.sqrt(1024))
        
    masks = F.interpolate(masks_origin.clone().unsqueeze(0), (side_len, side_len), mode="nearest").squeeze().reshape(num_hard_masks, -1, 1)
    
    self_attn_mat = torch.ones(1, n_token, n_token)
    resized_masks = masks.repeat(1, 1, n_token)
    # for regions outside any the mask, it shoud not attend to or attended by any patches.
    for i in range(num_hard_masks):
        self_attn_mat *= resized_masks[i].clone()
    # self_attn_mat = torch.tensor(self_attn_mat, dtype=torch.int16)
    
    resized_masks = torch.cat([resized_masks[i] for i in range(resized_masks.shape[0])], dim=1).unsqueeze(0)
    if not pure_cross:
        resized_masks = torch.cat((self_attn_mat * 10, resized_masks), dim=-1)
    
    resized_masks[resized_masks == 0] = -10000
    resized_masks[resized_masks == 10] = 0
    resized_masks[resized_masks > 0] = resized_masks[resized_masks > 0] - 1
    ratio = int(resized_masks.shape[-1] / resized_masks.shape[-2])
    
    mask_1024 = resized_masks.clone()
    mask_4096 = F.interpolate(mask_1024.clone().unsqueeze(0), (4096, 4096 * ratio), mode="bilinear")[0]
    mask_256 = F.interpolate(mask_1024.clone().unsqueeze(0), (256, 256 * ratio), mode="bilinear")[0]
    mask_64 = F.interpolate(mask_1024.clone().unsqueeze(0), (64, 64 * ratio), mode="bilinear")[0]
    
    mask_for_each_block[4096] = mask_4096.to(device)
    mask_for_each_block[1024] = mask_1024.to(device)
    mask_for_each_block[256] = mask_256.to(device)
    mask_for_each_block[64] = mask_64.to(device)
    
    
    return mask_for_each_block

def add_noise_to_image(noise_step, args, img: PIL.Image, vae, train_transforms, noise=None, noise_scheduler=None):
    """
    takes image path and noise step as input.
    """

    if noise_scheduler is None:
        noise_scheduler = DDPMScheduler.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="scheduler", local_files_only=True)
        
    timestep = torch.tensor(noise_step).to(vae.device)
    
    # img = Image.open("/home/a1901664/Projects/diffusion_base/man_soccer_grassland.png").convert("RGB")
    # img = Image.open("/home/a1901664/Projects/diffusion_base/cups_can.png").convert("RGB")
    img = train_transforms(img).unsqueeze(0).to(vae.device)
    
    # breakpoint()
    # generator.manual_seed(args.seed)
    latents = vae.encode(img.to(torch.float32)).latent_dist.sample()
    latents = latents * vae.config.scaling_factor
                    
    # Get the text embedding for conditioning
                    
    # Sample noise that we'll add to the latents
    if noise is None:
        # generator.manual_seed(args.seed)
        noise = torch.randn_like(latents)
    # Sample a random timestep for each image
    
    timesteps = timestep
    timesteps = timesteps.long() 
    # generator.manual_seed(args.seed)
    noisy_latents = noise_scheduler.add_noise(latents, noise, torch.tensor([timesteps]))
    
    del latents, img, noise
    torch.cuda.empty_cache()
    gc.collect()
    
    return noisy_latents



def resize_net_attn_map(net_attn_maps, target_size):
    net_attn_maps = F.interpolate(
        net_attn_maps.to(dtype=torch.float32).unsqueeze(0),
        size=target_size,
        mode='bilinear',
        align_corners=False
    ).squeeze() # (77,64,64)
    return net_attn_maps


def save_attn_map(attn_map, title, save_path):
    normalized_attn_map = (attn_map - np.min(attn_map)) / (np.max(attn_map) - np.min(attn_map)) * 255
    normalized_attn_map = normalized_attn_map.astype(np.uint8)
    image = Image.fromarray(np.uint8(normalized_attn_map))
    image.save(save_path, format='PNG', compression=0)

def feat_svd(in_feat, top_k=[0, 5]):
    """
    Do svd to image feature. Preseving the top_k sincular values.
    """
    # assert abs(top_k[-1]) > abs(top_k[0]) and top_k[0] * top_k[-1] >= 0, "invalid zero-out range!"
    C, W, H = in_feat.shape
    in_feat_flat = in_feat.reshape(C, W * H)
    U, s, V = torch.linalg.svd(in_feat_flat, full_matrices=False)
    
    # S = torch.zeros_like(in_feat_flat[1, :, :])
    # S.diagonal().copy_(s)
    if top_k[-1] == -1:
        # zero out to last
        s[min(C, top_k[0]):] = 0
    else:
        # zero out any range
        s[min(C, top_k[0]): min(C, top_k[-1])] = 0
        
    S = torch.diag(s)
    # S[top_k: min(W, H), top_k:min(W, H)] = 0
    
    feat_ = U @ S @ V
    
    feat_ = feat_.reshape(in_feat.shape)
    
    return feat_

def rleToMask(rleList,height,width):
    rows,cols = height,width
    rleNumbers = rleList
    rlePairs = np.array(rleNumbers).reshape(-1,2)
    img = np.zeros(rows*cols,dtype=np.uint8)
    for index,length in rlePairs:
        index -= 1
        img[index:index+length] = 255
    img = img.reshape(cols,rows)
    img = img.T
    return img


from PIL import Image
import numpy as np
import torch

def save_image(image, filename):
    """
    Saves an image (NumPy array, Tensor, or PIL Image) to a specified file.
    
    Parameters:
        image (numpy.ndarray, torch.Tensor, PIL.Image): The image to save.
        filename (str): The name of the file where the image will be saved.
    """
    # Check if the input image is a NumPy array
    if isinstance(image, np.ndarray):
        # Convert from NumPy array to PIL Image
        image = Image.fromarray(image.astype('uint8'), 'RGB')
    
    # Check if the input image is a PyTorch Tensor
    elif torch.is_tensor(image):
        # Convert from Tensor to PIL Image
        image = image.permute(1, 2, 0).cpu().numpy().astype('uint8')
        image = Image.fromarray(image, 'RGB')
    
    # Save the image
    image.save(filename)


import numpy as np
import random

def extract_random_blocks(image):
    """
    Extracts 3 to 6 non-overlapping square blocks of varying sizes from an image.
    
    Parameters:
        image (numpy.ndarray): The input image with shape (h, w, 3).
    
    Returns:
        numpy.ndarray: The original image.
        list: A list named 'references' containing the extracted blocks as numpy arrays.
    """
    h, w, _ = image.shape
    min_side = min(h, w)
    min_block_size = int(min_side * 0.2)
    
    # Determine the number of blocks to extract
    num_blocks = random.randint(1, 5)
    
    references = []
    used_areas = []
    
    for i in range(num_blocks):
        # Generate random block size and position
        block_size = random.randint(min_block_size, min_side // 2)
        max_x, max_y = w - block_size, h - block_size
        x, y = random.randint(0, max_x), random.randint(0, max_y)
        
        block = copy.deepcopy(image[y:y+block_size, x:x+block_size, :])
        
        if (block).sum() != 0:
            references.append(block)
            
        used_areas.append((x, y, block_size))
        image[y:y+block_size, x:x+block_size, :] -= block
        
    return image, references


from PIL import Image
import numpy as np

def crop_to_square_and_save(image, bbox):
    """
    Crops an area defined by a bounding box from an image and adjusts it to a square.
    The square is expanded or shrunk to fit within the image boundaries if necessary.
    
    Parameters:
        image (numpy.ndarray): The input image with shape (h, w, 3).
        bbox (tuple): A bounding box in the format (x_min, y_min, w, h).
    
    Returns:
        numpy.ndarray: The original image.
        list: A list named 'references' containing the cropped square as a numpy array.
    """
    x_min, y_min, w, h = bbox
    x_min, y_min, w, h = int(x_min), int(y_min), int(w), int(h)
    
    h_img, w_img, _ = image.shape
    
    # Determine the size of the square
    size = max(w, h)
    
    # Adjust to keep within image boundaries
    if x_min + size > w_img or y_min + size > h_img:
        # If the square goes out of bounds, shrink it
        size = min(w, h)
        if x_min + size > w_img:
            x_min = w_img - size
        if y_min + size > h_img:
            y_min = h_img - size
    else:
        # If possible, expand to the larger dimension
        if w > h:
            if y_min + size > h_img:
                y_min = h_img - size
        else:
            if x_min + size > w_img:
                x_min = w_img - size

    # Ensure the coordinates and size don't go out of bounds
    x_min, y_min = max(0, x_min), max(0, y_min)
    size = min(size, h_img - y_min, w_img - x_min)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Crop the image
    cropped_image = copy.deepcopy(image[y_min:y_min+size, x_min:x_min+size, :])
    cv2.imwrite("cropped_image.png", cropped_image)
    image_bbox = cv2.rectangle(image, (x_min, y_min), (x_min + w, y_min + h), thickness=2, color=(0, 0, 255))
    cv2.imwrite("image_bbox.png", image_bbox)
    image[y_min:y_min+size, x_min:x_min+size, :] -= cropped_image
    
    
    return image, cropped_image



def center_square_crop(img, img_h, img_w):
    crop_h = min(img_h, img_w)
    crop_w = crop_h
        
    x = img_w/2 - crop_h/2
    y = img_h/2 - crop_w/2

    crop_img = copy.deepcopy(img[int(y):int(y+crop_h), int(x):int(x+crop_w), :])
        
    return crop_img


def interpolate_tensor(in_tensor, tar_size, mode="bilinear"):
    if not isinstance(tar_size, tuple):
        raise TypeError("tar_size must be a tuple.")

    resized_tensor = F.interpolate(in_tensor, size=tar_size, mode=mode, align_corners=False)
    
    return resized_tensor


def interpolate_numpy(numpy_mat, tar_size, mode="nearest"):
    """
    Resizes an array to the target size using specified interpolation mode.

    Parameters:
    - numpy_mat (numpy.ndarray): The input array with shape (W, H, C).
    - tar_size (tuple): The target size as a tuple (W', H').
    - mode (str): The interpolation mode, "nearest" for nearest neighbor or 
                  "linear" for linear interpolation.

    Returns:
    - numpy.ndarray: The resized array with shape (W', H', C).
    """
    # Calculate zoom factors for each dimension
    if len(numpy_mat.shape) == 3:
        W, H, C = numpy_mat.shape
    elif len(numpy_mat.shape) == 2:
        W, H = numpy_mat.shape
        
    W_prime, H_prime = tar_size
    zoom_factor_W = W_prime / W
    zoom_factor_H = H_prime / H

    # Choose order based on mode
    if mode == "nearest":
        order = 0
    elif mode == "linear":
        order = 1
    else:
        raise ValueError("Unsupported interpolation mode. Use 'nearest' or 'linear'.")

    # Apply zoom
    if len(numpy_mat.shape) == 3:
        resized_mat = zoom(numpy_mat, (zoom_factor_W, zoom_factor_H, 1), order=order)
    else:
        resized_mat = zoom(numpy_mat, (zoom_factor_W, zoom_factor_H), order=order)

    return resized_mat

def images_to_references_dino(image_paths, max_len=10, device="cuda:0"):
    dinov2_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    dinov2 = AutoModel.from_pretrained('facebook/dinov2-base')
    image_tensors = []
    for i, image_path in enumerate(image_paths):
        img = Image.open(image_path).convert('RGB')
        inputs = dinov2_processor(images=img, return_tensors="pt")
        outputs = dinov2(**inputs)
        last_hidden_states = outputs.last_hidden_state
        cls_token, patch_tokens = last_hidden_states[:, :1, :], last_hidden_states[:, 1:, :]
        # img_feat = torch.concat((cls_token * patch_tokens, cls_token.repeat((1, 256, 1))), dim=-1) # (1, 256, 1536), used as image for cropping
        img_feat = patch_tokens # (1, 256, 1536), used as image for cropping
        feat_hw = (img_feat[0].shape)[0]
        img_feat = img_feat[0].reshape(int(feat_hw ** 0.5), int(feat_hw ** 0.5), -1)
        image_tensors.append(img_feat.to(device))
    
    references = [interpolate_tensor(input_img.reshape(1, -1, input_img.shape[0], input_img.shape[1]), tar_size=(8, 8), mode="bilinear").reshape(64, -1) for input_img in image_tensors]
    if len(references) < max_len:
        references += [torch.zeros_like(references[0]) for i in range(max_len - len(references))]
    else:
        references = references[:max_len]
    
    references = torch.stack(references)
    references = references.reshape(-1, references.shape[-1])
    
    return references


def augment_mask(mask, foreground_pix = 255):
    """
    对单通道 mask 进行数据增广
    """
    
    def random_dilate(mask, min_percentage=10, max_percentage=20):
        """
        随机膨胀 mask 范围 10~20%
        """
        # 计算膨胀的大小
        percentage = random.uniform(min_percentage, max_percentage) / 100
        size = int(percentage * min(mask.shape))

        # 使用一个结构元素进行膨胀
        structure = np.ones((size, size), dtype=np.int8)
        dilated_mask = ndimage.binary_dilation(mask, structure=structure).astype(mask.dtype) * 255

        return dilated_mask

    def random_erasing(mask, max_percentage=40, foreground_pix=255):
        """
        随机抹掉部分前景区域
        """
        # 计算抹掉的大小
        percentage = random.uniform(15, max_percentage) / 100
        size = int(percentage * min(mask.shape))
        
        mask_copy = mask.copy()
        
        # 确保抹掉的区域在mask区域内
        foreground_indices = np.argwhere(mask == foreground_pix)
        if len(foreground_indices) == 0:
            return mask_copy  # 没有前景区域，返回原始mask
        
        # 随机选择抹掉区域的中心
        center_idx = random.choice(foreground_indices)
        y_start = max(0, center_idx[0] - size // 2)
        x_start = max(0, center_idx[1] - size // 2)
        
        y_end = min(mask.shape[0], y_start + size)
        x_end = min(mask.shape[1], x_start + size)
        
        mask_copy[y_start:y_end, x_start:x_end] = 0
        
        return mask_copy

    
    # 随机选择是否使用膨胀和抹掉操作
    apply_dilate = random.choice([True, False])
    apply_erasing = random.choice([True, False])

    augmented_mask = mask
    if apply_dilate:
        augmented_mask = random_dilate(augmented_mask)
    if apply_erasing:
        augmented_mask = random_erasing(augmented_mask, foreground_pix=foreground_pix)
    
    return augmented_mask

def resize_image_to_fit_short(image, short_size=512):
    # 获取原始图片的宽度和高度
    try:
        original_width, original_height = image.size
    except:
        breakpoint()
    
    # 判断最短边
    if original_width < original_height:
        new_width = short_size
        new_height = int((original_height / original_width) * short_size)
    else:
        new_height = short_size
        new_width = int((original_width / original_height) * short_size)
    
    # 调整图片大小
    resized_image = image.resize((new_width, new_height))
    
    return resized_image


def extract_subject_features(args, image_paths, reference_unet, text_encoder, tokenizer, vae, noise_scheduler, subject_noise, weight_dtype, transforms, text="", device="cuda:0", subject_denoise_timestep = None, generator=None):

    references = []
    # image_paths to references
    if type(image_paths) == str:
        image_paths = [image_paths]
    
    for i, img_path in enumerate(image_paths):
        img = Image.open(img_path).convert('RGB')
        img = transforms(img)
        references.append(img)
    
    references = torch.stack(references)
    try:
        inputs = tokenizer(text, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")['input_ids']
    except:
        breakpoint()
        
    inputs = inputs[:, None, :]
    subject_encoder_hidden_states = text_encoder(inputs.to(reference_unet.device), return_dict=False)[0]
    
    subject_denoise_timestep = torch.tensor(subject_denoise_timestep, device=reference_unet.device).repeat(references.shape[0])
    subject_denoise_timestep = subject_denoise_timestep.long()
    # prepare references from unet, convert images to latent space

    comp_latents = vae.encode(references.to(weight_dtype).to(reference_unet.device)).latent_dist.sample()
    comp_latents = comp_latents * vae.config.scaling_factor
    
    subject_noise = torch.randn_like(comp_latents[:1, :, :, :]) # tensor(115.0338, device='cuda:0')

    noisy_comp_latents = noise_scheduler.add_noise(comp_latents, subject_noise, subject_denoise_timestep) # tensor(868.9863, device='cuda:0')
    # subject_features: [Block1 features for 10 ref img, Block2, ...,Block16]
    _, subject_features = reference_unet(noisy_comp_latents, subject_denoise_timestep, subject_encoder_hidden_states, return_dict=False, args=args)
    subject_features = [block_feat.reshape(1, -1, block_feat.shape[-1]) if block_feat is not None else None for block_feat in subject_features] # B, sub_image_patches, dim 
    
    
    return subject_features


def visualize_latent(latent: torch.Tensor, vae, generator=None):
    if generator is None:
        generator = torch.Generator(device="cuda").manual_seed(42)
        
    image = vae.decode(latent / vae.config.scaling_factor, return_dict=False, generator=generator)[0]
    image = (image + 1)*127
    img = image[0].cpu().numpy().astype(np.uint8).transpose(1,2,0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite("latent.png", img)
    return img

def save_histogram(array_1d, path, range_min, range_max):
    plt.figure(figsize=(10, 5))
    plt.hist(array_1d, bins=30, color='blue', edgecolor='black', range=(range_min, range_max))  
    plt.title('Histogram of Data Points')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    plt.tight_layout()

    plt.savefig(path)

    plt.close()



def plot_pixel_in_order(array_1d, path):
    # 设置图形大小和边框紧凑度
    plt.figure(figsize=(10, 5))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    # 绘制数据
    plt.plot(array_1d, linewidth=2)  # 可以通过调整linewidth来控制线条粗细

    # 设置坐标轴的范围
    plt.ylim(0, 30)
    plt.xlim(0, 4095)  # 索引从0到4095

    # 设置坐标轴标签
    plt.xlabel('Index')
    plt.ylabel('Value')

    # 显示图表
    plt.savefig(path)

from scipy.ndimage import zoom

def resize_single_channel_image(image, new_size):
    # 计算缩放因子
    zoom_factor = (new_size[0] / image.shape[0], new_size[1] / image.shape[1])
    # 使用双三次插值进行缩放
    return zoom(image, zoom_factor, order=3)


def top_m_indices(matrix, M):
    """
    找出矩阵中前M个最大值的位置.

    参数:
    matrix (numpy.ndarray): 输入的NxN矩阵.
    M (int): 需要找出的最大值的数量.

    返回:
    list: 前M个最大值的矩阵索引，以 (row, col) 的形式返回.
    """
    # 将矩阵展平为一维数组
    flat_array = matrix.flatten()
    
    # 找到前M个最大值的索引
    top_m_indices = np.argpartition(flat_array, -M)[-M:]
    
    # 将一维索引转换回二维索引
    top_m_indices_2d = np.unravel_index(top_m_indices, matrix.shape)
    
    # 获取前M个最大值的位置
    top_m_positions = list(zip(top_m_indices_2d[0], top_m_indices_2d[1]))
    
    return top_m_positions

def prepare_mean_masks_each_word(auto_masks, num_infer_steps=25, step_thresh=(5,25), expand_dist=None):
    
    def prepare_summed_masks(auto_masks, num_infer_steps, step_thresh):
        """
        auto_masks: [
            [
                [inf_step, var, ori_mask], [inf_step, var, ori_mask] # for each word
            ] 
        ]
        """
        num_words = len(auto_masks[0])
        # only keep the blocks inside the step_thresh
        auto_masks = auto_masks[step_thresh[0] * 16: (step_thresh[1]) * 16]
        
        steps_remain = step_thresh[1] - step_thresh[0]
        
        # TODO: bug for DDPM scheduler
        mask_vars_all_words = np.zeros(shape=(num_words, steps_remain, 16), dtype=np.float32)
        for i, per_block_masks in enumerate(auto_masks):
            for word_idx, per_word_mask in enumerate(per_block_masks):
                inf_step, block_idx, mask, variance = per_word_mask
                try:
                    mask_vars_all_words[word_idx][inf_step - step_thresh[0]][block_idx] = variance
                except:
                    breakpoint()
        

        block_step_pairs_all_words = []
        for word_id in range(num_words):
            mask_vars_each_word = mask_vars_all_words[word_id]
            # mask_vars_each_word = mask_vars_each_word[5:, :]
            # find the masks with highest var values.
            
            
            
            # find the first N block idxs with the maximum avg var
            avg_vars_per_block = np.mean(mask_vars_each_word, axis=0)
            top_k_var_blocks = np.argsort(avg_vars_per_block)[::-1]
            top_k_var_blocks = top_k_var_blocks[:1] if avg_vars_per_block[top_k_var_blocks][0] < 10 else top_k_var_blocks[:3]
            
            # then, find the steps that has the largest vars in this block
            avg_vars_per_block = mask_vars_each_word[:, top_k_var_blocks]
            top_k_var_steps = np.argsort(avg_vars_per_block, axis=0)[::-1]
            top_k_var_steps = top_k_var_steps[:3]
            
            block_step_pairs_per_word = []
            
            for row in top_k_var_steps:
                for (a, b) in zip(top_k_var_blocks, row):
                    block_step_pairs_per_word.append((a, b + step_thresh[0]))
            
            # block_step_pairs_per_word = [(a, b) for a, b in zip(top_k_var_blocks, row) for row in top_k_var_steps] # block, step
            block_step_pairs_all_words.append(block_step_pairs_per_word)
        
        # breakpoint()
        
        summed_masks_all_words = {i: [] for i in range(num_words)}
        for i, per_block_masks in enumerate(auto_masks):
            for word_idx, per_word_mask in enumerate(per_block_masks):
                inf_step, block_idx, mask_ori, variance = per_word_mask
                
                if not (block_idx, inf_step) in block_step_pairs_all_words[word_idx]: # since we ignored first 5 steps
                    continue
                
                summed_masks_all_words[word_idx].append(mask_ori)
        # breakpoint()
        if summed_masks_all_words[0] == []:
            breakpoint()
        return summed_masks_all_words, num_words

    def average_masks(summed_masks_all_words, num_words):
        mask_to_be_displayed = []
        for j in range(num_words):
            masks_per_word = summed_masks_all_words[j]
            avg_mask = sum(masks_per_word) / len(masks_per_word)
            
            avg_mean = avg_mask.mean()
            
            # ================================================================
            # Second Suppression
            # if avg_mean > 0:
            #     avg_mask[avg_mask < avg_mean] = 0
            # else:
            #     avg_mask[avg_mask < avg_mean] = -10000
                
            avg_mask, foreground_mask = foreground_minmax_norm(avg_mask, thresh=avg_mean)
            foreground_pixels = avg_mask[foreground_mask]
            foreground_seceond_thresh = foreground_pixels.mean()
            
            # if foreground_seceond_thresh > 0:
            #     foreground_seceond_thresh /= 2
            # else:
            #     foreground_seceond_thresh *= 2
            
            foreground_pixels[foreground_pixels >= foreground_seceond_thresh] = 255
            foreground_pixels[foreground_pixels < foreground_seceond_thresh] = 0
            avg_mask[foreground_mask] = foreground_pixels
            
            if avg_mean < 0:
                avg_mask[avg_mask == -10000] = 0
            # ================================================================
            
            
            # # ================================================================
            # # Mean Suppression
            # if avg_mean > 0:
            #     avg_mask[avg_mask < avg_mean] = 0
            #     avg_mask[avg_mask >= avg_mean] = 255
            # else:
            #     avg_mask[avg_mask >= avg_mean] = 255
            #     avg_mask[avg_mask < avg_mean] = 0
            # # ================================================================
            
            mask_to_be_displayed.append(avg_mask)
        return mask_to_be_displayed
    
    def foreground_minmax_norm(array, thresh):
        # 创建一个掩码，用于标记大于0的部分
        mask = array > thresh

        # 提取大于0的部分
        non_zero_values = array[mask]

        # 对大于0的部分进行Min-Max归一化
        min_val = np.min(non_zero_values)
        max_val = np.max(non_zero_values)
        normalized_values = (non_zero_values - min_val) / (max_val - min_val)

        # 创建一个新的数组来存储归一化的结果
        normalized_array = np.zeros_like(array, dtype=float)
        normalized_array[mask] = normalized_values
        
        return normalized_array, mask
    
    summed_masks_all_words, num_words = prepare_summed_masks(auto_masks, num_infer_steps=25, step_thresh=step_thresh)
    mask_to_be_displayed = average_masks(summed_masks_all_words, num_words)
    
    expanded_masks = []
    for i in range(num_words):
        mask = mask_to_be_displayed[i]
        if expand_dist is not None:
            mask = expand_mask(mask, max_distance=expand_dist, foreground_value=255)
            # mask = gaussian_filter(mask, sigma=args.gaussian_sigma)
        expanded_masks.append(mask)
    
    return expanded_masks
    

def compute_clip_text_image_similarity(image_input, text, clip_model, preprocess, device):
    # Load and preprocess the image
    image_input = preprocess(image_input).unsqueeze(0).to(device)
    
    # Tokenize and preprocess the text
    text_tokens = clip.tokenize([text]).to(device)
    
    with torch.no_grad():
        # Get image features
        image_features = clip_model.encode_image(image_input).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # Get text features
        text_features = clip_model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Compute similarity
        similarity = torch.matmul(image_features, text_features.T).squeeze().item()
    
    return similarity


def calculate_dino_similarity(model, processor, image1: PIL.Image, image2: PIL.Image, hard_mask=None, device='cuda'):
    # processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base', local_files_only=True)
    # model = AutoModel.from_pretrained('facebook/dinov2-base', local_files_only=True).to(device)
    with torch.no_grad():
        inputs1 = processor(images=image1, return_tensors="pt")['pixel_values'].to(device)
        if hard_mask is not None:
            image_size = inputs1.shape[-2:]
            hard_mask = torch.tensor(hard_mask, dtype = inputs1.dtype).to(device)
            hard_mask = F.interpolate(hard_mask.unsqueeze(0).unsqueeze(0), image_size, mode="nearest")
            inputs1 = inputs1 * hard_mask
        
        outputs1 = model(inputs1)
        
        image_features1 = outputs1.last_hidden_state
        
        # image_features1 = image_features1.mean(dim=1)
        
    with torch.no_grad():
        inputs2 = processor(images=image2, return_tensors="pt")['pixel_values'].to(device)
        
        if hard_mask is not None:
            inputs2 = inputs2 * hard_mask
        outputs2 = model(inputs2)
        
        image_features2 = outputs2.last_hidden_state

        # image_features2 = image_features2.mean(dim=1)

    cos = nn.CosineSimilarity(dim=0)
    sim = cos(image_features1[0],image_features2[0]).item()
    sim = (sim+1)/2
    
    del image_features2, image_features1, inputs1, inputs2, outputs1, outputs2, cos
    torch.cuda.empty_cache()
    gc.collect()
    
    return sim

def compute_clip_similarity(model, preprocess, image1, image2, device):
    image1_input = preprocess(image1).unsqueeze(0).to(device)
    image2_input = preprocess(image2).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image1_features = model.encode_image(image1_input).float()
        image2_features = model.encode_image(image2_input).float()
        
        image1_features /= image1_features.norm(dim=-1, keepdim=True)
        image2_features /= image2_features.norm(dim=-1, keepdim=True)
        
        similarity = torch.matmul(image1_features, image2_features.T).item()
        
    return similarity


def find_foreground_indices(image_path, shape: int):
    """
    Reads an image from a given path, resizes it to the specified shape, takes the first channel,
    and returns the indices of foreground pixels (non-255 values in the first channel).
    
    :param image_path: Path to the image file.
    :param shape: Desired size to which the image will be resized, must be a perfect square.
    :return: A numpy array of indices where foreground pixels in the first channel are located.
    """
    # 读取图像
    image = Image.open(image_path)
    
    # 计算新的尺寸（假设shape是完美的平方）
    new_size = int(np.sqrt(shape))
    if new_size * new_size != shape:
        raise ValueError("Shape must be a perfect square")
    
    # 调整图像大小并转换为灰度（只取第一个通道）
    resized_image = image.resize((new_size, new_size), resample=0)
    resized_image = np.array(resized_image)[:, :, 0]  # 取第一个通道
    
    # 将图像转换为一维数组
    flat_image = resized_image.flatten()
    
    # 查找所有非255值的像素的索引
    foreground_indices = np.where(flat_image != 255)[0]

    return foreground_indices, flat_image

def resize_and_crop_or_pad(img_array):
    # 随机生成一个缩放比例在0.5到1.2之间
    scale_factor = random.uniform(0.5, 1.2)
    
    # 创建PIL图像
    img = Image.fromarray(img_array)
    original_width, original_height = img.width, img.height
    
    # 计算新的尺寸
    new_width = int(img.width * scale_factor)
    new_height = int(img.height * scale_factor)
    
    if scale_factor > 1:
        # 缩放图像
        resized_img = img.resize((new_width, new_height), Image.ANTIALIAS)
        # 裁剪图像
        left = (new_width - original_width) // 2
        top = (new_height - original_height) // 2
        right = left + original_width
        bottom = top + original_height
        cropped_img = resized_img.crop((left, top, right, bottom))
        result_array = np.array(cropped_img)
    else:
        # 缩放图像
        resized_img = img.resize((new_width, new_height), Image.ANTIALIAS)
        # 创建一个白色背景的图像
        new_img = Image.new('RGB', (original_width, original_height), (255, 255, 255))
        # 计算粘贴位置
        upper_x = (original_width - new_width) // 2
        upper_y = (original_height - new_height) // 2
        # 将缩放后的图像粘贴到白色背景图像中
        new_img.paste(resized_img, (upper_x, upper_y))
        result_array = np.array(new_img)
    
    return result_array

def random_rotate(image_array):
    rows, cols = image_array.shape[:2]

    # 生成随机旋转角度
    angle = np.random.uniform(-360, 360)

    # 计算旋转矩阵的中心点
    center = (cols / 2, rows / 2)

    # 计算旋转矩阵
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 计算新图像的尺寸以确保整个图像都可以显示
    cos_val = np.abs(matrix[0, 0])
    sin_val = np.abs(matrix[0, 1])

    new_cols = int((rows * sin_val) + (cols * cos_val))
    new_rows = int((rows * cos_val) + (cols * sin_val))

    # 调整旋转矩阵以考虑平移
    matrix[0, 2] += (new_cols / 2) - center[0]
    matrix[1, 2] += (new_rows / 2) - center[1]

    # 应用仿射变换
    rotated_image = cv2.warpAffine(image_array, matrix, (new_cols, new_rows), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    return rotated_image

def expand_foreground_hard(mask, expansion_pixels=10):
    """
    扩展单通道二值掩码中的前景区域。
    
    参数:
    mask (numpy.ndarray): 输入的二值掩码，前景为1，背景为0。
    expansion_pixels (int): 扩展前景的像素数量。

    返回:
    numpy.ndarray: 扩展后的掩码。
    """
    # 生成用于扩展的结构元素
    structuring_element = np.ones((2*expansion_pixels+1, 2*expansion_pixels+1), dtype=bool)
    
    # 使用binary_dilation扩展前景
    expanded_mask = binary_dilation(mask, structure=structuring_element)
    
    return expanded_mask.astype(np.uint8)  # 转换为uint8以保持与输入一致的格式

def expand_foreground_soft(mask, expansion_pixels=10):
    """
    扩展单通道二值掩码中的前景区域，使其能够在expansion_pixels个像素内从1渐变到0。
    
    参数:
    mask (numpy.ndarray): 输入的二值掩码，前景为1，背景为0，渐变区域介于0到1之间。
    expansion_pixels (int): 扩展前景的像素数量。

    返回:
    numpy.ndarray: 扩展后的掩码。
    """
    
    # 计算前景到背景的距离
    distance = distance_transform_edt(mask == 0)

    # 将距离映射到0到1之间，1为前景区域，0为背景区域
    expanded_mask = np.clip(1 - distance / expansion_pixels, 0, 1)

    return expanded_mask


def fill_bounding_rect(mask):
    # Ensure mask is in the correct binary format
    mask = mask.astype(np.uint8)
    
    # Find all contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Combine all contours into one
    combined_contour = np.vstack(contours)
    
    # Get the bounding rectangle that contains all contours
    x, y, w, h = cv2.boundingRect(combined_contour)
    
    # Create a copy of the original mask to draw the rectangle on
    filled_mask = mask.copy()
    
    # Fill the bounding rectangle area with 1
    filled_mask[y:y+h, x:x+w] = 1
    
    # filled_mask = feather_mask(filled_mask)
    
    return filled_mask



def feather_mask(mask, sigma=5):
    """
    Feather the foreground of a binary mask using a Gaussian blur.

    Parameters:
    - mask: numpy array of shape (W, H) with foreground as 1 and background as 0.
    - sigma: Standard deviation for Gaussian kernel, controlling the amount of feathering.

    Returns:
    - feathered_mask: numpy array of shape (W, H) with feathered foreground.
    """
    # Ensure mask is in the correct binary format
    mask = mask.astype(np.float32)

    # Apply Gaussian blur to the mask
    blurred_mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)

    # Normalize the blurred mask to be within the range [0, 1]
    feathered_mask = blurred_mask / np.max(blurred_mask)

    return feathered_mask


def expand_mask(mask, max_distance, foreground_value=1):
    """
    对掩码的边缘进行拓展，拓展部分从前景值逐渐减少到0，原本是前景值的部分不变。
    
    参数:
    mask (np.ndarray): 大小为 (W, H) 的单通道 NumPy 掩码。
    max_distance (float): 拓展范围。
    foreground_value (int or float): 前景像素值。
    
    返回:
    np.ndarray: 拓展后的掩码，大小为 (W, H)。
    """
    if not max_distance:
        return mask
    # 计算到前景的距离
    distance = distance_transform_edt(mask != foreground_value)

    # 计算边缘扩展效果
    expanded_mask = foreground_value * np.exp(-distance / max_distance)

    # 保持原始前景不变
    expanded_mask[mask == foreground_value] = foreground_value

    expanded_mask[expanded_mask < 100] = 0
    return expanded_mask


def create_centered_image(img, in_size=1024):
    """
    Function to place a 3-channel image in the center of a black square canvas
    and resize it to 1024x1024.

    Args:
    img (PIL.Image): Input image.

    Returns:
    PIL.Image: Resized image on a black square canvas.
    tuple: Original size of the image.
    """
    # Get the dimensions of the input image
    original_size = img.size

    # Determine the size of the square canvas
    max_dim = max(original_size)
    square_canvas_size = (max_dim, max_dim)

    # Create a black square canvas
    img_square = Image.new("RGB", square_canvas_size, (0, 0, 0))

    # Calculate the position to paste the original image onto the square canvas
    paste_position = ((max_dim - original_size[0]) // 2, 
                      (max_dim - original_size[1]) // 2)

    # Paste the original image onto the square canvas
    img_square.paste(img, paste_position)

    # Resize the square canvas to 1024x1024
    img_1024 = img_square.resize((in_size, in_size), Image.ANTIALIAS)

    return img_1024, original_size


def extract_original_image(original_size, img_1024):
    """
    Function to extract the original image from a 1024x1024 black square canvas.

    Args:
    original_size (tuple): Original size of the image.
    img_1024 (PIL.Image): Image on a 1024x1024 black square canvas.

    Returns:
    PIL.Image: Extracted original image.
    """
    # Resize the 1024x1024 image back to the square canvas size
    max_dim = max(original_size)
    img_square = img_1024.resize((max_dim, max_dim), Image.ANTIALIAS)

    # Calculate the position where the original image was pasted
    paste_position = ((max_dim - original_size[0]) // 2, 
                      (max_dim - original_size[1]) // 2)

    # Extract the original image from the square canvas
    img_extracted = img_square.crop((paste_position[0], paste_position[1], 
                                     paste_position[0] + original_size[0], 
                                     paste_position[1] + original_size[1]))

    return img_extracted



def random_based_on_time():
    # 获取当前时间
    current_time = datetime.datetime.now()
    local_random = random.Random()
    local_random.seed(current_time.microsecond)
    # 生成并返回0到10之间的随机数
    return local_random.randint(1, 11)

def warp_image_affine(image_array):
    rows, cols = image_array.shape[:2]

    # 设置变形前后对应的三个点
    src_points = np.float32([[0, 0], [cols-1, 0], [0, rows-1]])
    dst_points = np.float32([[0, 0], [int(0.9 * (cols-1)), int(0.1 * (rows-1))], [int(0.1 * (cols-1)), int(0.9 * (rows-1))]])

    # 生成变形矩阵
    matrix = cv2.getAffineTransform(src_points, dst_points)

    # 应用仿射变换
    warped_image = cv2.warpAffine(image_array, matrix, (cols, rows))

    return warped_image

def warp_image_twist(image_array):
    rows, cols = image_array.shape[:2]

    # 生成网格坐标
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))

    # 使用矩阵运算创建扭曲映射
    map_x = x + 6.0 * (random_based_on_time() // 5) * np.sin(2 * np.pi * y / 180)
    map_y = y + 6.0 * (random_based_on_time() // 5)* np.sin(2 * np.pi * x / 180)

    # 应用映射进行图像扭曲
    warped_image = cv2.remap(image_array, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    return warped_image


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def visualize_img_cross_attn(query, key, token_ids, post_fix="", faltted_img=None, has_comp=0):
    B, head, n_token, dim = query.shape
    _, _, n_key_token, _ = key.shape
    faltted_img_ori = copy.copy(faltted_img)
    attn = query @ torch.permute(key, (0, 1, 3, 2)) # 2, head=5, n_q, n_k
    attn_mean = torch.mean(attn, 1, True) # avg among heads: 2, 1, n_q, n_k
    attn_mean = attn_mean[1][0] # get the conditional one, n_q, n_k
    for i in range(0, len(token_ids), max(1, len(token_ids) // 10)):
        # want to see which part of the main latent is the ref image tokens attending to.
        attn = attn_mean[:, token_ids[i] + n_token * has_comp]
        attn = np.array(attn.detach().cpu())
        
        attn = softmax(attn)
        # attn = (attn - np.min(attn)) / (np.max(attn) - np.min(attn)) * 255
        
        attn_max = round(attn.max(), 2)
        attn_min = round(attn.min(), 2)
        
        attn = attn.reshape(int(math.sqrt(n_token)), int(math.sqrt(n_token)))
        # attn_mean = cv2.cvtColor(attn_mean, cv2.COLOR_BGR2RGB)
        if faltted_img_ori is not None:
            faltted_img = copy.copy(faltted_img_ori)
            faltted_img[token_ids[i]] = 255
            faltted_img = faltted_img.reshape(int(math.sqrt(n_token)), int(math.sqrt(n_token)))
            
        res = np.concatenate((faltted_img, attn), axis=-1)
        plt.imsave(f"outs/attn_{str(post_fix)}_{token_ids[i]}_{attn_min}_{attn_max}.png", res.astype(np.uint8))


def visualize_img_attn_from_query(query, key, post_fix=""):
    B, head, n_token, dim = query.shape
    _, _, n_key_token, _ = key.shape
    attn = query @ torch.permute(key, (0, 1, 3, 2)) # 2, head=5, n_q, n_k
    attn_mean = torch.mean(attn, 1, True) # avg among heads: 2, 1, n_q, n_k
    attn_mean = attn_mean[1][0] # get the conditional one, n_q, n_k
    for i in range(0, attn_mean.shape[0], 5):
        # want to see which part of the main latent is the ref image tokens attending to.
        token_attn = attn_mean[i, :]
        token_attn = np.array(token_attn.detach().cpu())
        
        ratio = n_key_token // n_token
        attns = []
        for j in range(ratio):
            # breakpoint()
            # sub_attn = softmax(token_attn[j*n_token: (j+1)*n_token]) * 255
            sub_attn = token_attn[j*n_token: (j+1)*n_token]
            sub_attn = (sub_attn - np.min(sub_attn)) / (np.max(sub_attn) - np.min(sub_attn)) * 255
            
            attn_max = round(token_attn.max(), 2)
            attn_min = round(token_attn.min(), 2)
            # breakpoint()
            sub_attn = sub_attn.reshape(int(math.sqrt(n_token)), -1)
            attns.append(sub_attn)
            # breakpoint()
        if len(attns) > 1:
            if i > 500:
                breakpoint()
            res = np.hstack(attns)
        else:
            res = attns[0]
        plt.imsave(f"outs/attn_{str(post_fix)}_{i}_{attn_min}_{attn_max}.png", res.astype(np.uint8))


def visualize_attn(query, key, which_token, post_fix=""):
    B, head, n_token, dim = query.shape
    attn = query @ torch.permute(key, (0, 1, 3, 2)) # 2, head=5, n_q, n_k
    attn_mean = torch.mean(attn, 1, True) # avg among heads: 2, 1, n_q, n_k
    attn_mean = attn_mean[1][0] # get the conditional one, n_q, n_k
    if which_token == "mean":
        attn = torch.mean(attn_mean, -1) # 对所有key token取平均？
        attn = attn.reshape(int(math.sqrt(n_token)), int(math.sqrt(n_token)))
        attn = np.array(attn.detach().cpu())
        attn = (attn - np.min(attn)) / (np.max(attn) - np.min(attn)) * 255
        # attn_mean = cv2.cvtColor(attn_mean, cv2.COLOR_BGR2RGB)
        plt.imsave(f"outs/attn_mean_{str(post_fix)}.png", attn.astype(np.uint8))
    else:
        attn_mean = np.array(attn_mean.detach().cpu())
        for token in which_token:
            attn = attn_mean[:, token]
            attn = attn.reshape(int(math.sqrt(attn.shape[0])), int(math.sqrt(attn.shape[0])))
            
            attn = (attn - np.min(attn)) / (np.max(attn) - np.min(attn)) * 255
            # attn = (attn - np.min(attn))
            # attn_mean = cv2.cvtColor(attn_mean, cv2.COLOR_BGR2RGB)
            
            attn = resize_single_channel_image(np.array(attn, dtype=np.float32), (64, 64))
            
            plt.imsave(f"outs/attn_{str(post_fix)}_{str(token)}.png", attn)
            save_histogram(array_1d=np.array(attn).reshape(-1), path=f"outs/attn_{str(token)}_{str(post_fix)}_pixval.png", range_min=0, range_max=255)
            

def find_subsequence(target, source):
    for i in range(len(target) - len(source) + 1):
        if torch.all(target[i:i+len(source)] == source):
            start_index = i
            end_index = i + len(source) - 1
            return (start_index, end_index)
    return (-1, -1)  # 如果没有找到，返回-1, -1

    
def get_word_mask_according_to_text_emb(query, text_emb):
    B, head, n_token, dim = query.shape
    attn = query @ torch.permute(text_emb, (0, 1, 3, 2)) # 2, head=5, n_q, n_k
    attn = torch.mean(attn, 1, True) # avg among heads: 1, 1, n_q, 1
    
    ori_attn = attn.clone()
    ori_attn = ori_attn.flatten()
    ori_attn = ori_attn.reshape(int(ori_attn.shape[0] ** 0.5), int(ori_attn.shape[0] ** 0.5))
    
    try:
        attn = (attn - torch.min(attn)) / (torch.max(attn) - torch.min(attn)) 
    except:
        attn 
    attn = attn.flatten()
    attn = attn.reshape(int(attn.shape[0] ** 0.5), int(attn.shape[0] ** 0.5))
    
    # attn = resize_single_channel_image(np.array(attn.detach().cpu().numpy(), dtype=np.float32), (64, 64))
    # plt.imsave(f"text_attn.png", attn)
    
    return attn, ori_attn 



def extract_features(image: Image, extractor_type: str):
    if extractor_type == "dino":
        model_name = "facebook/dino-v2-small"
        processor = DINOv2Processor.from_pretrained(model_name)
        model = DINOv2Model.from_pretrained(model_name)
    elif extractor_type == "clip":
        model_name = "openai/clip-vit-base-patch32"
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)
    else:
        raise ValueError("extractor_type must be 'dino' or 'clip'")
    
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    if extractor_type == "dino":
        cls_token_feature = outputs.last_hidden_state[:, 0, :]
        patch_features = outputs.last_hidden_state[:, 1:, :]
    elif extractor_type == "clip":
        cls_token_feature = outputs.last_hidden_state[:, 0, :]
        patch_features = outputs.last_hidden_state[:, 1:, :]
    
    return cls_token_feature, patch_features
    
    
    pass
            
if __name__ == "__main__":
    # temp = torch.rand((320, 48, 48))
    
    # feat_svd(temp, top_k=5)
    
    image_paths = [
        "/home/zicheng/Projects/diffusion_base/data/example_imgs/Gothic.jpg",
        "/home/zicheng/Projects/diffusion_base/data/example_imgs/abstract.jpg"
    ]
    images_to_references(image_paths, max_len=10)