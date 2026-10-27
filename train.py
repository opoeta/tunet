# Windows training support

import os
import argparse
import math
import glob
#from glob import glob as VERY_UNIQUE_GLOB_ALIAS # Mantendo por seguranca heh
from PIL import Image, UnidentifiedImageError
import logging
import time
import random
import yaml
import copy
from types import SimpleNamespace
import itertools # For infinite dataloader cycle
import signal # For graceful shutdown signal handling
import re # For checkpoint pruning regex
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.utils import make_grid
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler, autocast
import lpips
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- Detect OS ---
CURRENT_OS = "Windows"

# --- DDP Setup ---
def setup_ddp():
    """Initialize the Windows Gloo process group when distributed launch is used."""
    if not dist.is_available():
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        print("INFO: PyTorch distributed is unavailable. Running in single-process mode on Windows.")
        return

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size == 1:
        logging.info("Running in single-process mode (World Size = 1).")
        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(local_rank)
                logging.debug("Single-process CUDA device set to cuda:%s", local_rank)
            except Exception as exc:
                logging.error("Could not select CUDA device cuda:%s: %s", local_rank, exc)
        else:
            logging.debug("CUDA is unavailable; using CPU.")
        return

    if not dist.is_initialized():
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", "29500")
        os.environ.update({
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(local_rank),
            "MASTER_ADDR": master_addr,
            "MASTER_PORT": master_port,
        })
        try:
            dist.init_process_group(
                backend="gloo",
                init_method=f"tcp://{master_addr}:{master_port}",
                world_size=world_size,
                rank=rank,
            )
            device_info = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
            if rank == 0:
                logging.info("Distributed initialized: rank %s/%s on %s (Gloo)", rank, world_size, device_info)
        except Exception as exc:
            logging.error("Distributed initialization failed on rank %s: %s", rank, exc, exc_info=True)
            raise RuntimeError(f"Distributed initialization failed on rank {rank}.") from exc

# --- DDP Cleanup & Helpers ---
def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def get_rank():
    if not dist.is_available() or not dist.is_initialized(): return 0
    return dist.get_rank()

def get_world_size():
    if not dist.is_available() or not dist.is_initialized(): return 1
    return dist.get_world_size()

def is_main_process():
    return get_rank() == 0

# --- UNet Model & Components (No changes needed here) ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels: mid_channels = out_channels
        mid_channels = max(1, mid_channels)
        self.d_block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.d_block(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.m_block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x): return self.m_block(x)

class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            conv_in_channels = in_channels + skip_channels
        else:
            up_out_channels = in_channels // 2
            self.up = nn.ConvTranspose2d(in_channels, up_out_channels, kernel_size=2, stride=2)
            conv_in_channels = up_out_channels + skip_channels
        conv_in_channels = max(1, conv_in_channels)
        out_channels = max(1, out_channels)
        self.conv = DoubleConv(conv_in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        if diffY != 0 or diffX != 0:
            pad_left = diffX // 2; pad_right = diffX - pad_left
            pad_top = diffY // 2; pad_bottom = diffY - pad_top
            x1 = nn.functional.pad(x1, [pad_left, pad_right, pad_top, pad_bottom])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.c_block = nn.Conv2d(max(1, in_channels), max(1, out_channels), kernel_size=1)
    def forward(self, x): return self.c_block(x)

class UNet(nn.Module):
    def __init__(self, n_ch=3, n_cls=3, hidden_size=64, bilinear=True):
        super().__init__()
        if n_ch <= 0 or n_cls <= 0 or hidden_size <= 0: raise ValueError("n_ch, n_cls, hidden_size must be positive")
        self.n_ch, self.n_cls, self.hidden_size, self.bilinear = n_ch, n_cls, hidden_size, bilinear
        h = hidden_size
        chs = {'enc1': max(1, h), 'enc2': max(1, h*2), 'enc3': max(1, h*4), 'enc4': max(1, h*8), 'bottle': max(1, h*16)}
        bottle_in_ch = chs['bottle']
        self.inc = DoubleConv(n_ch, chs['enc1'])
        self.down1 = Down(chs['enc1'], chs['enc2'])
        self.down2 = Down(chs['enc2'], chs['enc3'])
        self.down3 = Down(chs['enc3'], chs['enc4'])
        self.down4 = Down(chs['enc4'], bottle_in_ch)
        self.up1 = Up(bottle_in_ch, chs['enc4'], chs['enc4'], bilinear)
        self.up2 = Up(chs['enc4'],   chs['enc3'], chs['enc3'], bilinear)
        self.up3 = Up(chs['enc3'],   chs['enc2'], chs['enc2'], bilinear)
        self.up4 = Up(chs['enc2'],   chs['enc1'], chs['enc1'], bilinear)
        self.outc = OutConv(chs['enc1'], n_cls)

    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3); x5 = self.down4(x4)
        x = self.up1(x5, x4); x = self.up2(x, x3); x = self.up3(x, x2); x = self.up4(x, x1)
        return self.outc(x)

# --- Data Loading and Augmentation ---

# --- Augmentation Creation Function ---
def create_augmentations(augmentation_list):
    transforms = []
    uses_albumentations = False
    if not isinstance(augmentation_list, list):
        # Keep warning minimal
        # logging.warning(f"Augmentation list is not a list, using defaults.")
        return None

    for i, aug_config in enumerate(augmentation_list):
        if isinstance(aug_config, SimpleNamespace):
            if not hasattr(aug_config, '_target_'): logging.error(f"Aug {i}: Skip SimpleNamespace missing _target_"); continue
            target_str = aug_config._target_; params = {k: v for k, v in vars(aug_config).items() if k != '_target_'}
        elif isinstance(aug_config, dict):
            if '_target_' not in aug_config: logging.error(f"Aug {i}: Skip dict missing _target_"); continue
            target_str = aug_config['_target_']; params = {k: v for k, v in aug_config.items() if k != '_target_'}
        else: logging.error(f"Aug {i}: Skip invalid type {type(aug_config)}"); continue

        try:
            target_cls, is_torchvision, is_albumentation = None, False, False
            if target_str.startswith('torchvision.transforms.'):
                parts = target_str.split('.');
                if len(parts) == 3: cls_name = parts[-1]; target_cls = getattr(T, cls_name, None); is_torchvision = True
                else: logging.error(f"Aug {i}: Skip bad torchvision format {target_str}"); continue
                # Use DEBUG for mixing warning
                if is_torchvision and uses_albumentations: logging.debug(f"Aug {i}: Mixing Torchvision {cls_name} with Albumentations.")
            elif target_str.startswith('albumentations.'):
                parts = target_str.split('.')
                if len(parts) == 3 and parts[1] == 'pytorch': mod, cls_name = A.pytorch, parts[-1]
                elif len(parts) == 2: mod, cls_name = A, parts[-1]
                else: logging.error(f"Aug {i}: Skip bad albumentations format {target_str}"); continue
                target_cls = getattr(mod, cls_name, None); is_albumentation = True; uses_albumentations = True
            else: logging.error(f"Aug {i}: Skip unsupported lib {target_str}"); continue
            if target_cls is None: logging.error(f"Aug {i}: Class not found {target_str}"); continue

            try:
                transform_instance = target_cls(**params)
                transforms.append(transform_instance)
                # Use DEBUG for successful addition
                logging.debug(f"Added augmentation {i}: {target_str}")
            except Exception as e: logging.error(f"Aug {i}: Error instantiating {target_str}: {e}") # Keep error
        except Exception as e: logging.error(f"Aug {i}: Fail process {target_str}: {e}", exc_info=True) # Keep error for outer try

    if not transforms: return None
    # Use DEBUG for pipeline type confirmation
    if uses_albumentations: logging.debug(f"Creating Albumentations pipeline ({len(transforms)} steps)."); return A.Compose(transforms)
    else: logging.debug(f"Creating Torchvision pipeline ({len(transforms)} steps)."); return T.Compose(transforms)

# --- Augmented Dataset Class ---
class AugmentedImagePairSlicingDataset(Dataset):
    def __init__(self, src_dir, dst_dir, resolution, overlap_factor=0.0,
                 src_transforms=None, dst_transforms=None, shared_transforms=None,
                 final_transform=None):
        self.src_dir = os.path.abspath(src_dir)
        self.dst_dir = os.path.abspath(dst_dir)
        if not os.path.isdir(self.src_dir): raise FileNotFoundError(f"Src dir not found: {self.src_dir}")
        if not os.path.isdir(self.dst_dir): raise FileNotFoundError(f"Dst dir not found: {self.dst_dir}")
        self.resolution, self.overlap_factor = resolution, overlap_factor
        self.src_transforms, self.dst_transforms = src_transforms, dst_transforms
        self.shared_transforms, self.final_transform = shared_transforms, final_transform
        self.slice_info, self.skipped_count, self.processed_files, self.total_slices_generated = [], 0, 0, 0
        self.skipped_file_reasons = []
        overlap_pixels = int(resolution * overlap_factor); self.stride = max(1, resolution - overlap_pixels)
        src_glob_pattern = os.path.join(self.src_dir, '*.*')
        # Use DEBUG for file search pattern
        if is_main_process(): logging.debug(f"Dataset searching: {src_glob_pattern}")
        src_files = sorted(glob.glob(src_glob_pattern))
        if not src_files: logging.error(f"No source files found in '{self.src_dir}'"); # Keep Error
        # Use DEBUG for number of files found
        if is_main_process(): logging.debug(f"Found {len(src_files)} potential source files.")

        for i, src_path in enumerate(src_files):
            base_name = os.path.basename(src_path); dst_path = os.path.join(self.dst_dir, base_name)
            if not os.path.exists(dst_path): self.skipped_count += 1; self.skipped_file_reasons.append((base_name, "Dst missing")); continue
            try:
                with Image.open(src_path) as img_s, Image.open(dst_path) as img_d: w_s, h_s = img_s.size; w_d, h_d = img_d.size
                if (w_s, h_s) != (w_d, h_d): self.skipped_count += 1; self.skipped_file_reasons.append((base_name, f"Dims mismatch")); continue
                if w_s < resolution or h_s < resolution: self.skipped_count += 1; self.skipped_file_reasons.append((base_name, f"Too small")); continue
                num_slices_for_file = 0
                y_coords = list(range(0, max(0, h_s - resolution) + 1, self.stride)); x_coords = list(range(0, max(0, w_s - resolution) + 1, self.stride))
                if (h_s > resolution) and ((h_s - resolution) % self.stride != 0): y_coords.append(h_s - resolution)
                if (w_s > resolution) and ((w_s - resolution) % self.stride != 0): x_coords.append(w_s - resolution)
                unique_y = sorted(list(set(y_coords))); unique_x = sorted(list(set(x_coords)))
                for y in unique_y:
                    if not (0 <= y <= h_s - resolution): continue
                    for x in unique_x:
                        if not (0 <= x <= w_s - resolution): continue
                        crop_box = (x, y, x + resolution, y + resolution)
                        self.slice_info.append({'src_path': src_path, 'dst_path': dst_path, 'crop_box': crop_box}); num_slices_for_file += 1
                if num_slices_for_file > 0: self.processed_files += 1; self.total_slices_generated += num_slices_for_file
            except (FileNotFoundError, UnidentifiedImageError) as file_err:
                 self.skipped_count += 1; self.skipped_file_reasons.append((base_name, type(file_err).__name__));
                 # Keep Warning for bad files
                 if is_main_process(): logging.warning(f"Skipping {base_name}: {file_err}")
            except Exception as e:
                self.skipped_count += 1; self.skipped_file_reasons.append((base_name, f"Error: {type(e).__name__}"))
                # Keep Warning for other errors
                if is_main_process(): logging.warning(f"Skipping {base_name} due to error: {e}", exc_info=False)

        if is_main_process():
            # Keep INFO for overall summary
            logging.info(f"Dataset Init: Processed {self.processed_files}/{len(src_files)} files -> {self.total_slices_generated} slices. Skipped {self.skipped_count} files.")
            # Use DEBUG for skip reasons unless count is high? Maybe keep top N as WARNING.
            if self.skipped_count > 0:
                 limit = 5
                 logging.warning(f"--- Top {min(limit, self.skipped_count)} File Skip Reasons ---") # Keep Warning
                 for i, (name, reason) in enumerate(self.skipped_file_reasons):
                      if i >= limit: logging.warning(f"  ... ({self.skipped_count - limit} more)"); break # Keep Warning
                      logging.warning(f"  - {name}: {reason}") # Keep Warning
                 logging.warning("-----------------------------") # Keep Warning
        if len(self.slice_info) == 0:
             err_msg = f"CRITICAL: Dataset has 0 usable slices. "
             if self.processed_files > 0: err_msg += f"Processed {self.processed_files} files but generated no slices. Check resolution/overlap."
             elif self.skipped_count > 0: err_msg += f"All {self.skipped_count} potential files skipped."
             else: err_msg += f"No source files found/processed in '{self.src_dir}'."
             logging.error(err_msg) # Keep Error
             raise ValueError(err_msg)
        elif is_main_process():
             # Keep INFO for final confirmation
             logging.info(f"Dataset ready. Total usable slices: {len(self.slice_info)}")

    def __len__(self): return len(self.slice_info)
    def __getitem__(self, idx):
        try: info = self.slice_info[idx]; src_path, dst_path, crop_box = info['src_path'], info['dst_path'], info['crop_box']
        except IndexError: logging.error(f"Index {idx} out of bounds for slice_info (len {len(self.slice_info)})."); return None # Keep Error
        try: src_img = Image.open(src_path).convert('RGB'); dst_img = Image.open(dst_path).convert('RGB')
        except Exception as load_e: logging.error(f"Item {idx}: Img load error ({os.path.basename(src_path)}): {load_e}"); return None # Keep Error
        try:
            src_slice_pil = src_img.crop(crop_box); dst_slice_pil = dst_img.crop(crop_box)
            src_img.close(); dst_img.close()
            use_numpy = isinstance(self.shared_transforms, A.Compose) or isinstance(self.src_transforms, A.Compose) or isinstance(self.dst_transforms, A.Compose)
            if use_numpy: src_slice_current, dst_slice_current = np.array(src_slice_pil), np.array(dst_slice_pil)
            else: src_slice_current, dst_slice_current = src_slice_pil, dst_slice_pil

            if self.shared_transforms:
                if isinstance(self.shared_transforms, A.Compose): aug = self.shared_transforms(image=src_slice_current, mask=dst_slice_current); src_slice_current, dst_slice_current = aug['image'], aug['mask']
                elif isinstance(self.shared_transforms, T.Compose): seed = random.randint(0, 2**32-1); torch.manual_seed(seed); random.seed(seed); src_slice_current = self.shared_transforms(src_slice_current); torch.manual_seed(seed); random.seed(seed); dst_slice_current = self.shared_transforms(dst_slice_current)
            if self.src_transforms:
                if isinstance(self.src_transforms, A.Compose): src_slice_current = self.src_transforms(image=src_slice_current)['image']
                elif isinstance(self.src_transforms, T.Compose): src_slice_current = self.src_transforms(src_slice_current)
            if self.dst_transforms:
                if isinstance(self.dst_transforms, A.Compose): dst_slice_current = self.dst_transforms(image=dst_slice_current)['image']
                elif isinstance(self.dst_transforms, T.Compose): dst_slice_current = self.dst_transforms(dst_slice_current)

            if self.final_transform:
                if use_numpy: src_final_input, dst_final_input = Image.fromarray(src_slice_current), Image.fromarray(dst_slice_current)
                else: src_final_input, dst_final_input = src_slice_current, dst_slice_current
                src_tensor, dst_tensor = self.final_transform(src_final_input), self.final_transform(dst_final_input)
            else: # Manual conversion if no final transform
                if not use_numpy: src_slice_current, dst_slice_current = np.array(src_slice_current), np.array(dst_slice_current)
                src_tensor = torch.from_numpy(src_slice_current.transpose(2, 0, 1)).float() / 255.0; dst_tensor = torch.from_numpy(dst_slice_current.transpose(2, 0, 1)).float() / 255.0
                norm = T.Normalize(mean=NORM_MEAN.tolist(), std=NORM_STD.tolist()); src_tensor, dst_tensor = norm(src_tensor), norm(dst_tensor)
            return src_tensor, dst_tensor
        except Exception as e: logging.error(f"Item {idx}: Transform error ({os.path.basename(src_path)}): {e}", exc_info=False); return None # Keep Error

# --- Helper Functions ---
NORM_MEAN = torch.tensor([0.5, 0.5, 0.5]); NORM_STD = torch.tensor([0.5, 0.5, 0.5])
def denormalize(tensor):
    if tensor is None: return None
    try: mean = NORM_MEAN.to(tensor.device).view(1, 3, 1, 1); std = NORM_STD.to(tensor.device).view(1, 3, 1, 1); return torch.clamp(tensor * std + mean, 0, 1)
    except Exception as e: logging.error(f"Denormalize error: {e}"); return tensor
def cycle(iterable):
    iterator = iter(iterable)
    while True:
        try: yield next(iterator)
        except StopIteration: iterator = iter(iterable)

# --- Preview Capture/Saving ---
def save_previews(model, fixed_src_batch, fixed_dst_batch, output_dir, current_epoch, global_step, device, preview_save_count, preview_refresh_rate):
    if not is_main_process() or fixed_src_batch is None or fixed_dst_batch is None or fixed_src_batch.nelement() == 0: return
    num_grid_cols = 3; num_samples = min(fixed_src_batch.size(0), num_grid_cols)
    if num_samples == 0: return
    src_select, dst_select = fixed_src_batch[:num_samples].cpu(), fixed_dst_batch[:num_samples].cpu()
    model.eval(); device_type = device.type
    # Determine AMP usage for inference based on scaler status (safer)
    use_amp_inf = scaler.is_enabled() if 'scaler' in globals() and scaler is not None else False
    pred_select = None
    try:
        # on CUDA use autocast, otherwise just do a plain no_grad
        if device_type == 'cuda' and use_amp_inf:
            ctx = autocast(device_type=device_type, enabled=True)
        else:
            ctx = torch.no_grad()
        with ctx:
            src_dev = src_select.to(device)
            
            model_module = model.module if isinstance(model, DDP) else model
            predicted_batch = model_module(src_dev)
        pred_select = predicted_batch.cpu().float()
    except Exception as e: logging.error(f"Preview inference error (Step {global_step}): {e}"); model.train(); return # Keep Error
    model.train()
    src_denorm, pred_denorm, dst_denorm = denormalize(src_select), denormalize(pred_select), denormalize(dst_select)
    if src_denorm is None or pred_denorm is None or dst_denorm is None: logging.error(f"Preview denorm error (Step {global_step})"); return # Keep Error
    combined = [item for i in range(num_samples) for item in [src_denorm[i], dst_denorm[i], pred_denorm[i]]]
    if not combined: return
    try: grid_tensor = torch.stack(combined); grid = make_grid(grid_tensor, nrow=num_grid_cols, padding=2, normalize=False)
    except Exception as e: logging.error(f"Preview grid error (Step {global_step}): {e}"); returnçÎø¶‰Ëkºwµçq•…¹ÕÁ}‘‘À ¤ìÉ•ÑÕÉ¸4(4(€€€€Œ€´´´QÉ…¥¹¥¹œ1½½ÀY…É¥…‰±•Ì€´´´4(€€€±½‰…±}ÍÑ•À€ôÍÑ…ÉÑ}ÍÑ•Àì¥Ñ•É}•Á½ €ô½¹™¥œ¹ÑÉ…¥¹¥¹œ¹¥Ñ•É…Ñ¥½¹Í}Á•É}•Á½ 4(€€€™¥á•‘}ÍÉŒ°™¥á•‘}‘ÍĞ€ô9½¹”°9½¹”ìÁÉ•Ù¥•İ}½Õ¹Ğ€ô€À4(€€€•Á}°Ä°•Á}±Á¥ÁÌ°•Á}ÍÑ•ÁÌ€ô€À¸À°€À¸À°€À4(€€€‰…Ñ¡}Ñ¥µ•Ì€ômt4(4(€€€€Œ€´´´…ÁÑÕÉ”%¹¥Ñ¥…°AÉ•Ù¥•Ü	…Ñ €´´´4(€€€ÁÉ•Ù¥•İ}¥¹Ñ•ÉÙ…°€ô½¹™¥œ¹±½¥¹œ¹ÁÉ•Ù¥•İ}‰…Ñ¡}¥¹Ñ•ÉÙ…°4(€€€¥˜¥Í}µ…¥¹}ÁÉ½•ÍÌ ¤…¹ÁÉ•Ù¥•İ}¥¹Ñ•ÉÙ…°€ø€Àè4(€€€€€€€™¥á•‘}ÍÉŒ°™¥á•‘}‘ÍĞ€ô…ÁÑÕÉ•}ÁÉ•Ù¥•İ}‰…Ñ ¡½¹™¥œ°ÍÉ}ÑÉ…¹Í™½ÉµÌ°‘ÍÑ}ÑÉ…¹Í™½ÉµÌ°Í¡…É•‘}ÑÉ…¹Í™½ÉµÌ°ÍÑ…¹‘…É‘}ÑÉ…¹Í™½É´¤4(€€€€€€€¥˜™¥á•‘}ÍÉŒ¥Ì9½¹”è±½¥¹œ¹İ…É¹¥¹œ ‰%¹¥Ñ¥…°ÁÉ•Ù¥•Ü‰…Ñ …ÁÑÕÉ”™…¥±•¸ˆ¤4(4(€€€€Œ€´´´5…¥¸QÉ…¥¹¥¹œ1½½À€´´´4(€€€µ½‘•°¹ÑÉ…¥¸ ¤4(€€€ÑÉäè4(€€€€€€€İ¡¥±”QÉÕ”è4(€€€€€€€€€€€¥˜Í¡ÕÑ‘½İ¹}É•ÅÕ•ÍÑ•è4(€€€€€€€€€€€€€€€¥˜¥Í}µ…¥¹}ÁÉ½•ÍÌ ¤è±½¥¹œ¹¥¹™¼¡˜‰M¡ÕÑ‘½İ¸É•ÅÕ•ÍÑ• ÍÑ•Àí±½‰…±}ÍÑ•Áô¸ˆ¤4(€€€€€€€€€€€€€€€‰É•…¬4(4(€€€€€€€€€€€±½½Á}¥Ñ•É}ÍÑ…ÉĞ€ôÑ¥µ”¹Ñ¥µ” ¤4(€€€€€€€€€€€ÕÉÉ•¹Ñ}•Á}¥‘à€ô±½‰…±}ÍÑ•À€¼¼¥Ñ•É}•Á½ ì¥Í}¹•İ}•Á½ €ô€¡±½‰…±}ÍÑ•À€”¥Ñ•É}•Á½ €ôô€À¤4(€€€€€€€€€€€¥˜¥Í}¹•İ}•Á½ è4(€€€€€€€€€€€€€€€¥˜¥Í¥¹ÍÑ…¹”¡‘…Ñ…±½…‘•È¹Í…µÁ±•È°¥ÍÑÉ¥‰ÕÑ•‘M…µÁ±•È¤è‘…Ñ…±½…‘•È¹Í…µÁ±•È¹Í•Ñ}•Á½ ¡ÕÉÉ•¹Ñ}•Á}¥‘à¤4(€€€€€€€€€€€€€€€¥˜¥Í}µ…¥¹}ÁÉ½•ÍÌ ¤…¹€¡±½‰…±}ÍÑ•À€ø€À½ÈÍÑ…ÉÑ}ÍÑ•À€ø€À¤è±½¥¹œ¹¥¹™¼¡˜ˆ´´´Á½ íÕÉÉ•¹Ñ}•Á}¥‘à€¬€ÅôMÑ…ÉĞ€¡MÑ•Àí±½‰…±}ÍÑ•Áô¤€´´´ˆ¤4(€€€€€€€€€€€€€€€•Á}°Ä°•Á}±Á¥ÁÌ°•Á}ÍÑ•ÁÌ€ô€À¸À°€À¸À°€À4(4(€€€€€€€€€€€‘…Ñ…}±½…‘}ÍÑ…ÉĞ€ôÑ¥µ”¹Ñ¥µ” ¤4(€€€€€€€€€€€ÑÉäè‰…Ñ €ô¹•áĞ¡‘…Ñ…±½…‘•É}¥Ñ•È¤4(€€€€€€€€€€€•á•ÁĞá•ÁÑ¥½¸…Ì”è±½¥¹œ¹•ÉÉ½È¡˜‰Mí±½‰…±}ÍÑ•Áôè	…Ñ ±½…•ÉÉ½Èèí•ô°Í­¥ÁÁ¥¹œ¸ˆ°•á}¥¹™¼õQÉÕ”¤ì±½‰…±}ÍÑ•À€¬ô€Äì½¹Ñ¥¹Õ”4(€€€€€€€€€€€¥˜‰…Ñ ¥Ì9½¹”è±½¥¹œ¹İ…É¹¥¹œ¡˜‰Mí±½‰…±}ÍÑ•ÁôèM­¥ÁÁ•‰…Ñ €¡½±±…Ñ¥½¸•ÉÉ½È¤¸ˆ¤ì±½‰…±}ÍÑ•À€¬ô€Äì½¹Ñ¥¹Õ”4(€€€€€€€€€€€€ÍÉŒ°‘ÍĞ€ô‰…Ñ ì‘…Ñ…}±½…‘}Ñ¥µ”€ôÑ¥µ”¹Ñ¥µ” ¤€´‘…Ñ…}±½…‘}ÍÑ…ÉĞ4(€€€€€€€€€€€ÍÉŒ°‘ÍĞ€ô‰…Ñ 4(€€€€€€€€€€€‘…Ñ…}±½…‘}Ñ¥µ”€ôÑ¥µ”¹Ñ¥µ” ¤€´‘…Ñ…}±½…‘}ÍÑ…ÉĞ4(4(€€€€€€€€€€€€ŒƒŠRŠR Í•¹¥¹ÁÕÑÌÑ¼Ñ¡”Í…µ”‘•Ù¥”…ÌÑ¡”µ½‘•°ƒŠRŠR 4(€€€€€€€€€€€ÑÉ…¹Í™•É}ÍÑ…ÉĞ€ôÑ¥µ”¹Ñ¥µ” ¤4(€€€€€€€€€€€ÍÉŒ€ôÍÉŒ¹Ñ¼¡‘•Ù¥”¤4(€€€€€€€€€€€‘ÍĞ€ô‘ÍĞ¹Ñ¼¡‘•Ù¥”¤4(€€€€€€€€€€€ÑÉ…¹Í™•É}Ñ¥µ”€ôÑ¥µ”¹Ñ¥µ” ¤€´ÑÉ…¹Í™•É}ÍÑ…ÉĞ4(4(€€€€€€€€€€€€Œ€´´´½Éİ…É°1½ÍÌ°	…­İ…É°=ÁÑ¥µ¥é”€´´´4(€€€€€€€€€€€½µÁÕÑ•}ÍÑ…ÉĞ€ôÑ¥µ”¹Ñ¥µ” ¤4(€€€€€€€€€€€½ÁÑ¥µ¥é•È¹é•É½}É…¡Í•Ñ}Ñ½}¹½¹”õQÉÕ”¤4(4(€€€€€€€€€€€€Œ€ğğğÔÍ•¤¸¸¸€øøø4(€€€€€€€€€€€€Œ=¹±äÕÍ”…ÕÑ½…ÍĞ½¹Ñ•áĞ¥˜5@¥Ì•™™•Ñ¥Ù•±ä•¹…‰±•€¡¤¹”¸°½¸U¤4(€€€€€€€€€€€¥˜ÕÍ•}…µÁ}•™˜è4(€€€€€€€€€€€€€€€İ¥Ñ …ÕÑ½…ÍĞ¡‘•Ù¥•}ÑåÁ”õ‘•Ù¥•}ÑåÁ”°•¹…‰±•õQÉÕ”¤è4(€€€€€€€€€€€€€€€€€€€½ÕĞ€ôµ½‘•°¡ÍÉŒ¤4(€€€€€€€€€€€€€€€€€€€°Ä€ôÉ¥Ñ•É¥½¹}°Ä¡½ÕĞ°‘ÍĞ¤4(€€€€€€€€€€€€€€€€€€€±À€ôÑ½É ¹Ñ•¹Í½È À¸À°‘•Ù¥”õ‘•Ù¥”¤4(€€€€€€€€€€€€€€€€€€€¥˜ÕÍ•}±Á¥ÁÌ…¹±½ÍÍ}™¹}±Á¥ÁÌè4(€€€€€€€€€€€€€€€€€€€€€€€ÑÉäè±À€ô±½ÍÍ}™¹}±Á¥ÁÌ¡½ÕĞ°‘ÍĞ¤¹µ•…¸ ¤4(€€€€€€€€€€€€€€€€€€€€€€€•á•ÁĞá•ÁÑ¥½¸…Ì•}±Àè±À€ôÑ½É ¹Ñ•¹Í½È À¸À°‘•Ù¥”õ‘•Ù¥”¤4(€€€€€€€€€€€€€€€€€€€€Œ1½ÍÌ…±Õ±…Ñ¥½¸¥¹Í¥‘”…ÕÑ½…ÍĞ½¹Ñ•áĞ4(€€€€€€€€€€€€€€€€€€€±½ÍÌ€ô°Ä€¬½¹™¥œ¹ÑÉ…¥¹¥¹œ¹±…µ‰‘…}±Á¥ÁÌ€¨±À4(€€€€€€€€€€€•±Í”è4(€€€€€€€€€€€€€€€€ŒIÕ¸ÍÑ…¹‘…É™½Éİ…ÉÁ…ÍÌİ¡•¸5@¥Ì‘¥Í…‰±•€¡AT¤¸4(€€€€€€€€€€€€€€€½ÕĞ€ôµ½‘•°¡ÍÉŒ¤4(€€€€€€€€€€€€€€€°Ä€ôÉ¥Ñ•É¥½¹}°Ä¡½ÕĞ°‘ÍĞ¤4(€€€€€€€€€€€€€€€±À€ôÑ½É ¹Ñ•¹Í½È À¸À°‘•Ù¥”õ‘•Ù¥”¤4(€€€€€€€€€€€€€€€¥˜ÕÍ•}±Á¥ÁÌ…¹±½ÍÍ}™¹}±Á¥ÁÌè4(€€€€€€€€€€€€€€€€€€€ÑÉäè±À€ô±½ÍÍ}™¹}±Á¥ÁÌ¡½ÕĞ°‘ÍĞ¤¹µ•…¸ ¤4(€€€€€€€€€€€€€€€€€€€•á•ÁĞá•ÁÑ¥½¸…Ì•}±Àè±À€ôÑ½É ¹Ñ•¹Í½È À¸À°‘•Ù¥”õ‘•Ù¥”¤4(€€€€€€€€€€€€€€€€Œ1½ÍÌ…±Õ±…Ñ¥½¸½ÕÑÍ¥‘”…ÕÑ½…ÍĞ½¹Ñ•áĞ4(€€€€€€€€€€€€€€€±½ÍÌ€ô°Ä€¬½¹™¥œ¹ÑÉ…¥¹¥¹œ¹±…µ‰‘…}±Á¥ÁÌ€¨±À4(€€€€€€€€€€€€Œ€ğğğ5=%%UQ=MPUM9€øøø4(4(€€€€€€€€€€€¥˜¹½ĞÑ½É ¹¥Í™¥¹¥Ñ”¡±½ÍÌ¤è4(€€€€€€€€€€€€€€€±½¥¹œ¹•ÉÉ½È¡˜‰Mí±½‰…±}ÍÑ•Áôè9…8½%¹˜±½ÍÌ€¡í±½ÍÌ¹¥Ñ•´ ¥ô¤„M­¥ÀÕÁ‘…Ñ”¸ˆ¤4(€€€€€€€€€€€€€€€±½‰…±}ÍÑ•À€¬ô€Ä4(€€€€€€€€€€€€€€€½¹Ñ¥¹Õ”€Œ-••ÀÉÉ½È4(4(€€€€€€€€€€€€ŒM…±•ÈÍÑ•À½¹±ä¥˜•™™•Ñ¥Ù•±ä•¹…‰±•4(€€€€€€€€€€€¥˜ÕÍ•}…µÁ}•™˜è4(€€€€€€€€€€€€€€€€Í…±•È¹Í…±”¡±½ÍÌ¤¹‰…­İ…É ¤4(€€€€€€€€€€€€€€€€Í…±•È¹ÍÑ•À¡½ÁÑ¥µ¥é•È¤4(€€€€€€€€€€€€€€€€Í…±•È¹ÕÁ‘…Ñ” ¤4(€€€€€€€€€€€•±Í”è4(€€€€€€€€€€€€€€€€±½ÍÌ¹‰…­İ…É ¤€ŒMÑ…¹‘…É‰…­İ…É¥˜5@¥Ì½™˜4(€€€€€€€€€€€€€€€€½ÁÑ¥µ¥é•È¹ÍÑ•À ¤€ŒMÑ…¹‘…É½ÁÑ¥µ¥é•ÈÍÑ•À4(4(€€€€€€€€€€€½µÁÕÑ•}Ñ¥µ”€ôÑ¥µ”¹Ñ¥µ” ¤€´½µÁÕÑ•}ÍÑ…ÉĞ4(4(4(€€€€€€€€€€€½µÁÕÑ•}Ñ¥µ”€ôÑ¥µ”¹Ñ¥µ” ¤€´½µÁÕÑ•}ÍÑ…ÉĞ4(4(€€€€€€€€€€€‰…Ñ¡}°Ä€ô°Ä¹‘•Ñ…  ¤¹¥Ñ•´ ¤ì‰…Ñ¡}±À€ô±À¹‘•Ñ…  ¤¹¥Ñ•´ ¤¥˜ÕÍ•}±Á¥ÁÌ•±Í”€À¸À4(€€€€€€€€€€€¥˜İ½É±‘}Í¥é”€ø€Äè4(€€€€€€€€€€€€€€€°Å}Ğ€ôÑ½É ¹Ñ•¹Í½È¡‰…Ñ¡}°Ä°‘•Ù¥”õ‘•Ù¥”¤ì±Á}Ğ€ôÑ½É ¹Ñ•¹Í½È¡‰…Ñ¡}±À°‘•Ù¥”õ‘•Ù¥”¤4(€€€€€€€€€€€€€€€‘¥ÍĞ¹…±±}É•‘Õ”¡°Å}Ğ°½Àõ‘¥ÍĞ¹I•‘Õ•=À¹Y¤ì‘¥ÍĞ¹…±±}É•‘Õ”¡±Á}Ğ°½Àõ‘¥ÍĞ¹I•‘Õ•=À¹Y¤4(€€€€€€€€€€€€€€€‰…Ñ¡}°Ä€ô°Å}Ğ¹¥Ñ•´ ¤ì‰…Ñ¡}±À€ô±Á}Ğ¹¥Ñ•´ ¤4(€€€€€€€€€€€•Á}°Ä€¬ô‰…Ñ¡}°Äì•Á}±Á¥ÁÌ€¬ô‰…Ñ¡}±Àì•Á}ÍÑ•ÁÌ€¬ô€Ä4(4(€€€€€€€€€€€¥Ñ•É}Ñ¥µ”€ôÑ¥µ”¹Ñ¥µ” ¤€´±½½Á}¥Ñ•É}ÍÑ…ÉĞì‰…Ñ¡}Ñ¥µ•Ì¹…ÁÁ•¹¡¥Ñ•É}Ñ¥µ”¤ì‰…Ñ¡}Ñ¥µ•Ì€ô‰…Ñ¡}Ñ¥µ•Íl´ÄÀÀét4(€€€€€€€€€€€…Ù}Ñ¥µ”€ôÍÕ´¡‰…Ñ¡}Ñ¥µ•Ì¤€¼±•¸¡‰…Ñ¡}Ñ¥µ•Ì¤¥˜‰…Ñ¡}Ñ¥µ•Ì•±Í”€À¸À4(4(€€€€€€€€€€€±½‰…±}ÍÑ•À€¬ô€Äì™¥¹…±}±½‰…±}ÍÑ•À€ô±½‰…±}ÍÑ•À4(4(€€€€€€€€€€€¥˜¥Í}µ…¥¹}ÁÉ½•ÍÌ ¤…¹±½‰…±}ÍÑ•À€”½¹™¥œ¹±½¥¹œ¹±½}¥¹Ñ•ÉÙ…°€ôô€Àè4(€€€€€€€€€€€€€€€ÕÉÉ•¹Ñ}±È€ô½ÁÑ¥µ¥é•È¹Á…É…µ}É½ÕÁÍlÁul±Èt4(€€€€€€€€€€€€€€€±½¥¹œ¹¥¹™¼¡˜ŸŠ˜€¡éí‘…Ñ…}±½…‘}Ñ¥µ”è¸Í™ôPéíÑÉ…¹Í™•É}Ñ¥µ”è¸Í™ôéí½µÁÕÑ•}Ñ¥µ”è¸Í™ô¤œ¤4(€€€€€€€€€€€€€€€…Ù}•Á}°Ä€ô•Á}°Ä€¼•Á}ÍÑ•ÁÌ¥˜•Á}ÍÑ•ÁÌ€ø€À•±Í”€À¸À4(€€€€€€€€€€€€€€€…Ù}•Á}±À€ô•Á}±Á¥ÁÌ€¼•Á}ÍÑ•ÁÌ¥˜ÕÍ•}±Á¥ÁÌ…¹•Á}ÍÑ•ÁÌ€ø€À•±Í”€À¸À4(€€€€€€€€€€€€€€€ÍÑ•ÁÍ}¥¹}•À€ô±½‰…±}ÍÑ•À€”¥Ñ•É}•Á½ ½È¥Ñ•É}•Á½ 4(€€€€€€€€€€€€€€€±½}µÍœ€ô€¡˜Á½¡míÕÉÉ•¹Ñ}•Á}¥‘à€¬€ÅõtMÑ•Ámí±½‰…±}ÍÑ•Áõt€¡íÍÑ•ÁÍ}¥¹}•Áô½í¥Ñ•É}•Á½¡ô¤°€œ4(€€€€€€€€€€€€€€€€€€€€€€€€€€˜0Äéí‰…Ñ¡}°Äè¸Ñ™ô¡Ùœéí…Ù}•Á}°Äè¸Ñ™ô¤œ¤4(€€€€€€€€€€€€€€€¥˜ÕÍ•}±Á¥ÁÌè±½}µÍœ€¬ô€¡˜œ°1A%ALéí‰…Ñ¡}±Àè¸Ñ™ô¡Ùœéí…Ù}•Á}±Àè¸Ñ™ô¤œ¤4(€€€€€€€€€€€€€€€±½}µÍœ€¬ô€¡˜œ°1HéíÕÉÉ•¹Ñ}±Èè¸Å•ô°P½MÑ•Àéí…Ù}Ñ¥µ”è¸Í™õÌ€œ4(€€€€€€€€€€€€€€€€€€€€€€€€€€€˜œ¡éí‘…Ñ…}±½…‘}Ñ¥µ”è¸Í™ôPéíÑÉ…¹Í™•É}Ñ¥µ”è¸Í™ôéí½µÁÕÑ•}Ñ¥µ”è¸Í™ô¤œ¤4(€€€€€€€€€€€€€€€±½¥¹œ¹¥¹™¼¡±½}µÍœ¤4(4(€€€€€€€€€€€¥˜¥Í}µ…¥¹}ÁÉ½•ÍÌ ¤…¹ÁÉ•Ù¥•İ}¥¹Ñ•ÉÙ…°€ø€À…¹±½‰…±}ÍÑ•À€”ÁÉ•Ù¥•İ}¥¹Ñ•ÉÙ…°€ôô€Àè4(€€€€€€€€€€€€€€€É•™É•Í €ô€¡™¥á•‘}ÍÉŒ¥Ì9½¹”¤½È€¡½¹™¥œ¹±½¥¹œ¹ÁÉ•Ù¥•İ}É•™É•Í¡}É…Ñ”€ø€À…¹ÁÉ•Ù¥•İ}½Õ¹Ğ€ø€À…¹€¡ÁÉ•Ù¥•İ}½Õ¹Ğ€”½¹™¥œ¹±½¥¹œ¹ÁÉ•Ù¥•İ}É•™É•Í¡}É…Ñ”€ôô€À¤¤4(€€€€€€€€€€€€€€€¥˜É•™É•Í è4(€€€€€€€€€€€€€€€€€€€€¹•İ}ÍÉŒ°¹•İ}‘ÍĞ€ô…ÁÑÕÉ•}ÁÉ•Ù¥•İ}‰…Ñ ¡½¹™¥œ°ÍÉ}ÑÉ…¹Í™½ÉµÌ°‘ÍÑ}ÑÉ…¹Í™½ÉµÌ°Í¡…É•‘}ÑÉ…¹Í™½ÉµÌ°ÍÑ…¹‘…É‘}ÑÉ…¹Í™½É´¤4(€€€€€€€€€€€€€€€€€€€€¥˜¹•İ}ÍÉŒ¥Ì¹½Ğ9½¹”è™¥á•‘}ÍÉŒ°™¥á•‘}‘ÍĞ€ô¹•İ}ÍÉŒ°¹•İ}‘ÍĞ4(€€€€€€€€€€€€€€€€€€€€•±¥˜™¥á•‘}ÍÉŒ¥Ì9½¹”è±½¥¹œ¹İ…É¹¥¹œ¡˜‰Mí±½‰…±}ÍÑ•ÁôèAÉ•Ù¥•ÜÉ•™É•Í ™…¥±•€¡¹¼¥¹¥Ñ¥…°¤¸ˆ¤4(€€€€€€€€€€€€€€€€€€€€•±Í”è±½¥¹œ¹İ…É¹¥¹œ¡˜‰Mí±½‰…±}ÍÑ•ÁôèAÉ•Ù¥•ÜÉ•™É•Í ™…¥±•°ÕÍ¥¹œ½±¸ˆ¤4(€€€€€€€€€€€€€€€¥˜™¥á•‘}ÍÉŒ¥Ì¹½Ğ9½¹”èÍ…Ù•}ÁÉ•Ù¥•İÌ¡µ½‘•°°™¥á•‘}ÍÉŒ°™¥á•‘}‘ÍĞ°½¹™¥œ¹‘…Ñ„¹½ÕÑÁÕÑ}‘¥È°ÕÉÉ•¹Ñ}•Á}¥‘à°±½‰…±}ÍÑ•À°‘•Ù¥”°ÁÉ•Ù¥•İ}½Õ¹Ğ°½¹™¥œ¹±½¥¹œ¹ÁÉ•Ù¥•İ}É•™É•Í¡}É…Ñ”¤ìÁÉ•Ù¥•İ}½Õ¹Ğ€¬ô€Ä4(€€€€€€€€€€€€€€€•±¥˜ÁÉ•Ù¥•İ}½Õ¹Ğ€ôô€Àè±½¥¹œ¹İ…É¹¥¹œ¡˜‰Mí±½‰…±}ÍÑ•ÁôèM­¥ÁÁ¥¹œÁÉ•Ù¥•Ü€¡¹¼‰…Ñ ¤¸ˆ¤4(4(€€€€€€€€€€€Í…Ù•}¥¹Ñ•ÉÙ…°€ô½¹™¥œ¹Í…Ù¥¹œ¹Í…Ù•}¥Ñ•É…Ñ¥½¹Í}¥¹Ñ•ÉÙ…°ìÍ…Ù•}¹½Ü€ô€¡Í…Ù•}¥¹Ñ•ÉÙ…°€ø€À…¹±½‰…±}ÍÑ•À€”Í…Ù•}¥¹Ñ•ÉÙ…°€ôô€À¤4(€€€€€€€€€€€•Á½¡}•¹‘}¹½Ü€ô€¡±½‰…±}ÍÑ•À€”¥Ñ•É}•Á½ €ôô€À¤…¹±½‰…±}ÍÑ•À€ø€À4(€€€€€€€€€€€¥˜¥Í}µ…¥¹}ÁÉ½•ÍÌ ¤…¹€¡Í…Ù•}¹½Ü½È•Á½¡}•¹‘}¹½Ü¤è4(€€€€€€€€€€€€€€€€­ÁÑ}•À€ô±½‰…±}ÍÑ•À€¼¼¥Ñ•É}•Á½ ì•Á}­ÁÑ}Á…Ñ €ô½Ì¹Á…Ñ ¹©½¥¸¡½¹™¥œ¹‘…Ñ„¹½ÕÑÁÕÑ}‘¥È°˜ÑÕ¹•Ñ}•Á½¡}í­ÁÑ}•ÀèÀå‘ô¹ÁÑ œ¤¥˜•Á½¡}•¹‘}¹½Ü•±Í”9½¹”4(€€€€€€€€€€€€€€€€±…Ñ•ÍÑ}Á…Ñ €ô½Ì¹Á…Ñ ¹©½¥¸¡½¹™¥œ¹‘…Ñ„¹½ÕÑÁÕÑ}‘¥È°€ÑÕ¹•Ñ}±…Ñ•ÍĞ¹ÁÑ œ¤ì™}‘¥Ğ€ô½¹™¥}Ñ½}‘¥Ğ¡½¹™¥œ¤4(€€€€€€€€€€€€€€€€µ}ÍÑ…Ñ”€ôµ½‘•°¹µ½‘Õ±”¹ÍÑ…Ñ•}‘¥Ğ ¤¥˜¥Í¥¹ÍÑ…¹”¡µ½‘•°°@¤•±Í”µ½‘•°¹ÍÑ…Ñ•}‘¥Ğ ¤4(€€€€€€€€€€€€€€€€€ŒM…Ù”Í…±•ÈÍÑ…Ñ”½¹±ä¥˜5@İ…Ì•™™•Ñ¥Ù•±äÕÍ•4(€€€€€€€€€€€€€€€€Í…±•É}ÍÑ…Ñ”€ôÍ…±•È¹ÍÑ…Ñ•}‘¥Ğ ¤¥˜ÕÍ•}…µÁ}•™˜•±Í”9½¹”4(€€€€€€€€€€€€€€€€­ÁÑ}‘…Ñ„€ôì•Á½ œè­ÁÑ}•À°€±½‰…±}ÍÑ•Àœè±½‰…±}ÍÑ•À°€µ½‘•±}ÍÑ…Ñ•}‘¥Ğœèµ}ÍÑ…Ñ”°€½ÁÑ¥µ¥é•É}ÍÑ…Ñ•}‘¥Ğœè½ÁÑ¥µ¥é•È¹ÍÑ…Ñ•}‘¥Ğ ¤°4(€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Í…±•É}ÍÑ…Ñ•}‘¥ĞœèÍ…±•É}ÍÑ…Ñ”°€½¹™¥œœè™}‘¥Ğ°€•™™•Ñ¥Ù•}µ½‘•±}Í¥é”œè•™™}Í¥é•ô4(€€€€€€€€€€€€€€€€ÑÉäè4(€€€€€€€€€€€€€€€€€€€€É•…Í½¸€ô€‰¥¹Ñ•ÉÙ…°ˆ¥˜Í…Ù•}¹½Ü•±Í”€‰•Á½ •¹ˆ4(€€€€€€€€€€€€€€€€€€€€Ñ½É ¹Í…Ù”¡­ÁÑ}‘…Ñ„°±…Ñ•ÍÑ}Á…Ñ ¤4(€€€€€€€€€€€€€€€€€€€€±½¥¹œ¹¥¹™¼¡˜‰M…Ù•±…Ñ•ÍĞ¡•­Á½¥¹Ğ€¡íÉ•…Í½¹ô¤ MÑ•Àí±½‰…±}ÍÑ•Áôˆ¤4(€€€€€€€€€€€€€€€€€€€€¥˜•Á}­ÁÑ}Á…Ñ è4(€€€€€€€€€€€€€€€€€€€€€€€€€Ñ½É ¹Í…Ù”¡­ÁÑ}‘…Ñ„°•Á}­ÁÑ}Á…Ñ ¤4(€€€€€€€€€€€€€€€€€€€€€€€€€±½¥¹œ¹¥¹™¼¡˜‰M…Ù••Á½ ¡•­Á½¥¹Ğèí½Ì¹Á…Ñ ¹‰…Í•¹…µ”¡•Á}­ÁÑ}Á…Ñ ¥ôˆ¤4(€€€€€€€€€€€€€€€€€€€€€€€€€¥˜¡…Í…ÑÑÈ¡½¹™¥œ¹Í…Ù¥¹œ°€­••Á}±…ÍÑ}¡•­Á½¥¹ÑÌœ¤…¹½¹™¥œ¹Í…Ù¥¹œ¹­••Á}±…ÍÑ}¡•­Á½¥¹ÑÌ€øô€Àè4(€€€€€€€€€€€€€€€€€€€€€€€€€€€€€ÁÉÕ¹•}¡•­Á½¥¹ÑÌ¡½¹™¥œ¹‘…Ñ„¹½ÕÑÁÕÑ}‘¥È°½¹™¥œ¹Í…Ù¥¹œ¹­••Á}±…ÍÑ}¡•­Á½¥¹ÑÌ¤4(€€€€€€€€€€€€€€€€•á•ÁĞá•ÁÑ¥½¸…Ì”è±½¥¹œ¹•ÉÉ½È¡˜‰¡•­Á½¥¹ĞÍ…Ù”™…¥±• MÑ•Àí±½‰…±}ÍÑ•Áôèí•ôˆ°•á}¥¹™¼õQÉÕ”¤4(€€€€€€€€Œ€´´´¹½˜QÉ…¥¹¥¹œ1½½À€´´´4(€€€€€€€ÑÉ…¥¹¥¹}ÍÕ•ÍÍ™Õ°€ôQÉÕ”4(€€€•á•ÁĞ-•å‰½…É‘%¹Ñ•ÉÉÕÁĞè4(€€€€€€€¥˜¥Í}µ…¥¹}ÁÉ½•ÍÌ ¤è±½¥¹œ¹İ…É¹¥¹œ ‰-•å‰½…É‘%¹Ñ•ÉÉÕÁĞ¸M¡ÕÑÑ¥¹œ‘½İ¸¸¸¸ˆ¤4(€€€€€€€¥˜¹½ĞÍ¡ÕÑ‘½İ¹}É•ÅÕ•ÍÑ•è¡…¹‘±•}Í¥¹…°¡Í¥¹…°¹M%%9P°9½¹”¤4(€€€•á•ÁĞá•ÁÑ¥½¸…Ì±½½Á}•ÉÈè4(€€€€€€€±½¥¹œ¹•ÉÉ½È ‰Q0QÉ…¥¹¥¹œ±½½À•ÉÉ½Èèˆ°•á}¥¹™¼õQÉÕ”¤ì4(€€€€€€€Í¡ÕÑ‘½İ¹}É•ÅÕ•ÍÑ•€ôQÉÕ”4(€€€™¥¹…±±äè4(€€€€€€€¥˜Í¡ÕÑ‘½İ¹}É•ÅÕ•ÍÑ•…¹¥Í}µ…¥¹}ÁÉ½•ÍÌ ¤è4(€€€€€€€€€€€±½¥¹œ¹¥¹™¼¡˜‰A•É™½Éµ¥¹œ™¥¹…°¡•­Á½¥¹ĞÍ…Ù”€¡MÑ…Ñ” MÑ•Àí™¥¹…±}±½‰…±}ÍÑ•Áô¤¸¸¸ˆ¤4(€€€€€€€€€€€ÑÉäè4(€€€€€€€€€€€€€€€€™¥¹…±}•À€ô™¥¹…±}±½‰…±}ÍÑ•À€¼¼¥Ñ•É}•Á½ ì±…Ñ•ÍÑ}Á…Ñ €ô½Ì¹Á…Ñ ¹©½¥¸¡½¹™¥œ¹‘…Ñ„¹½ÕÑÁÕÑ}‘¥È°€ÑÕ¹•Ñ}±…Ñ•ÍĞ¹ÁÑ œ¤ì™}‘¥Ğ€ô½¹™¥}Ñ½}‘¥Ğ¡½¹™¥œ¤4(€€€€€€€€€€€€€€€€¥˜µ½‘•°…¹½ÁÑ¥µ¥é•È…¹Í…±•Èè4(€€€€€€€€€€€€€€€€€€€€µ}ÍÑ…Ñ”€ôµ½‘•°¹µ½‘Õ±”¹ÍÑ…Ñ•}‘¥Ğ ¤¥˜¥Í¥¹ÍÑ…¹”¡µ½‘•°°@¤•±Í”µ½‘•°¹ÍÑ…Ñ•}‘¥Ğ ¤4(€€€€€€€€€€€€€€€€€€€€Í…±•É}ÍÑ…Ñ”€ôÍ…±•È¹ÍÑ…Ñ•}‘¥Ğ ¤¥˜ÕÍ•}…µÁ}•™˜•±Í”9½¹”€Œ¡•¬•™™•Ñ¥Ù”ÕÍ”4(€€€€€€€€€€€€€€€€€€€€™¥¹…±}‘…Ñ„€ôì•Á½ œè™¥¹…±}•À°€±½‰…±}ÍÑ•Àœè™¥¹…±}±½‰…±}ÍÑ•À°€µ½‘•±}ÍÑ…Ñ•}‘¥Ğœèµ}ÍÑ…Ñ”°€½ÁÑ¥µ¥é•É}ÍÑ…Ñ•}‘¥Ğœè½ÁÑ¥µ¥é•È¹ÍÑ…Ñ•}‘¥Ğ ¤°4(€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Í…±•É}ÍÑ…Ñ•}‘¥ĞœèÍ…±•É}ÍÑ…Ñ”°€½¹™¥œœè™}‘¥Ğ°€•™™•Ñ¥Ù•}µ½‘•±}Í¥é”œè•™™}Í¥é•ô4(€€€€€€€€€€€€€€€€€€€€Ñ½É ¹Í…Ù”¡™¥¹…±}‘…Ñ„°±…Ñ•ÍÑ}Á…Ñ ¤ì±½¥¹œ¹¥¹™¼¡˜‰¥¹…°¡•­Á½¥¹ĞÍ…Ù•èí±…Ñ•ÍÑ}Á…Ñ¡ôˆ¤4(€€€€€€€€€€€€€€€€•±Í”è±½¥¹œ¹•ÉÉ½È ‰…¹¹½ĞÍ…Ù”™¥¹…°¡•­Á½¥¹Ğè5½‘•°½=ÁÑ¥µ¥é•È½M…±•Èµ¥ÍÍ¥¹œ¸ˆ¤4(€€€€€€€€€€€•á•ÁĞá•ÁÑ¥½¸…Ì”è±½¥¹œ¹•ÉÉ½È¡˜‰¥¹…°¡•­Á½¥¹ĞÍ…Ù”™…¥±•èí•ôˆ°•á}¥¹™¼õQÉÕ”¤4(4(€€€€€€€¥˜¥Í}µ…¥¹}ÁÉ½•ÍÌ ¤è4(€€€€€€€€€€€ÍÑ…ÑÕÌ€ô€‰ÍÕ•ÍÍ™Õ±±äˆ¥˜ÑÉ…¥¹¥¹}ÍÕ•ÍÍ™Õ°…¹¹½ĞÍ¡ÕÑ‘½İ¹}É•ÅÕ•ÍÑ••±Í”€‰É…•™Õ±±äˆ¥˜Í¡ÕÑ‘½İ¹}É•ÅÕ•ÍÑ••±Í”€‰İ¥Ñ •ÉÉ½ÉÌˆ4(€€€€€€€€€€€±½¥¹œ¹¥¹™¼¡˜‰QÉ…¥¹¥¹œ™¥¹¥Í¡•íÍÑ…ÑÕÍô…ĞMÑ•Àí™¥¹…±}±½‰…±}ÍÑ•Áô¸ˆ¤4(€€€€€€€±•…¹ÕÁ}‘‘À ¤ì4(€€€€€€€¥˜¥Í}µ…¥¹}ÁÉ½•ÍÌ ¤è±½¥¹œ¹¥¹™¼ ‰MÉ¥ÁĞ™¥¹¥Í¡•¸ˆ¤4(4(4(Œ€´´´½¹™¥œ!•±Á•ÈÕ¹Ñ¥½¹Ì€´´´4)‘•˜½¹™¥}Ñ½}‘¥Ğ¡Í¹Ì¤è4(€€€¥˜¥Í¥¹ÍÑ…¹”¡Í¹Ì°M¥µÁ±•9…µ•ÍÁ…”¤èÉ•ÑÕÉ¸í¬è½¹™¥}Ñ½}‘¥Ğ¡Ø¤™½È¬°Ø¥¸Í¹Ì¹}}‘¥Ñ}|¹¥Ñ•µÌ ¥ô4(€€€•±¥˜¥Í¥¹ÍÑ…¹”¡Í¹Ì°€¡±¥ÍĞ°ÑÕÁ±”¤¤èÉ•ÑÕÉ¸ÑåÁ”¡Í¹Ì¤¡½¹™¥}Ñ½}‘¥Ğ¡¥Ñ•´¤™½È¥Ñ•´¥¸Í¹Ì¤4(€€€É•ÑÕÉ¸Í¹Ì4)‘•˜‘¥Ñ}Ñ½}Í¹Ì¡¤è4(€€€¥˜¥Í¥¹ÍÑ…¹”¡°‘¥Ğ¤è4(€€€€€€€Í…™•}€ôíô4(€€€€€€€™½È­•ä°Ù…±Õ”¥¸¹¥Ñ•µÌ ¤è4(€€€€€€€€€€€Í…™•}­•ä€ôÍÑÈ¡­•ä¤¹É•Á±…” œ´œ°€|œ¤4(€€€€€€€€€€€¥˜¹½ĞÍ…™•}­•ä¹¥Í¥‘•¹Ñ¥™¥•È ¤è±½¥¹œ¹İ…É¹¥¹œ¡˜‰½¹™¥œ­•ä€í­•åôœ´øíÍ…™•}­•åôœ¥¹Ù…±¥¥‘•¹Ñ¥™¥•È¸ˆ¤4(€€€€€€€€€€€Í…™•}‘mÍ…™•}­•åt€ô‘¥Ñ}Ñ½}Í¹Ì¡Ù…±Õ”¤4(€€€€€€€É•ÑÕÉ¸M¥µÁ±•9…µ•ÍÁ…” ¨©Í…™•}¤4(€€€•±¥˜¥Í¥¹ÍÑ…¹”¡°€¡±¥ÍĞ°ÑÕÁ±”¤¤èÉ•ÑÕÉ¸ÑåÁ”¡¤¡‘¥Ñ}Ñ½}Í¹Ì¡¥Ñ•´¤™½È¥Ñ•´¥¸¤4(€€€É•ÑÕÉ¸4)‘•˜µ•É•}½¹™¥Ì¡‰…Í”°ÕÍ•È¤è4(€€€µ•É•€ô½Áä¹‘••Á½Áä¡‰…Í”¤4(€€€™½È­•ä°Ù…±Õ”¥¸ÕÍ•È¹¥Ñ•µÌ ¤è4(€€€€€€€¥˜­•ä¥¸µ•É•…¹¥Í¥¹ÍÑ…¹”¡µ•É•‘m­•åt°‘¥Ğ¤…¹¥Í¥¹ÍÑ…¹”¡Ù…±Õ”°‘¥Ğ¤èµ•É•‘m­•åt€ôµ•É•}½¹™¥Ì¡µ•É•‘m­•åt°Ù…±Õ”¤4(€€€€€€€•±Í”èµ•É•‘m­•åt€ôÙ…±Õ”4(€€€É•ÑÕÉ¸µ•É•4(4(Œ€´´´¡•­Á½¥¹ĞAÉÕ¹¥¹œ!•±Á•È€´´´4(ŒI•µ½Ù”½±¡•­Á½¥¹Ğ™¥±•Ì‘ÕÉ¥¹œ±•…¹ÕÀ¸4)‘•˜ÁÉÕ¹•}¡•­Á½¥¹ÑÌ¡½ÕÑÁÕÑ}‘¥È°­••Á}±…ÍĞ¤è4(€€€¥˜­••Á}±…ÍĞ€ğ€Àè±½¥¹œ¹‘•‰Õœ¡˜‰AÉÕ¹¥¹œ‘¥Í…‰±•€¡­••Àõí­••Á}±…ÍÑô¤¸ˆ¤ìÉ•ÑÕÉ¸4(€€€¥˜­••Á}±…ÍĞ€ôô€Àè±½¥¹œ¹¥¹™¼ ‰AÉÕ¹¥¹œ…±°•Á½ ¡•­Á½¥¹ÑÌ€¡­••ÀôÀ¤¸ˆ¤€Œ-••À%9<™½ÈÑ¡¥Ì…Í”4(€€€ÑÉäè4(€€€€€€€­ÁÑ}™¥±•Í}¥¹™¼€ômtìÁ…ÑÑ•É¸€ô½Ì¹Á…Ñ ¹©½¥¸¡½ÕÑÁÕÑ}‘¥È°€ÑÕ¹•Ñ}•Á½¡|¨¹ÁÑ œ¤ì•Á½¡}™¥±•Ì€ô±½ˆ¹±½ˆ¡Á…ÑÑ•É¸¤4(€€€€€€€™½È™}Á…Ñ ¥¸•Á½¡}™¥±•Ìè4(€€€€€€€€€€€‰…Í•¹…µ”€ô½Ì¹Á…Ñ ¹‰…Í•¹…µ”¡™}Á…Ñ ¤ìµ…Ñ €ôÉ”¹µ…Ñ ¡È‰ÑÕ¹•Ñ}•Á½¡|¡q¬¥p¹ÁÑ ˆ°‰…Í•¹…µ”¤4(€€€€€€€€€€€¥˜µ…Ñ è4(€€€€€€€€€€€€€€€•Á½¡}¹Õ´€ô¥¹Ğ¡µ…Ñ ¹É½ÕÀ Ä¤¤4(€€€€€€€€€€€€€€€ÑÉäèµÑ¥µ”€ô½Ì¹Á…Ñ ¹•ÑµÑ¥µ”¡™}Á…Ñ ¤ì­ÁÑ}™¥±•Í}¥¹™¼¹…ÁÁ•¹¡ìÁ…Ñ œè™}Á…Ñ °€•Á½ œè•Á½¡}¹Õ´°€µÑ¥µ”œèµÑ¥µ•ô¤4(€€€€€€€€€€€€€€€•á•ÁĞ=MÉÉ½È…Ì”è±½¥¹œ¹İ…É¹¥¹œ¡˜‰AÉÕ¹¥¹œèÍ­¥Àí‰…Í•¹…µ•ô°…¹¹½Ğ•ĞµÑ¥µ”èí•ôˆ¤€Œ-••À]…É¹¥¹œ4(€€€€€€€­ÁÑ}™¥±•Í}¥¹™¼¹Í½ÉĞ¡­•äõ±…µ‰‘„àè€¡ál•Á½ t°álµÑ¥µ”t¤°É•Ù•ÉÍ”õQÉÕ”¤4(€€€€€€€¥˜±•¸¡­ÁÑ}™¥±•Í}¥¹™¼¤€ğô­••Á}±…ÍĞè±½¥¹œ¹‘•‰Õœ¡˜‰AÉÕ¹¥¹œè½Õ¹í±•¸¡­ÁÑ}™¥±•Í}¥¹™¼¥ô°­••Á¥¹œí­••Á}±…ÍÑô¸9½¹”É•µ½Ù•¸ˆ¤ìÉ•ÑÕÉ¸4(€€€€€€€™¥±•Í}Ñ½}É•µ½Ù”€ô­ÁÑ}™¥±•Í}¥¹™½m­••Á}±…ÍĞétìÉ•µ½Ù•‘}½Õ¹Ğ€ô€À4(€€€€€€€€Œ-••À%9<™½ÈÁÉÕ¹¥¹œÍÑ…ÉĞµ•ÍÍ…”4(€€€€€€€±½¥¹œ¹¥¹™¼¡˜‰AÉÕ¹¥¹œè-••Á¥¹œ±…ÍĞí­••Á}±…ÍÑô¡•­Á½¥¹ÑÌ¸I•µ½Ù¥¹œí±•¸¡™¥±•Í}Ñ½}É•µ½Ù”¥ô½˜í±•¸¡­ÁÑ}™¥±•Í}¥¹™¼¥ôÑ½Ñ…°¸ˆ¤4(€€€€€€€™½È­ÁÑ}¥¹™¼¥¸™¥±•Í}Ñ½}É•µ½Ù”è4(€€€€€€€€€€€ÑÉäè4(€€€€€€€€€€€€€€€½Ì¹É•µ½Ù”¡­ÁÑ}¥¹™½lÁ…Ñ t¤4(€€€€€€€€€€€€€€€€ŒUÍ”	U™½È¥¹‘¥Ù¥‘Õ…°™¥±”É•µ½Ù…°±½Ì4(€€€€€€€€€€€€€€€±½¥¹œ¹‘•‰Õœ¡˜ˆ€I•µ½Ù•èí½Ì¹Á…Ñ ¹‰…Í•¹…µ”¡­ÁÑ}¥¹™½lÁ…Ñ t¥ôˆ¤4(€€€€€€€€€€€€€€€É•µ½Ù•‘}½Õ¹Ğ€¬ô€Ä4(€€€€€€€€€€€•á•ÁĞá•ÁÑ¥½¸…Ì”è±½¥¹œ¹İ…É¹¥¹œ¡˜ˆ€…¥±•É•µ½Ù”í½Ì¹Á…Ñ ¹‰…Í•¹…µ”¡­ÁÑ}¥¹™½lÁ…Ñ t¥ôèí•ôˆ¤€Œ-••À]…É¹¥¹œ4(€€€€€€€€Œ-••À%9<™½ÈÁÉÕ¹¥¹œÍÕµµ…Éä4(€€€€€€€¥˜É•µ½Ù•‘}½Õ¹Ğ€ø€Àè±½¥¹œ¹¥¹™¼¡˜‰AÉÕ¹¥¹œ™¥¹¥Í¡•¸I•µ½Ù•íÉ•µ½Ù•‘}½Õ¹Ñô¡•­Á½¥¹Ğ¡Ì¤¸ˆ¤4(€€€•á•ÁĞá•ÁÑ¥½¸…Ì”è±½¥¹œ¹•ÉÉ½È¡˜‰¡•­Á½¥¹ĞÁÉÕ¹¥¹œ•ÉÉ½È¥¸€í½ÕÑÁÕÑ}‘¥Éôœèí•ôˆ°•á}¥¹™¼õQÉÕ”¤€Œ-••ÀÉÉ½È4(4(4(Œ€´´´5…¥¸á•ÕÑ¥½¸€¡¥˜}}¹…µ•}|€ôô€‰}}µ…¥¹}|ˆé€¤€´´´4)¥˜}}¹…µ•}|€ôô€‰}}µ…¥¹}|ˆè4(€€€Á…ÉÍ•È€ô…ÉÁ…ÉÍ”¹ÉÕµ•¹ÑA…ÉÍ•È¡‘•ÍÉ¥ÁÑ¥½¸ôQÉ…¥¸U9•ĞÙ¥„e50½¹™¥œ½¸]¥¹‘½İÌœ¤4(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ğ œ´µ½¹™¥œœ°ÑåÁ”õÍÑÈ°É•ÅÕ¥É•õQÉÕ”°¡•±ÀôA…Ñ Ñ¼Ñ¡”UMHe50½¹™¥ÕÉ…Ñ¥½¸™¥±”œ¤4(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ğ œ´µÑÉ…¥¹¥¹œ¹‰…Ñ¡}Í¥é”œ°ÑåÁ”õ¥¹Ğ°‘•ÍĞôÑÉ…¥¹¥¹}‰…Ñ¡}Í¥é”œ°‘•™…Õ±Ğõ9½¹”°¡•±Àô=Ù•ÉÉ¥‘”ÑÉ…¥¹¥¹œ‰…Ñ¡}Í¥é”œ¤4(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ğ œ´µÑÉ…¥¹¥¹œ¹±Èœ°ÑåÁ”õ™±½…Ğ°‘•ÍĞôÑÉ…¥¹¥¹}±Èœ°‘•™…Õ±Ğõ9½¹”°¡•±Àô=Ù•ÉÉ¥‘”ÑÉ…¥¹¥¹œ±•…É¹¥¹}É…Ñ”œ¤4(€€€Á…ÉÍ•È¹…‘‘}…ÉÕµ•¹Ğ œ´µ‘…Ñ„¹½ÕÑÁÕÑ}‘¥Èœ°ÑåÁ”õÍÑÈ°‘•ÍĞô‘…Ñ…}½ÕÑÁÕÑ}‘¥Èœ°‘•™…Õ±Ğõ9½¹”°¡•±Àô=Ù•ÉÉ¥‘”‘…Ñ„½ÕÑÁÕÑ}‘¥Èœ¤4(€€€±¥}…ÉÌ€ôÁ…ÉÍ•È¹Á…ÉÍ•}…ÉÌ ¤4(4(€€€€Œ€´´´%¹¥Ñ¥…°½¹™¥œ1½…‘¥¹œ1½¥¹œ€¡M¥µÁ±”½¹Í½±”¤€´´´4(€€€¥¹¥Ñ¥…±}±½}±•Ù•°€ô±½¥¹œ¹%9<4(€€€¥¹¥Ñ¥…±}±½}™½Éµ…Ğ€ô€œ”¡…ÍÑ¥µ”¥Ìm=9%t€”¡µ•ÍÍ…”¥Ìœ€ŒM¥µÁ±”™½Éµ…Ğ™½È½¹™¥œÁ¡…Í”4(€€€±½¥¹œ¹‰…Í¥½¹™¥œ¡±•Ù•°õ¥¹¥Ñ¥…±}±½}±•Ù•°°™½Éµ…Ğõ¥¹¥Ñ¥…±}±½}™½Éµ…Ğ°¡…¹‘±•ÉÌõm±½¥¹œ¹MÑÉ•…µ!…¹‘±•È ¥t°™½É”õQÉÕ”¤€ŒUÍ”™½É”õQÉÕ”Ñ¼½Ù•ÉÉ¥‘”Á½Ñ•¹Ñ¥…°ÁÉ•Ù¥½ÕÌ‰…Í¥½¹™¥Ì4(€€€±½¥¹œ¹¥¹™¼ ‰MÉ¥ÁĞÍÑ…ÉÑ¥¹œ¸¸¸ˆ¤4(€€€±½¥¹œ¹¥¹™¼¡˜‰UÍ•È½¹™¥œèí±¥}…ÉÌ¹½¹™¥ôˆ¤4(4(€€€€Œ€´´´1½…	…Í”½¹™¥œ€´´´4(€€€‰…Í•}½¹™¥}‘¥Ğ€ôíô4(€€€ÑÉäè4(€€€€€€€ÕÍ•É}™}…‰Ì€ô½Ì¹Á…Ñ ¹…‰ÍÁ…Ñ ¡±¥}…ÉÌ¹½¹™¥œ¤ìÕÍ•É}™}‘¥È€ô½Ì¹Á…Ñ ¹‘¥É¹…µ”¡ÕÍ•É}™}…‰Ì¤ìÍÉ¥ÁÑ}‘¥È€ô½Ì¹Á…Ñ ¹‘¥É¹…µ”¡½Ì¹Á…Ñ ¹…‰ÍÁ…Ñ ¡}}™¥±•}|¤¤4(€€€€€€€‰…Í•}½ÁÑÌ€ôm½Ì¹Á…Ñ ¹©½¥¸¡ÕÍ•É}™}‘¥È°€‰…Í”œ°€‰…Í”¹å…µ°œ¤°½Ì¹Á…Ñ ¹©½¥¸¡ÍÉ¥ÁÑ}‘¥È°€‰…Í”œ°€‰…Í”¹å…µ°œ¥t4(€€€€€€€‰…Í•}Á…Ñ €ô¹•áĞ ¡À™½ÈÀ¥¸‰…Í•}½ÁÑÌ¥˜½Ì¹Á…Ñ ¹•á¥ÍÑÌ¡À¤¤°9½¹”¤4(€€€€€€€¥˜¹½Ğ‰…Í•}Á…Ñ èÉ…¥Í”¥±•9½Ñ½Õ¹‘ÉÉ½È¡˜‰	…Í”½¹™¥œ€ ‰…Í”½‰…Í”¹å…µ°œ¤¹½Ğ™½Õ¹¥¸í‰…Í•}½ÁÑÍôˆ¤4(€€€€€€€İ¥Ñ ½Á•¸¡‰…Í•}Á…Ñ °€Èœ¤…Ì˜è‰…Í•}½¹™¥}‘¥Ğ€ôå…µ°¹Í…™•}±½…¡˜¤½Èíô4(€€€€€€€±½¥¹œ¹¥¹™¼¡˜‰1½…‘•‰…Í”½¹™¥œèí‰…Í•}Á…Ñ¡ôˆ¤€Œ-••À%9<4(€€€•á•ÁĞá•ÁÑ¥½¸…Ì”è±½¥¹œ¹•ÉÉ½È¡˜‰I%Q%0è…¥±•±½…‘¥¹œ‰…Í”½¹™¥œèí•ô¸á¥Ñ¥¹œ¸ˆ°•á}¥¹™¼õQÉÕ”¤ì•á¥Ğ Ä¤€Œ-••ÀÉÉ½È4(4(€€€€Œ€´´´1½…UÍ•È½¹™¥œ€´´´4(€€€ÕÍ•É}½¹™¥}‘¥Ğ€ôíô4(€€€ÑÉäè4(€€€€€€€ÕÍ•É}™}…‰Ì€ô½Ì¹Á…Ñ ¹…‰ÍÁ…Ñ ¡±¥}…ÉÌ¹½¹™¥œ¤4(€€€€€€€¥˜¹½Ğ½Ì¹Á…Ñ ¹•á¥ÍÑÌ¡ÕÍ•É}™}…‰Ì¤èÉ…¥Í”¥±•9½Ñ½Õ¹‘ÉÉ½È¡˜‰UÍ•È½¹™¥œ¹½Ğ™½Õ¹èíÕÍ•É}™}…‰Íôˆ¤4(€€€€€€€İ¥Ñ ½Á•¸¡ÕÍ•É}™}…‰Ì°€Èœ¤…Ì˜èÕÍ•É}½¹™¥}‘¥Ğ€ôå…µ°¹Í…™•}±½…¡˜¤½Èíô4(€€€€€€€±½¥¹œ¹¥¹™¼¡˜‰1½…‘•ÕÍ•È½¹™¥œèíÕÍ•É}™}…‰Íôˆ¤€Œ-••À%9<4(€€€•á•ÁĞá•ÁÑ¥½¸…Ì”è±½¥¹œ¹•ÉÉ½È¡˜‰I%Q%0è…¥±•±½…‘¥¹œÕÍ•È½¹™¥œèí•ô¸á¥Ñ¥¹œ¸ˆ°•á}¥¹™¼õQÉÕ”¤ì•á¥Ğ Ä¤€Œ-••ÀÉÉ½È4(4(€€€€Œ€´´´5•É”€˜=Ù•ÉÉ¥‘•Ì€´´´4(€€€ÑÉäèµ•É•‘}‘¥Ğ€ôµ•É•}½¹™¥Ì¡‰…Í•}½¹™¥}‘¥Ğ°ÕÍ•É}½¹™¥}‘¥Ğ¤4(€€€•á•ÁĞá•ÁÑ¥½¸…Ì”è±½¥¹œ¹•ÉÉ½È¡˜‰I%Q%0è½¹™¥œµ•É”™…¥±•èí•ô¸á¥Ñ¥¹œ¸ˆ°•á}¥¹™¼õQÉÕ”¤ì•á¥Ğ Ä¤€Œ-••ÀÉÉ½È4(€€€½Ù•ÉÉ¥‘•Ì€ôíô4(€€€¥˜±¥}…ÉÌ¹ÑÉ…¥¹¥¹}‰…Ñ¡}Í¥é”¥Ì¹½Ğ9½¹”è½Ù•ÉÉ¥‘•Ì¹Í•Ñ‘•™…Õ±Ğ ÑÉ…¥¹¥¹œœ°íô¥l‰…Ñ¡}Í¥é”t€ô±¥}…ÉÌ¹ÑÉ…¥¹¥¹}‰…Ñ¡}Í¥é”4(€€€¥˜±¥}…ÉÌ¹ÑÉ…¥¹¥¹}±È¥Ì¹½Ğ9½¹”è½Ù•ÉÉ¥‘•Ì¹Í•Ñ‘•™…Õ±Ğ ÑÉ…¥¹¥¹œœ°íô¥l±Èt€ô±¥}…ÉÌ¹ÑÉ…¥¹¥¹}±È4(€€€¥˜±¥}…ÉÌ¹‘…Ñ…}½ÕÑÁÕÑ}‘¥È¥Ì¹½Ğ9½¹”è½Ù•ÉÉ¥‘•Ì¹Í•Ñ‘•™…Õ±Ğ ‘…Ñ„œ°íô¥l½ÕÑÁÕÑ}‘¥Èt€ô½Ì¹Á…Ñ ¹…‰ÍÁ…Ñ ¡±¥}…ÉÌ¹‘…Ñ…}½ÕÑÁÕÑ}‘¥È¤4(€€€¥˜½Ù•ÉÉ¥‘•Ìèµ•É•‘}‘¥Ğ€ôµ•É•}½¹™¥Ì¡µ•É•‘}‘¥Ğ°½Ù•ÉÉ¥‘•Ì¤ì±½¥¹œ¹¥¹™¼¡˜‰ÁÁ±¥•1$½Ù•ÉÉ¥‘•Ìèí½Ù•ÉÉ¥‘•Íôˆ¤€Œ-••À%9<4(4(€€€€Œ€´´´½¹Ù•ÉĞÑ¼9…µ•ÍÁ…”€˜Y…±¥‘…Ñ”€´´´4(€€€ÑÉäè½¹™¥œ€ô‘¥Ñ}Ñ½}Í¹Ì¡µ•É•‘}‘¥Ğ¤4(€€€•á•ÁĞá•ÁÑ¥½¸…Ì”è±½¥¹œ¹•ÉÉ½È¡˜‰I%Q%0è½¹™¥œ½¹Ù•ÉÍ¥½¸™…¥±•èí•ô¸á¥Ñ¥¹œ¸ˆ°•á}¥¹™¼õQÉÕ”¤ì•á¥Ğ Ä¤€Œ-••ÀÉÉ½È4(4(€€€€Œ€´´´½¹™¥ÕÉ…Ñ¥½¸Y…±¥‘…Ñ¥½¸€¡-••À½¹¥Í”°•ÉÉ½ÉÌ…É”É¥Ñ¥…°¤€´´´4(€€€µ¥ÍÍ¥¹œ°•ÉÉ½É}µÍÌ€ômt°mt4(€€€É•Å}­•åÌ€ôì‘…Ñ„œèlÍÉ}‘¥Èœ°€‘ÍÑ}‘¥Èœ°€½ÕÑÁÕÑ}‘¥Èœ°€É•Í½±ÕÑ¥½¸œ°€½Ù•É±…Á}™…Ñ½Èt°€µ½‘•°œèlµ½‘•±}Í¥é•}‘¥µÌt°€ÑÉ…¥¹¥¹œœèl¥Ñ•É…Ñ¥½¹Í}Á•É}•Á½ œ°€‰…Ñ¡}Í¥é”œ°€±Èœ°€±½ÍÌœ°€±…µ‰‘…}±Á¥ÁÌœ°€ÕÍ•}…µÀt°€Í…Ù¥¹œœèl­••Á}±…ÍÑ}¡•­Á½¥¹ÑÌuô4(€€€€Œ¡•¬É•ÅÕ¥É•Í•Ñ¥½¹Ì…¹­•åÌ4(€€€™½ÈÍ•Œ°­•åÌ¥¸É•Å}­•åÌ¹¥Ñ•µÌ ¤è4(€€€€€€€Í•}½‰¨€ô•Ñ…ÑÑÈ¡½¹™¥œ°Í•Œ°9½¹”¤4(€€€€€€€¥˜Í•}½‰¨¥Ì9½¹”è4(€€€€€€€€€€€•ÉÉ½É}µÍÌ¹…ÁÁ•¹¡˜‰5¥ÍÍ¥¹œÉ•ÅÕ¥É•Í•Ñ¥½¸è€íÍ•ôœˆ¤4(€€€€€€€€€€€µ¥ÍÍ¥¹œ¹…ÁÁ•¹¡Í•Œ¤4(€€€€€€€€€€€½¹Ñ¥¹Õ”€ŒM­¥À­•åÌ¡•¬¥˜Í•Ñ¥½¸¥Ìµ¥ÍÍ¥¹œ4(€€€€€€€€Œ%˜Í•Ñ¥½¸•á¥ÍÑÌ°¡•¬¥ÑÌ­•åÌ4(€€€€€€€™½È¬¥¸­•åÌè4(€€€€€€€€€€€¥˜•Ñ…ÑÑÈ¡Í•}½‰¨°¬°9½¹”¤¥Ì9½¹”è4(€€€€€€€€€€€€€€€•ÉÉ½É}µÍÌ¹…ÁÁ•¹¡˜‰5¥ÍÍ¥¹œ½¹Õ±°É•ÅÕ¥É•Ù…±Õ”è€íÍ•ô¹í­ôœˆ¤4(€€€€€€€€€€€€€€€µ¥ÍÍ¥¹œ¹…ÁÁ•¹¡˜‰íÍ•ô¹í­ôˆ¤4(4(4(€€€€Œ•™…Õ±Ñ¥¹œ€¡ÕÍ”Í¥µÁ±”•Ñ…ÑÑÈİ¥Ñ ‘•™…Õ±Ğ°…Ù½¥•áÑÉ„±½Ì¡•É”¤4(€€€½¹™¥œ¹‘…Ñ…±½…‘•È€ô•Ñ…ÑÑÈ¡½¹™¥œ°€‘…Ñ…±½…‘•Èœ°M¥µÁ±•9…µ•ÍÁ…” ¤¤4(€€€½¹™¥œ¹‘…Ñ…±½…‘•È¹¹Õµ}İ½É­•ÉÌ€ô•Ñ…ÑÑÈ¡½¹™¥œ¹‘…Ñ…±½…‘•È°€¹Õµ}İ½É­•ÉÌœ°€À¤4(€€€½¹™¥œ¹‘…Ñ…±½…‘•È¹ÁÉ•™•Ñ¡}™…Ñ½È€ô•Ñ…ÑÑÈ¡½¹™¥œ¹‘…Ñ…±½…‘•È°€ÁÉ•™•Ñ¡}™…Ñ½Èœ°€È¥˜½¹™¥œ¹‘…Ñ…±½…‘•È¹¹Õµ}İ½É­•ÉÌ€ø€À•±Í”9½¹”¤4(€€€½¹™¥œ¹Í…Ù¥¹œ€ô•Ñ…ÑÑÈ¡½¹™¥œ°€Í…Ù¥¹œœ°M¥µÁ±•9…µ•ÍÁ…” ¤¤ì½¹™¥œ¹Í…Ù¥¹œ¹Í…Ù•}¥Ñ•É…Ñ¥½¹Í}¥¹Ñ•ÉÙ…°€ô•Ñ…ÑÑÈ¡½¹™¥œ¹Í…Ù¥¹œ°€Í…Ù•}¥Ñ•É…Ñ¥½¹Í}¥¹Ñ•ÉÙ…°œ°€À¤4(€€€½¹™¥œ¹±½¥¹œ€ô•Ñ…ÑÑÈ¡½¹™¥œ°€±½¥¹œœ°M¥µÁ±•9…µ•ÍÁ…” ¤¤ì½¹™¥œ¹±½¥¹œ¹±½}¥¹Ñ•ÉÙ…°€ô•Ñ…ÑÑÈ¡½¹™¥œ¹±½¥¹œ°€±½}¥¹Ñ•ÉÙ…°œ°€ÔÀ¤4(€€€½¹™¥œ¹±½¥¹œ¹ÁÉ•Ù¥•İ}‰…Ñ¡}¥¹Ñ•ÉÙ…°€ô•Ñ…ÑÑÈ¡½¹™¥œ¹±½¥¹œ°€ÁÉ•Ù¥•İ}‰…Ñ¡}¥¹Ñ•ÉÙ…°œ°€ÔÀÀ¤ì½¹™¥œ¹±½¥¹œ¹ÁÉ•Ù¥•İ}É•™É•Í¡}É…Ñ”€ô•Ñ…ÑÑÈ¡½¹™¥œ¹±½¥¹œ°€ÁÉ•Ù¥•İ}É•™É•Í¡}É…Ñ”œ°€Ô¤4(€€€€Œ¥¹…°Y…±Õ”¡•­Ì4(€€€¥˜¹½Ğµ¥ÍÍ¥¹œè4(€€€€€€€ÑÉäè€Œ½µ‰¥¹”¡•­Ì™½È‰É•Ù¥Ñä4(€€€€€€€€€€€¥˜½¹™¥œ¹ÑÉ…¥¹¥¹œ¹¥Ñ•É…Ñ¥½¹Í}Á•É}•Á½ €ğô€Àè•ÉÉ½É}µÍÌ¹…ÁÁ•¹ ‰ÑÉ…¥¸¹¥Ñ•É}•Á½ ğôÀˆ¤4(€€€€€€€€€€€¥˜½¹™¥œ¹ÑÉ…¥¹¥¹œ¹‰…Ñ¡}Í¥é”€ğô€Àè•ÉÉ½É}µÍÌ¹…ÁÁ•¹ ‰ÑÉ…¥¸¹‰…Ñ¡}Í¥é”ğôÀˆ¤4(€€€€€€€€€€€¥˜¹½Ğ¥Í¥¹ÍÑ…¹”¡½¹™¥œ¹ÑÉ…¥¹¥¹œ¹±È°€¡¥¹Ğ°™±½…Ğ¤¤½È½¹™¥œ¹ÑÉ…¥¹¥¹œ¹±È€ğô€Àè•ÉÉ½É}µÍÌ¹…ÁÁ•¹ ‰ÑÉ…¥¸¹±ÈğôÀˆ¤4(€€€€€€€€€€€¥˜½¹™¥œ¹ÑÉ…¥¹¥¹œ¹±½ÍÌ¹½Ğ¥¸l°Äœ°€°Ä­±Á¥ÁÌtè•ÉÉ½É}µÍÌ¹…ÁÁ•¹ ‰ÑÉ…¥¸¹±½ÍÌ¥¹Ù…±¥ˆ¤4(€€€€€€€€€€€¥˜½¹™¥œ¹ÑÉ…¥¹¥¹œ¹±…µ‰‘…}±Á¥ÁÌ€ğ€Àè•ÉÉ½É}µÍÌ¹…ÁÁ•¹ ‰ÑÉ…¥¸¹±…µ‰‘…}±Á¥ÁÌğÀˆ¤4(€€€€€€€€€€€¥˜¹½Ğ€ À¸À€ğô½¹™¥œ¹‘…Ñ„¹½Ù•É±…Á}™…Ñ½È€ğ€Ä¸À¤è•ÉÉ½É}µÍÌ¹…ÁÁ•¹ ‰‘…Ñ„¹½Ù•É±…À¥¹Ù…±¥ˆ¤4(€€€€€€€€€€€‘ÌôÄØì4(€€€€€€€€€€€¥˜½¹™¥œ¹‘…Ñ„¹É•Í½±ÕÑ¥½¸ğôÀ½È½¹™¥œ¹‘…Ñ„¹É•Í½±ÕÑ¥½¸€”‘Ì€„ôÀè•ÉÉ½É}µÍÌ¹…ÁÁ•¹¡˜‰‘…Ñ„¹É•Í½±ÕÑ¥½¸¹½Ğ€øÀ€˜‘¥Ø‰äí‘Íôˆ¤4(€€€€€€€€€€€¥˜¹½Ğ½Ì¹Á…Ñ ¹¥Í‘¥È¡½¹™¥œ¹‘…Ñ„¹ÍÉ}‘¥È¤è•ÉÉ½É}µÍÌ¹…ÁÁ•¹¡˜‰‘…Ñ„¹ÍÉ}‘¥Èµ¥ÍÍ¥¹œèí½¹™¥œ¹‘…Ñ„¹ÍÉ}‘¥Éôˆ¤€Œ‘‘•Á…Ñ Ñ¼µÍœ4(€€€€€€€€€€€¥˜¹½Ğ½Ì¹Á…Ñ ¹¥Í‘¥È¡½¹™¥œ¹‘…Ñ„¹‘ÍÑ}‘¥È¤è•ÉÉ½É}µÍÌ¹…ÁÁ•¹¡˜‰‘…Ñ„¹‘ÍÑ}‘¥Èµ¥ÍÍ¥¹œèí½¹™¥œ¹‘…Ñ„¹‘ÍÑ}‘¥Éôˆ¤€Œ‘‘•Á…Ñ Ñ¼µÍœ4(€€€€€€€€€€€¥˜½¹™¥œ¹µ½‘•°¹µ½‘•±}Í¥é•}‘¥µÌ€ğô€Àè•ÉÉ½É}µÍÌ¹…ÁÁ•¹ ‰µ½‘•°¹Í¥é”ğôÀˆ¤4(€€€€€€€€€€€¥˜½¹™¥œ¹‘…Ñ…±½…‘•È¹¹Õµ}İ½É­•ÉÌ€ğ€Àè•ÉÉ½É}µÍÌ¹…ÁÁ•¹ ‰±½…‘•È¹İ½É­•ÉÌğÀˆ¤4(€€€€€€€€€€€¥˜½¹™¥œ¹‘…Ñ…±½…‘•È¹¹Õµ}İ½É­•ÉÌøÀ…¹½¹™¥œ¹‘…Ñ…±½…‘•È¹ÁÉ•™•Ñ¡}™…Ñ½È¥Ì¹½Ğ9½¹”…¹½¹™¥œ¹‘…Ñ…±½…‘•È¹ÁÉ•™•Ñ¡}™…Ñ½ÈğÈè•ÉÉ½É}µÍÌ¹…ÁÁ•¹ ‰±½…‘•È¹ÁÉ•™•Ñ ğÈ¥¹Ù…±¥ˆ¤4(€€€€€€€€€€€¥˜½¹™¥œ¹Í…Ù¥¹œ¹Í…Ù•}¥Ñ•É…Ñ¥½¹Í}¥¹Ñ•ÉÙ…°€ğ€Àè•ÉÉ½É}µÍÌ¹…ÁÁ•¹ ‰Í…Ù”¹¥¹Ñ•ÉÙ…°ğÀˆ¤4(€€€€€€€€€€€¥˜½¹™¥œ¹Í…Ù¥¹œ¹­••Á}±…ÍÑ}¡•­Á½¥¹ÑÌ€ğ€´Äè•ÉÉ½É}µÍÌ¹…ÁÁ•¹ ‰Í…Ù”¹­••Àğ´Äˆ¤4(€€€€€€€€€€€¥˜½¹™¥œ¹±½¥¹œ¹±½}¥¹Ñ•ÉÙ…°€ğô€Àè•ÉÉ½É}µÍÌ¹…ÁÁ•¹ ‰±½œ¹¥¹Ñ•ÉÙ…°ğôÀˆ¤4(€€€€€€€€€€€¥˜½¹™¥œ¹±½¥¹œ¹ÁÉ•Ù¥•İ}‰…Ñ¡}¥¹Ñ•ÉÙ…°€ğ€Àè•ÉÉ½É}µÍÌ¹…ÁÁ•¹ ‰±½œ¹ÁÉ•Ù¥•ÜğÀˆ¤4(€€€€€€€€€€€¥˜½¹™¥œ¹±½¥¹œ¹ÁÉ•Ù¥•İ}É•™É•Í¡}É…Ñ”€ğ€Àè•ÉÉ½É}µÍÌ¹…ÁÁ•¹ ‰±½œ¹É•™É•Í ğÀˆ¤4(€€€€€€€•á•ÁĞá•ÁÑ¥½¸…Ì”è•ÉÉ½É}µÍÌ¹…ÁÁ•¹¡˜‰Y…±¥‘…Ñ¥½¸•ÉÉ½Èèí•ôˆ¤4(€€€€Œ‰½ÉĞ½¸ÉÉ½ÉÌ4(€€€¥˜•ÉÉ½É}µÍÌ½Èµ¥ÍÍ¥¹œè4(€€€€€€€€ÁÉ¥¹Ğ ‰q¸ˆ€¬€ˆôˆ¨ÌÀ€¬€ˆ=9%II=IL€ˆ€¬€ˆôˆ¨ÌÀ¤ì…±±}•ÉÉÌ€ôÍ½ÉÑ•¡±¥ÍĞ¡Í•Ğ¡•ÉÉ½É}µÍÌ€¬m˜‰5¥ÍÍ¥¹œèíµôˆ™½È´¥¸µ¥ÍÍ¥¹t¤¤¤4(€€€€€€€€™½È¤°µÍœ¥¸•¹Õµ•É…Ñ”¡…±±}•ÉÉÌ¤èÁÉ¥¹Ğ¡˜ˆ€í¤¬Åô¸íµÍôˆ¤4(€€€€€€€€ÁÉ¥¹Ğ ˆôˆ¨àÈ€¬€‰q¹A±•…Í”™¥à½¹™¥œ…¹É•ÍÑ…ÉĞ¸ˆ¤ì•á¥Ğ Ä¤4(€€€•±Í”è±½¥¹œ¹¥¹™¼ ‰m=9%t½¹™¥ÕÉ…Ñ¥½¸Ù…±¥‘…Ñ•¸ˆ¤€Œ-••ÀÍ¥µÁ±”%9<½¹™¥Éµ…Ñ¥½¸4(4(€€€€Œ@¹ØY…È¡•¬€¡-••À…Ì]…É¹¥¹œ¤4(€€€¥˜½Ì¹•¹Ù¥É½¸¹•Ğ ‰]=I1}M%iˆ°€ˆÄˆ¤€„ô€ˆÄˆ…¹½Ì¹•¹Ù¥É½¸¹•Ğ ‰1=1}I9,ˆ°9½¹”¤¥Ì9½¹”è4(€€€€€€€±½¥¹œ¹İ…É¹¥¹œ ‰@ÉÕ¸‘•Ñ•Ñ•€¡]=I1}M%iøÄ¤‰ÕĞ1=1}I9,¹½ĞÍ•Ğ¸UÍ”Ñ¡”ÍÕÁÁ½ÉÑ•‘¥ÍÑÉ¥‰ÕÑ•±…Õ¹¡•È½ÈÍ•ĞÑ¡”É…¹¬Ù…É¥…‰±•Ì¸ˆ¤€Œ-••À]…É¹¥¹œ4(4(€€€€Œ€´´´MÑ…ÉĞQÉ…¥¹¥¹œ€´´´4(€€€ÑÉäèÑÉ…¥¸¡½¹™¥œ¤4(€€€•á•ÁĞá•ÁÑ¥½¸…Ìµ…¥¹}•ÉÈè±½¥¹œ¹•ÉÉ½È ‰U¹¡…¹‘±••ÉÉ½È½ÕÑÍ¥‘”ÑÉ…¥¸±½½Àèˆ°•á}¥¹™¼õQÉÕ”¤ì±•…¹ÕÁ}‘‘À ¤ì•á¥Ğ Ä¤€ŒM…¥Ô€4