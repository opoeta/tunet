# Cross-Platform support

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
import platform # OS detection

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
CURRENT_OS = platform.system()

# --- DDP Setup ---
def setup_ddp():
    """Initializes the DDP process group if needed."""
    if not dist.is_available():
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        # Use print for essential info if logging isn't fully configured yet
        print(f"INFO: PyTorch distributed not available. Running in single-process mode on {CURRENT_OS}.")
        return

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")

    if world_size == 1:
        # Only log basic info for single process
        # Use logging.debug for details that shouldn't clutter console
        logging.info(f"Running in single-process mode (World Size = 1).")

        # --- Device Setting for Single Process ---
        # teste MPS (mac) 
        if CURRENT_OS == 'Darwin' and torch.backends.mps.is_available() and torch.backends.mps.is_built():
             # MPS doesn't use local_rank like CUDA for device selection
             logging.debug(f"Single process mode: MPS available.")
        elif torch.cuda.is_available():
            try:
                torch.cuda.set_device(local_rank)
                logging.debug(f"Single process mode: CUDA device set to cuda:{local_rank}")
            except Exception as e:
                 # Use ERROR level for actual problems
                logging.error(f"Error setting CUDA device cuda:{local_rank} in single process mode: {e}. Check CUDA visibility/drivers.")
        else:
             logging.debug(f"Single process mode: No CUDA or MPS available.")
        # --- End Device Setting ---
        return # DDP init not needed

    # --- Initialize DDP for world_size > 1 ---
    if not dist.is_initialized():
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port

        # Backend Selection: Prioritize NCCL if CUDA available, else use Gloo (for CPU/MPS DDP)
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        if CURRENT_OS == 'Darwin' and not torch.cuda.is_available():
            # Force Gloo on Mac if no CUDA (necessary for MPS DDP)
            backend = 'gloo'
            logging.debug(f"Forcing Gloo backend for DDP on macOS (No CUDA detected).")

        try:
            init_method = f'tcp://{master_addr}:{master_port}'
            dist.init_process_group(backend=backend, init_method=init_method, world_size=world_size, rank=rank)

            # Determine device info for logging (actual device set in train())
            if CURRENT_OS == 'Darwin' and torch.backends.mps.is_available() and torch.backends.mps.is_built():
                 device_info = "mps"
            elif torch.cuda.is_available():
                 # local_rank is relevant for CUDA device selection
                 device_info = f"cuda:{local_rank}"
            else:
                 device_info = "cpu"

            if rank == 0: # Log DDP success only on rank 0
                logging.info(f"DDP Initialized: Rank {rank}/{world_size} using {device_info} target (Backend: {backend})")

        except RuntimeError as e:
            # Attempt Gloo fallback if NCCL fails (especially on non-Linux)
            if 'nccl' in str(e).lower() and backend == 'nccl':
                logging.warning(f"NCCL backend failed (OS: {CURRENT_OS}). Trying Gloo...") # Keep Warning
                backend = 'gloo'
                try:
                    dist.init_process_group(backend=backend, init_method=init_method, world_size=world_size, rank=rank)
                    # Determine device info again after fallback
                    if CURRENT_OS == 'Darwin' and torch.backends.mps.is_available() and torch.backends.mps.is_built(): device_info = "mps"
                    elif torch.cuda.is_available(): device_info = f"cuda:{local_rank}"
                    else: device_info = "cpu"
                    if rank == 0: logging.info(f"DDP Re-Initialized using Gloo backend: Rank {rank}/{world_size} on {device_info}") # Keep INFO
                except Exception as e_gloo:
                    logging.error(f"Gloo backend also failed on Rank {rank}: {e_gloo}", exc_info=True) # Keep Error
                    raise RuntimeError(f"DDP Initialization failed on Rank {rank} with both NCCL and Gloo.") from e_gloo
            else:
                 logging.error(f"DDP initialization failed on Rank {rank} (Backend: {backend}): {e}", exc_info=True) # Keep Error
                 raise RuntimeError(f"DDP Initialization failed on Rank {rank}.") from e
        except Exception as e:
            logging.error(f"Unexpected error during DDP initialization on Rank {rank}: {e}", exc_info=True) # Keep Error
            raise RuntimeError(f"Unexpected DDP Initialization error on Rank {rank}.") from e


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
            logging.info(f"×MyâÚ$z{-®éÜj×6SÖFWf–6RĞ¢F—7BæÆÅ÷&VGV6R†Ã÷BÂ÷ÖF—7Bå&VGV6T÷ädr“²F—7BæÆÅ÷&VGV6R†Ç÷BÂ÷ÖF—7Bå&VGV6T÷ädrĞ¢&F6…öÃÒÃ÷Bæ—FVÒ‚“²&F6…öÇÒÇ÷Bæ—FVÒ‚Ğ¢WöÃ³Ò&F6…öÃ²WöÇ—2³Ò&F6…öÇ²W÷7FW2³ÒĞ Ğ¢—FW%÷F–ÖRÒF–ÖRçF–ÖR‚’ÒÆö÷ö—FW%÷7F'C²&F6…÷F–ÖW2æVæB†—FW%÷F–ÖR“²&F6…÷F–ÖW2Ò&F6…÷F–ÖW5²Ó¥ĞĞ¢fu÷F–ÖRÒ7VÒ†&F6…÷F–ÖW2’òÆVâ†&F6…÷F–ÖW2’–b&F6…÷F–ÖW2VÇ6Rã Ğ Ğ¢vÆö&Å÷7FW³Ò²f–æÅövÆö&Å÷7FWÒvÆö&Å÷7FW Ğ Ğ¢–b—5öÖ–å÷&ö6W72‚’æBvÆö&Å÷7FWR6öæf–ræÆövv–æræÆöuö–çFW'fÂÓÒ Ğ¢7W'&VçEöÇ"Ò÷F–Ö—¦W"ç&Õöw&÷W5³Õ²vÇ"uĞĞ¢Æövv–æræ–æfò†b~(
b„C§¶FFöÆöE÷F–ÖS¢ã6gÒC§·G&ç6fW%÷F–ÖS¢ã6gÒ3§¶6ö×WFU÷F–ÖS¢ã6gÒ’rĞ¢fuöWöÃÒWöÃòW÷7FW2–bW÷7FW2âVÇ6Rã Ğ¢fuöWöÇÒWöÇ—2òW÷7FW2–bW6UöÇ—2æBW÷7FW2âVÇ6Rã Ğ¢7FW5ö–åöWÒvÆö&Å÷7FWR—FW%öWö6‚÷"—FW%öWö6€Ğ¢Æöuö×6rÒ†btWö6…·¶7W'&VçEöWö–G‚²ÕÒ7FW·¶vÆö&Å÷7FWÕÒ‡·7FW5ö–åöWÒ÷¶—FW%öWö6‡Ò’ÂpĞ¢btÃ§¶&F6…öÃ¢ãFgÒ„fs§¶fuöWöÃ¢ãFgÒ’rĞ¢–bW6UöÇ—3¢Æöuö×6r³Ò†brÂÅ•3§¶&F6…öÇ¢ãFgÒ„fs§¶fuöWöÇ¢ãFgÒ’rĞ¢Æöuö×6r³Ò†brÂÅ#§¶7W'&VçEöÇ#¢ãWÒÂBõ7FW§¶fu÷F–ÖS¢ã6g×2pĞ¢br„C§¶FFöÆöE÷F–ÖS¢ã6gÒC§·G&ç6fW%÷F–ÖS¢ã6gÒ3§¶6ö×WFU÷F–ÖS¢ã6gÒ’rĞ¢Æövv–æræ–æfò†Æöuö×6rĞ Ğ¢–b—5öÖ–å÷&ö6W72‚’æB&Wf–Wuö–çFW'fÂâæBvÆö&Å÷7FWR&Wf–Wuö–çFW'fÂÓÒ Ğ¢&Vg&W6‚Ò†f—†VE÷7&2—2æöæR’÷"†6öæf–ræÆövv–ærç&Wf–Wu÷&Vg&W6…÷&FRâæB&Wf–Wuö6÷VçBâæB‡&Wf–Wuö6÷VçBR6öæf–ræÆövv–ærç&Wf–Wu÷&Vg&W6…÷&FRÓÒ’Ğ¢–b&Vg&W6ƒ Ğ¢æWu÷7&2ÂæWuöG7BÒ6GW&U÷&Wf–Wuö&F6‚†6öæf–rÂ7&5÷G&ç6f÷&×2ÂG7E÷G&ç6f÷&×2Â6†&VE÷G&ç6f÷&×2Â7FæF&E÷G&ç6f÷&ÒĞ¢–bæWu÷7&2—2æ÷BæöæS¢f—†VE÷7&2Âf—†VEöG7BÒæWu÷7&2ÂæWuöG7@Ğ¢VÆ–bf—†VE÷7&2—2æöæS¢Æövv–ærçv&æ–ær†b%7¶vÆö&Å÷7FWÓ¢&Wf–Wr&Vg&W6‚f–ÆVB†æò–æ—F–Â’â"Ğ¢VÇ6S¢Æövv–ærçv&æ–ær†b%7¶vÆö&Å÷7FWÓ¢&Wf–Wr&Vg&W6‚f–ÆVBÂW6–æröÆBâ"Ğ¢–bf—†VE÷7&2—2æ÷BæöæS¢6fU÷&Wf–Ww2†ÖöFVÂÂf—†VE÷7&2Âf—†VEöG7BÂ6öæf–ræFFæ÷WGWEöF—"Â7W'&VçEöWö–G‚ÂvÆö&Å÷7FWÂFWf–6RÂ&Wf–Wuö6÷VçBÂ6öæf–ræÆövv–ærç&Wf–Wu÷&Vg&W6…÷&FR“²&Wf–Wuö6÷VçB³ÒĞ¢VÆ–b&Wf–Wuö6÷VçBÓÒ¢Æövv–ærçv&æ–ær†b%7¶vÆö&Å÷7FWÓ¢6¶—–ær&Wf–Wr†æò&F6‚’â"Ğ Ğ¢6fUö–çFW'fÂÒ6öæf–rç6f–ærç6fUö—FW&F–öç5ö–çFW'fÃ²6fUöæ÷rÒ‡6fUö–çFW'fÂâæBvÆö&Å÷7FWR6fUö–çFW'fÂÓÒĞ¢Wö6…öVæEöæ÷rÒ†vÆö&Å÷7FWR—FW%öWö6‚ÓÒ’æBvÆö&Å÷7FWâ Ğ¢–b—5öÖ–å÷&ö6W72‚’æB‡6fUöæ÷r÷"Wö6…öVæEöæ÷r“ Ğ¢6·EöWÒvÆö&Å÷7FWòò—FW%öWö6ƒ²Wö6·E÷F‚Ò÷2çF‚æ¦ö–â†6öæf–ræFFæ÷WGWEöF—"ÂbwGVæWEöWö6…÷¶6·EöW£–GÒçF‚r’–bWö6…öVæEöæ÷rVÇ6RæöæPĞ¢ÆFW7E÷F‚Ò÷2çF‚æ¦ö–â†6öæf–ræFFæ÷WGWEöF—"ÂwGVæWEöÆFW7BçF‚r“²6fuöF–7BÒ6öæf–u÷FõöF–7B†6öæf–rĞ¢Õ÷7FFRÒÖöFVÂæÖöGVÆRç7FFUöF–7B‚’–b—6–ç7Fæ6R†ÖöFVÂÂDE’VÇ6RÖöFVÂç7FFUöF–7B‚Ğ¢26fR66ÆW"7FFRöæÇ’–bÕv2VffV7F—fVÇ’W6V@Ğ¢66ÆW%÷7FFRÒ66ÆW"ç7FFUöF–7B‚’–bW6Uö×öVfbVÇ6RæöæPĞ¢6·EöFFÒ²vWö6‚s¢6·EöWÂvvÆö&Å÷7FWs¢vÆö&Å÷7FWÂvÖöFVÅ÷7FFUöF–7Bs¢Õ÷7FFRÂv÷F–Ö—¦W%÷7FFUöF–7Bs¢÷F–Ö—¦W"ç7FFUöF–7B‚’ÀĞ¢w66ÆW%÷7FFUöF–7Bs¢66ÆW%÷7FFRÂv6öæf–rs¢6fuöF–7BÂvVffV7F—fUöÖöFVÅ÷6—¦Rs¢Vfe÷6—¦WĞĞ¢G'“ Ğ¢&V6öâÒ&–çFW'fÂ"–b6fUöæ÷rVÇ6R&Wö6‚VæB Ğ¢F÷&6‚ç6fR†6·EöFFÂÆFW7E÷F‚Ğ¢Æövv–æræ–æfò†b%6fVBÆFW7B6†V6·ö–çB‡·&V6öçÒ’7FW¶vÆö&Å÷7FWÒ"Ğ¢–bWö6·E÷Fƒ Ğ¢F÷&6‚ç6fR†6·EöFFÂWö6·E÷F‚Ğ¢Æövv–æræ–æfò†b%6fVBWö6‚6†V6·ö–çC¢¶÷2çF‚æ&6VæÖR†Wö6·E÷F‚—Ò"Ğ¢–b†6GG"†6öæf–rç6f–ærÂv¶VWöÆ7Eö6†V6·ö–çG2r’æB6öæf–rç6f–æræ¶VWöÆ7Eö6†V6·ö–çG2ãÒ Ğ¢'VæUö6†V6·ö–çG2†6öæf–ræFFæ÷WGWEöF—"Â6öæf–rç6f–æræ¶VWöÆ7Eö6†V6·ö–çG2Ğ¢W†6WBW†6WF–öâ2S¢Æövv–æræW'&÷"†b$6†V6·ö–çB6fRf–ÆVB7FW¶vÆö&Å÷7FWÓ¢¶WÒ"ÂW†5ö–æfóÕG'VRĞ¢2ÒÒÒVæBöbG&–æ–ærÆö÷ÒÒĞĞ¢G&–æ–æu÷7V66W76gVÂÒG'VPĞ¢W†6WB¶W–&ö&D–çFW''WC Ğ¢–b—5öÖ–å÷&ö6W72‚“¢Æövv–ærçv&æ–ær‚$¶W–&ö&D–çFW''WBâ6‡WGF–ærF÷vââââ"Ğ¢–bæ÷B6‡WFF÷vå÷&WVW7FVC¢†æFÆU÷6–væÂ‡6–væÂå4”t”åBÂæöæRĞ¢W†6WBW†6WF–öâ2Æö÷öW'# Ğ¢Æövv–æræW'&÷"‚$dDÂG&–æ–ærÆö÷W'&÷#¢"ÂW†5ö–æfóÕG'VR“°Ğ¢6‡WFF÷vå÷&WVW7FVBÒG'VPĞ¢f–æÆÇ“ Ğ¢–b6‡WFF÷vå÷&WVW7FVBæB—5öÖ–å÷&ö6W72‚“ Ğ¢Æövv–æræ–æfò†b%W&f÷&Ö–ærf–æÂ6†V6·ö–çB6fR…7FFR7FW¶f–æÅövÆö&Å÷7FWÒ’âââ"Ğ¢G'“ Ğ¢f–æÅöWÒf–æÅövÆö&Å÷7FWòò—FW%öWö6ƒ²ÆFW7E÷F‚Ò÷2çF‚æ¦ö–â†6öæf–ræFFæ÷WGWEöF—"ÂwGVæWEöÆFW7BçF‚r“²6fuöF–7BÒ6öæf–u÷FõöF–7B†6öæf–rĞ¢–bÖöFVÂæB÷F–Ö—¦W"æB66ÆW# Ğ¢Õ÷7FFRÒÖöFVÂæÖöGVÆRç7FFUöF–7B‚’–b—6–ç7Fæ6R†ÖöFVÂÂDE’VÇ6RÖöFVÂç7FFUöF–7B‚Ğ¢66ÆW%÷7FFRÒ66ÆW"ç7FFUöF–7B‚’–bW6Uö×öVfbVÇ6RæöæR26†V6²VffV7F—fRW6PĞ¢f–æÅöFFÒ²vWö6‚s¢f–æÅöWÂvvÆö&Å÷7FWs¢f–æÅövÆö&Å÷7FWÂvÖöFVÅ÷7FFUöF–7Bs¢Õ÷7FFRÂv÷F–Ö—¦W%÷7FFUöF–7Bs¢÷F–Ö—¦W"ç7FFUöF–7B‚’ÀĞ¢w66ÆW%÷7FFUöF–7Bs¢66ÆW%÷7FFRÂv6öæf–rs¢6fuöF–7BÂvVffV7F—fUöÖöFVÅ÷6—¦Rs¢Vfe÷6—¦WĞĞ¢F÷&6‚ç6fR†f–æÅöFFÂÆFW7E÷F‚“²Æövv–æræ–æfò†b$f–æÂ6†V6·ö–çB6fVC¢¶ÆFW7E÷F‡Ò"Ğ¢VÇ6S¢Æövv–æræW'&÷"‚$6ææ÷B6fRf–æÂ6†V6·ö–çC¢ÖöFVÂô÷F–Ö—¦W"õ66ÆW"Ö—76–ærâ"Ğ¢W†6WBW†6WF–öâ2S¢Æövv–æræW'&÷"†b$f–æÂ6†V6·ö–çB6fRf–ÆVC¢¶WÒ"ÂW†5ö–æfóÕG'VRĞ Ğ¢–b—5öÖ–å÷&ö6W72‚“ Ğ¢7FGW2Ò'7V66W76gVÆÇ’"–bG&–æ–æu÷7V66W76gVÂæBæ÷B6‡WFF÷vå÷&WVW7FVBVÇ6R&w&6VgVÆÇ’"–b6‡WFF÷vå÷&WVW7FVBVÇ6R'v—F‚W'&÷'2 Ğ¢Æövv–æræ–æfò†b%G&–æ–ærf–æ—6†VB·7FGW7ÒB7FW¶f–æÅövÆö&Å÷7FWÒâ"Ğ¢6ÆVçWöFG‚“°Ğ¢–b—5öÖ–å÷&ö6W72‚“¢Æövv–æræ–æfò‚%67&—Bf–æ—6†VBâ"Ğ Ğ Ğ¢2ÒÒÒ6öæf–r†VÇW"gVæ7F–öç2ÒÒĞĞ¦FVb6öæf–u÷FõöF–7B‡6ç2“ Ğ¢–b—6–ç7Fæ6R‡6ç2Â6–×ÆTæÖW76R“¢&WGW&â¶³¢6öæf–u÷FõöF–7B‡b’f÷"²Âb–â6ç2åõöF–7Eõòæ—FV×2‚—ĞĞ¢VÆ–b—6–ç7Fæ6R‡6ç2Â†Æ—7BÂGWÆR’“¢&WGW&âG—R‡6ç2’†6öæf–u÷FõöF–7B†—FVÒ’f÷"—FVÒ–â6ç2Ğ¢&WGW&â6ç0Ğ¦FVbF–7E÷Fõ÷6ç2†B“ Ğ¢–b—6–ç7Fæ6R†BÂF–7B“ Ğ¢6fUöBÒ·ĞĞ¢f÷"¶W’ÂfÇVR–âBæ—FV×2‚“ Ğ¢6fUö¶W’Ò7G"†¶W’’ç&WÆ6R‚rÒrÂuòrĞ¢–bæ÷B6fUö¶W’æ—6–FVçF–f–W"‚“¢Æövv–ærçv&æ–ær†b$6öæf–r¶W’w¶¶W—ÒrÓâw·6fUö¶W—Òr–çfÆ–B–FVçF–f–W"â"Ğ¢6fUöE·6fUö¶W•ÒÒF–7E÷Fõ÷6ç2‡fÇVRĞ¢&WGW&â6–×ÆTæÖW76R‚¢§6fUöBĞ¢VÆ–b—6–ç7Fæ6R†BÂ†Æ—7BÂGWÆR’“¢&WGW&âG—R†B’†F–7E÷Fõ÷6ç2†—FVÒ’f÷"—FVÒ–âBĞ¢&WGW&â@Ğ¦FVbÖW&vUö6öæf–w2†&6RÂW6W"“ Ğ¢ÖW&vVBÒ6÷’æFVW6÷’†&6RĞ¢f÷"¶W’ÂfÇVR–âW6W"æ—FV×2‚“ Ğ¢–b¶W’–âÖW&vVBæB—6–ç7Fæ6R†ÖW&vVE¶¶W•ÒÂF–7B’æB—6–ç7Fæ6R‡fÇVRÂF–7B“¢ÖW&vVE¶¶W•ÒÒÖW&vUö6öæf–w2†ÖW&vVE¶¶W•ÒÂfÇVRĞ¢VÇ6S¢ÖW&vVE¶¶W•ÒÒfÇVPĞ¢&WGW&âÖW&vV@Ğ Ğ¢2ÒÒÒ6†V6·ö–çB'Væ–ær†VÇW"ÒÒĞĞ¢26†V6"Ö6÷2FVÆWF66†R'VæUö6†V6·ö–çG0Ğ¦FVb'VæUö6†V6·ö–çG2†÷WGWEöF—"Â¶VWöÆ7B“ Ğ¢–b¶VWöÆ7BÂ¢Æövv–æræFV'Vr†b%'Væ–ærF—6&ÆVB†¶VW×¶¶VWöÆ7GÒ’â"“²&WGW&àĞ¢–b¶VWöÆ7BÓÒ¢Æövv–æræ–æfò‚%'Væ–ærÆÂWö6‚6†V6·ö–çG2†¶VWÓ’â"’2¶VW”ädòf÷"F†—266PĞ¢G'“ Ğ¢6·Eöf–ÆW5ö–æfòÒµÓ²GFW&âÒ÷2çF‚æ¦ö–â†÷WGWEöF—"ÂwGVæWEöWö6…ò¢çF‚r“²Wö6…öf–ÆW2ÒvÆö"ævÆö"‡GFW&âĞ¢f÷"e÷F‚–âWö6…öf–ÆW3 Ğ¢&6VæÖRÒ÷2çF‚æ&6VæÖR†e÷F‚“²ÖF6‚Ò&RæÖF6‚‡"'GVæWEöWö6…ò…ÆB²•ÂçF‚"Â&6VæÖRĞ¢–bÖF6ƒ Ğ¢Wö6…öçVÒÒ–çB†ÖF6‚æw&÷Wƒ’Ğ¢G'“¢×F–ÖRÒ÷2çF‚ævWF×F–ÖR†e÷F‚“²6·Eöf–ÆW5ö–æfòæVæB‡²wF‚s¢e÷F‚ÂvWö6‚s¢Wö6…öçVÒÂv×F–ÖRs¢×F–ÖWÒĞ¢W†6WBõ4W'&÷"2S¢Æövv–ærçv&æ–ær†b%'Væ–æs¢6¶—¶&6VæÖWÒÂ6ææ÷BvWB×F–ÖS¢¶WÒ"’2¶VWv&æ–æpĞ¢6·Eöf–ÆW5ö–æfòç6÷'B†¶W“ÖÆÖ&Fƒ¢‡…²vWö6‚uÒÂ…²v×F–ÖRuÒ’Â&WfW'6SÕG'VRĞ¢–bÆVâ†6·Eöf–ÆW5ö–æfò’ÃÒ¶VWöÆ7C¢Æövv–æræFV'Vr†b%'Væ–æs¢f÷VæB¶ÆVâ†6·Eöf–ÆW5ö–æfò—ÒÂ¶VW–ær¶¶VWöÆ7GÒâæöæR&VÖ÷fVBâ"“²&WGW&àĞ¢f–ÆW5÷Fõ÷&VÖ÷fRÒ6·Eöf–ÆW5ö–æfõ¶¶VWöÆ7C¥Ó²&VÖ÷fVEö6÷VçBÒ Ğ¢2¶VW”ädòf÷"'Væ–ær7F'BÖW76vPĞ¢Æövv–æræ–æfò†b%'Væ–æs¢¶VW–ærÆ7B¶¶VWöÆ7GÒ6†V6·ö–çG2â&VÖ÷f–ær¶ÆVâ†f–ÆW5÷Fõ÷&VÖ÷fR—Òöb¶ÆVâ†6·Eöf–ÆW5ö–æfò—ÒF÷FÂâ"Ğ¢f÷"6·Eö–æfò–âf–ÆW5÷Fõ÷&VÖ÷fS Ğ¢G'“ Ğ¢÷2ç&VÖ÷fR†6·Eö–æfõ²wF‚uÒĞ¢2W6RDT%Trf÷"–æF—f–GVÂf–ÆR&VÖ÷fÂÆöw0Ğ¢Æövv–æræFV'Vr†b"&VÖ÷fVC¢¶÷2çF‚æ&6VæÖR†6·Eö–æfõ²wF‚uÒ—Ò"Ğ¢&VÖ÷fVEö6÷VçB³ÒĞ¢W†6WBW†6WF–öâ2S¢Æövv–ærçv&æ–ær†b"f–ÆVB&VÖ÷fR¶÷2çF‚æ&6VæÖR†6·Eö–æfõ²wF‚uÒ—Ó¢¶WÒ"’2¶VWv&æ–æpĞ¢2¶VW”ädòf÷"'Væ–ær7VÖÖ'Ğ¢–b&VÖ÷fVEö6÷VçBâ¢Æövv–æræ–æfò†b%'Væ–ærf–æ—6†VBâ&VÖ÷fVB·&VÖ÷fVEö6÷VçGÒ6†V6·ö–çB‡2’â"Ğ¢W†6WBW†6WF–öâ2S¢Æövv–æræW'&÷"†b$6†V6·ö–çB'Væ–ærW'&÷"–âw¶÷WGWEöF—'Òs¢¶WÒ"ÂW†5ö–æfóÕG'VR’2¶VWW'&÷ Ğ Ğ Ğ¢2ÒÒÒÖ–âW†V7WF–öâ†–bõöæÖUõòÓÒ%õöÖ–åõò#¦’ÒÒĞĞ¦–bõöæÖUõòÓÒ%õöÖ–åõò# Ğ¢'6W"Ò&w'6Rä&wVÖVçE'6W"†FW67&—F–öãÒuG&–âTæWBf–”ÔÂ6öæf–rW6–ærDE„7&÷72ÕÆFf÷&Ò’rĞ¢'6W"æFEö&wVÖVçB‚rÒÖ6öæf–rrÂG—S×7G"Â&WV—&VCÕG'VRÂ†VÇÒuF‚FòF†RU4U"”ÔÂ6öæf–wW&F–öâf–ÆRrĞ¢'6W"æFEö&wVÖVçB‚rÒ×G&–æ–æræ&F6…÷6—¦RrÂG—SÖ–çBÂFW7CÒwG&–æ–æuö&F6…÷6—¦RrÂFVfVÇCÔæöæRÂ†VÇÒt÷fW'&–FRG&–æ–ær&F6…÷6—¦RrĞ¢'6W"æFEö&wVÖVçB‚rÒ×G&–æ–æræÇ"rÂG—SÖfÆöBÂFW7CÒwG&–æ–æuöÇ"rÂFVfVÇCÔæöæRÂ†VÇÒt÷fW'&–FRG&–æ–ærÆV&æ–æu÷&FRrĞ¢'6W"æFEö&wVÖVçB‚rÒÖFFæ÷WGWEöF—"rÂG—S×7G"ÂFW7CÒvFFö÷WGWEöF—"rÂFVfVÇCÔæöæRÂ†VÇÒt÷fW'&–FRFF÷WGWEöF—"rĞ¢6Æ•ö&w2Ò'6W"ç'6Uö&w2‚Ğ Ğ¢2ÒÒÒ–æ—F–Â6öæf–rÆöF–ærÆövv–ær…6–×ÆR6öç6öÆR’ÒÒĞĞ¢–æ—F–ÅöÆöuöÆWfVÂÒÆövv–ærä”ädğĞ¢–æ—F–ÅöÆöuöf÷&ÖBÒrR†67F–ÖR—2´4ôäd”uÒR†ÖW76vR—2r26–×ÆRf÷&ÖBf÷"6öæf–r†6PĞ¢Æövv–æræ&6–46öæf–r†ÆWfVÃÖ–æ—F–ÅöÆöuöÆWfVÂÂf÷&ÖCÖ–æ—F–ÅöÆöuöf÷&ÖBÂ†æFÆW'3Õ¶Æövv–ærå7G&VÔ†æFÆW"‚•ÒÂf÷&6SÕG'VR’2W6Rf÷&6SÕG'VRFò÷fW'&–FR÷FVçF–Â&Wf–÷W2&6–46öæf–w0Ğ¢Æövv–æræ–æfò‚%67&—B7F'F–ærâââ"Ğ¢Æövv–æræ–æfò†b%W6W"6öæf–s¢¶6Æ•ö&w2æ6öæf–wÒ"Ğ Ğ¢2ÒÒÒÆöB&6R6öæf–rÒÒĞĞ¢&6Uö6öæf–uöF–7BÒ·ĞĞ¢G'“ Ğ¢W6W%ö6fuö'2Ò÷2çF‚æ'7F‚†6Æ•ö&w2æ6öæf–r“²W6W%ö6fuöF—"Ò÷2çF‚æF—&æÖR‡W6W%ö6fuö'2“²67&—EöF—"Ò÷2çF‚æF—&æÖR†÷2çF‚æ'7F‚…õöf–ÆUõò’Ğ¢&6Uö÷G2Ò¶÷2çF‚æ¦ö–â‡W6W%ö6fuöF—"Âv&6RrÂv&6Rç–ÖÂr’Â÷2çF‚æ¦ö–â‡67&—EöF—"Âv&6RrÂv&6Rç–ÖÂr•ĞĞ¢&6U÷F‚ÒæW‡B‚‡f÷"–â&6Uö÷G2–b÷2çF‚æW†—7G2‡’’ÂæöæRĞ¢–bæ÷B&6U÷Fƒ¢&—6Rf–ÆTæ÷Df÷VæDW'&÷"†b$&6R6öæf–r‚v&6Rö&6Rç–ÖÂr’æ÷Bf÷VæB–â¶&6Uö÷G7Ò"Ğ¢v—F‚÷Vâ†&6U÷F‚Âw"r’2c¢&6Uö6öæf–uöF–7BÒ–ÖÂç6fUöÆöB†b’÷"·ĞĞ¢Æövv–æræ–æfò†b$ÆöFVB&6R6öæf–s¢¶&6U÷F‡Ò"’2¶VW”ädğĞ¢W†6WBW†6WF–öâ2S¢Æövv–æræW'&÷"†b$5$•D”4Ã¢f–ÆVBÆöF–ær&6R6öæf–s¢¶WÒâW†—F–ærâ"ÂW†5ö–æfóÕG'VR“²W†—Bƒ’2¶VWW'&÷ Ğ Ğ¢2ÒÒÒÆöBW6W"6öæf–rÒÒĞĞ¢W6W%ö6öæf–uöF–7BÒ·ĞĞ¢G'“ Ğ¢W6W%ö6fuö'2Ò÷2çF‚æ'7F‚†6Æ•ö&w2æ6öæf–rĞ¢–bæ÷B÷2çF‚æW†—7G2‡W6W%ö6fuö'2“¢&—6Rf–ÆTæ÷Df÷VæDW'&÷"†b%W6W"6öæf–ræ÷Bf÷VæC¢·W6W%ö6fuö'7Ò"Ğ¢v—F‚÷Vâ‡W6W%ö6fuö'2Âw"r’2c¢W6W%ö6öæf–uöF–7BÒ–ÖÂç6fUöÆöB†b’÷"·ĞĞ¢Æövv–æræ–æfò†b$ÆöFVBW6W"6öæf–s¢·W6W%ö6fuö'7Ò"’2¶VW”ädğĞ¢W†6WBW†6WF–öâ2S¢Æövv–æræW'&÷"†b$5$•D”4Ã¢f–ÆVBÆöF–ærW6W"6öæf–s¢¶WÒâW†—F–ærâ"ÂW†5ö–æfóÕG'VR“²W†—Bƒ’2¶VWW'&÷ Ğ Ğ¢2ÒÒÒÖW&vRb÷fW'&–FW2ÒÒĞĞ¢G'“¢ÖW&vVEöF–7BÒÖW&vUö6öæf–w2†&6Uö6öæf–uöF–7BÂW6W%ö6öæf–uöF–7BĞ¢W†6WBW†6WF–öâ2S¢Æövv–æræW'&÷"†b$5$•D”4Ã¢6öæf–rÖW&vRf–ÆVC¢¶WÒâW†—F–ærâ"ÂW†5ö–æfóÕG'VR“²W†—Bƒ’2¶VWW'&÷ Ğ¢÷fW'&–FW2Ò·ĞĞ¢–b6Æ•ö&w2çG&–æ–æuö&F6…÷6—¦R—2æ÷BæöæS¢÷fW'&–FW2ç6WFFVfVÇB‚wG&–æ–ærrÂ·Ò•²v&F6…÷6—¦RuÒÒ6Æ•ö&w2çG&–æ–æuö&F6…÷6—¦PĞ¢–b6Æ•ö&w2çG&–æ–æuöÇ"—2æ÷BæöæS¢÷fW'&–FW2ç6WFFVfVÇB‚wG&–æ–ærrÂ·Ò•²vÇ"uÒÒ6Æ•ö&w2çG&–æ–æuöÇ Ğ¢–b6Æ•ö&w2æFFö÷WGWEöF—"—2æ÷BæöæS¢÷fW'&–FW2ç6WFFVfVÇB‚vFFrÂ·Ò•²v÷WGWEöF—"uÒÒ÷2çF‚æ'7F‚†6Æ•ö&w2æFFö÷WGWEöF—"Ğ¢–b÷fW'&–FW3¢ÖW&vVEöF–7BÒÖW&vUö6öæf–w2†ÖW&vVEöF–7BÂ÷fW'&–FW2“²Æövv–æræ–æfò†b$Æ–VB4Ä’÷fW'&–FW3¢¶÷fW'&–FW7Ò"’2¶VW”ädğĞ Ğ¢2ÒÒÒ6öçfW'BFòæÖW76RbfÆ–FFRÒÒĞĞ¢G'“¢6öæf–rÒF–7E÷Fõ÷6ç2†ÖW&vVEöF–7BĞ¢W†6WBW†6WF–öâ2S¢Æövv–æræW'&÷"†b$5$•D”4Ã¢6öæf–r6öçfW'6–öâf–ÆVC¢¶WÒâW†—F–ærâ"ÂW†5ö–æfóÕG'VR“²W†—Bƒ’2¶VWW'&÷ Ğ Ğ¢2ÒÒÒ6öæf–wW&F–öâfÆ–FF–öâ„¶VW6öæ6—6RÂW'&÷'2&R7&—F–6Â’ÒÒĞĞ¢Ö—76–ærÂW'&÷%ö×6w2ÒµÒÂµĞĞ¢&Wö¶W—2Ò²vFFs¢²w7&5öF—"rÂvG7EöF—"rÂv÷WGWEöF—"rÂw&W6öÇWF–öârÂv÷fW&Æöf7F÷"uÒÂvÖöFVÂs¢²vÖöFVÅ÷6—¦UöF–×2uÒÂwG&–æ–ærs¢²v—FW&F–öç5÷W%öWö6‚rÂv&F6…÷6—¦RrÂvÇ"rÂvÆ÷72rÂvÆÖ&FöÇ—2rÂwW6Uö×uÒÂw6f–ærs¢²v¶VWöÆ7Eö6†V6·ö–çG2u×ĞĞ¢26†V6²&WV—&VB6V7F–öç2æB¶W—0Ğ¢f÷"6V2Â¶W—2–â&Wö¶W—2æ—FV×2‚“ Ğ¢6V5öö&¢ÒvWFGG"†6öæf–rÂ6V2ÂæöæRĞ¢–b6V5öö&¢—2æöæS Ğ¢W'&÷%ö×6w2æVæB†b$Ö—76–ær&WV—&VB6V7F–öã¢w·6V7Òr"Ğ¢Ö—76–æræVæB‡6V2Ğ¢6öçF–çVR26¶—¶W—26†V6²–b6V7F–öâ—2Ö—76–æpĞ¢2–b6V7F–öâW†—7G2Â6†V6²—G2¶W—0Ğ¢f÷"²–â¶W—3 Ğ¢–bvWFGG"‡6V5öö&¢Â²ÂæöæR’—2æöæS Ğ¢W'&÷%ö×6w2æVæB†b$Ö—76–æröçVÆÂ&WV—&VBfÇVS¢w·6V7Òç¶·Òr"Ğ¢Ö—76–æræVæB†b'·6V7Òç¶·Ò"Ğ Ğ Ğ¢2FVfVÇF–ær‡W6R6–×ÆRvWFGG"v—F‚FVfVÇBÂfö–BW‡G&Æöw2†W&RĞ¢6öæf–ræFFÆöFW"ÒvWFGG"†6öæf–rÂvFFÆöFW"rÂ6–×ÆTæÖW76R‚’Ğ¢FVe÷v÷&²Ò–b5U%$TåEôõ2ÓÒuv–æF÷w2rVÇ6R#²6öæf–ræFFÆöFW"æçVÕ÷v÷&¶W'2ÒvWFGG"†6öæf–ræFFÆöFW"ÂvçVÕ÷v÷&¶W'2rÂFVe÷v÷&²Ğ¢6öæf–ræFFÆöFW"ç&VfWF6…öf7F÷"ÒvWFGG"†6öæf–ræFFÆöFW"Âw&VfWF6…öf7F÷"rÂ"–b6öæf–ræFFÆöFW"æçVÕ÷v÷&¶W'2âVÇ6RæöæRĞ¢6öæf–rç6f–ærÒvWFGG"†6öæf–rÂw6f–ærrÂ6–×ÆTæÖW76R‚’“²6öæf–rç6f–ærç6fUö—FW&F–öç5ö–çFW'fÂÒvWFGG"†6öæf–rç6f–ærÂw6fUö—FW&F–öç5ö–çFW'fÂrÂĞ¢6öæf–ræÆövv–ærÒvWFGG"†6öæf–rÂvÆövv–ærrÂ6–×ÆTæÖW76R‚’“²6öæf–ræÆövv–æræÆöuö–çFW'fÂÒvWFGG"†6öæf–ræÆövv–ærÂvÆöuö–çFW'fÂrÂSĞ¢6öæf–ræÆövv–ærç&Wf–Wuö&F6…ö–çFW'fÂÒvWFGG"†6öæf–ræÆövv–ærÂw&Wf–Wuö&F6…ö–çFW'fÂrÂS“²6öæf–ræÆövv–ærç&Wf–Wu÷&Vg&W6…÷&FRÒvWFGG"†6öæf–ræÆövv–ærÂw&Wf–Wu÷&Vg&W6…÷&FRrÂRĞ¢2f–æÂfÇVR6†V6·0Ğ¢–bæ÷BÖ—76–æs Ğ¢G'“¢26öÖ&–æR6†V6·2f÷"'&Wf—GĞ¢–b6öæf–rçG&–æ–æræ—FW&F–öç5÷W%öWö6‚ÃÒ¢W'&÷%ö×6w2æVæB‚'G&–âæ—FW%öWö6ƒÃÓ"Ğ¢–b6öæf–rçG&–æ–æræ&F6…÷6—¦RÃÒ¢W'&÷%ö×6w2æVæB‚'G&–âæ&F6…÷6—¦SÃÓ"Ğ¢–bæ÷B—6–ç7Fæ6R†6öæf–rçG&–æ–æræÇ"Â†–çBÂfÆöB’’÷"6öæf–rçG&–æ–æræÇ"ÃÒ¢W'&÷%ö×6w2æVæB‚'G&–âæÇ#ÃÓ"Ğ¢–b6öæf–rçG&–æ–æræÆ÷72æ÷B–â²vÃrÂvÃ¶Ç—2uÓ¢W'&÷%ö×6w2æVæB‚'G&–âæÆ÷72–çfÆ–B"Ğ¢–b6öæf–rçG&–æ–æræÆÖ&FöÇ—2Â¢W'&÷%ö×6w2æVæB‚'G&–âæÆÖ&FöÇ—3Ã"Ğ¢–bæ÷BƒãÃÒ6öæf–ræFFæ÷fW&Æöf7F÷"Âã“¢W'&÷%ö×6w2æVæB‚&FFæ÷fW&Æ–çfÆ–B"Ğ¢G3Óc°Ğ¢–b6öæf–ræFFç&W6öÇWF–öãÃÓ÷"6öæf–ræFFç&W6öÇWF–öâRG2Ó¢W'&÷%ö×6w2æVæB†b&FFç&W6öÇWF–öâæ÷BãbF—b'’¶G7Ò"Ğ¢–bæ÷B÷2çF‚æ—6F—"†6öæf–ræFFç7&5öF—"“¢W'&÷%ö×6w2æVæB†b&FFç7&5öF—"Ö—76–æs¢¶6öæf–ræFFç7&5öF—'Ò"’2FFVBF‚Fò×6pĞ¢–bæ÷B÷2çF‚æ—6F—"†6öæf–ræFFæG7EöF—"“¢W'&÷%ö×6w2æVæB†b&FFæG7EöF—"Ö—76–æs¢¶6öæf–ræFFæG7EöF—'Ò"’2FFVBF‚Fò×6pĞ¢–b6öæf–ræÖöFVÂæÖöFVÅ÷6—¦UöF–×2ÃÒ¢W'&÷%ö×6w2æVæB‚&ÖöFVÂç6—¦SÃÓ"Ğ¢–b6öæf–ræFFÆöFW"æçVÕ÷v÷&¶W'2Â¢W'&÷%ö×6w2æVæB‚&ÆöFW"çv÷&¶W'3Ã"Ğ¢–b6öæf–ræFFÆöFW"æçVÕ÷v÷&¶W'3ãæB6öæf–ræFFÆöFW"ç&VfWF6…öf7F÷"—2æ÷BæöæRæB6öæf–ræFFÆöFW"ç&VfWF6…öf7F÷#Ã#¢W'&÷%ö×6w2æVæB‚&ÆöFW"ç&VfWF6ƒÃ"–çfÆ–B"Ğ¢–b6öæf–rç6f–ærç6fUö—FW&F–öç5ö–çFW'fÂÂ¢W'&÷%ö×6w2æVæB‚'6fRæ–çFW'fÃÃ"Ğ¢–b6öæf–rç6f–æræ¶VWöÆ7Eö6†V6·ö–çG2ÂÓ¢W'&÷%ö×6w2æVæB‚'6fRæ¶VWÂÓ"Ğ¢–b6öæf–ræÆövv–æræÆöuö–çFW'fÂÃÒ¢W'&÷%ö×6w2æVæB‚&Æöræ–çFW'fÃÃÓ"Ğ¢–b6öæf–ræÆövv–ærç&Wf–Wuö&F6…ö–çFW'fÂÂ¢W'&÷%ö×6w2æVæB‚&Æörç&Wf–WsÃ"Ğ¢–b6öæf–ræÆövv–ærç&Wf–Wu÷&Vg&W6…÷&FRÂ¢W'&÷%ö×6w2æVæB‚&Æörç&Vg&W6ƒÃ"Ğ¢W†6WBW†6WF–öâ2S¢W'&÷%ö×6w2æVæB†b%fÆ–FF–öâW'&÷#¢¶WÒ"Ğ¢2&÷'BöâW'&÷'0Ğ¢–bW'&÷%ö×6w2÷"Ö—76–æs Ğ¢&–çB‚%Æâ"²#Ò"£3²"4ôäd”rU%$õ%2"²#Ò"£3“²ÆÅöW''2Ò6÷'FVB†Æ—7B‡6WB†W'&÷%ö×6w2²¶b$Ö—76–æs¢¶×Ò"f÷"Ò–âÖ—76–æuÒ’’Ğ¢f÷"’Â×6r–âVçVÖW&FR†ÆÅöW''2“¢&–çB†b"¶’³Òâ¶×6wÒ"Ğ¢&–çB‚#Ò"£ƒ"²%ÆåÆV6Rf—‚6öæf–ræB&W7F'Bâ"“²W†—BƒĞ¢VÇ6S¢Æövv–æræ–æfò‚%´4ôäd”uÒ6öæf–wW&F–öâfÆ–FFVBâ"’2¶VW6–×ÆR”ädò6öæf—&ÖF–öàĞ Ğ¢2DEVçbf"6†V6²„¶VW2v&æ–ærĞ¢–b÷2æVçf—&öâævWB‚%tõ$ÄEõ4•¤R"Â#"’Ò#"æB÷2æVçf—&öâævWB‚$Äô4Åõ$ä²"ÂæöæR’—2æöæS Ğ¢Æövv–ærçv&æ–ær‚$DE'VâFWFV7FVB…tõ$ÄEõ4•¤Sã’'WBÄô4Åõ$ä²æ÷B6WBâW6RwF÷&6‡'Vâr÷"Vç7W&RVçbf'2&R6WBâ"’2¶VWv&æ–æpĞ Ğ¢2ÒÒÒ7F'BG&–æ–ærÒÒĞĞ¢G'“¢G&–â†6öæf–rĞ¢W†6WBW†6WF–öâ2Ö–åöW'#¢Æövv–æræW'&÷"‚%Væ†æFÆVBW'&÷"÷WG6–FRG&–âÆö÷¢"ÂW†5ö–æfóÕG'VR“²6ÆVçWöFG‚“²W†—Bƒ’26—R Ğ 