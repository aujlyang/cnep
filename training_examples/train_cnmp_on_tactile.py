"""
Training example for CNMP with tactile image and force conditioning.

This script demonstrates how to train a CNMP model on robot tactile data,
using tactile images and 3D force as conditional inputs to model robot
end-effector trajectories.

Data structure expected:
- data_folder/
  - demo_0/
    - img/
      - 000.jpg
      - 001.jpg
      - ...
    - data.csv  # Contains trajectory (x,y,z) and force (fx,fy,fz)
  - demo_1/
    ...
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# Add models folder to path
folder_path = '../models/'
if folder_path not in sys.path:
    sys.path.append(folder_path)

# Add data folder to path
data_folder_path = '../data/'
if data_folder_path not in sys.path:
    sys.path.append(data_folder_path)

from cnmp import CNMP
from tactile_data_loader import TactileDataset, create_feature_extractor

torch.set_float32_matmul_precision('high')


def get_free_gpu():
    """Get the GPU with lowest utilization."""
    if not torch.cuda.is_available():
        return None
    
    gpu_util = []
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        gpu_util.append((i, torch.cuda.utilization()))
    gpu_util.sort(key=lambda x: x[1])
    return gpu_util[0][0]


# Device setup
if torch.cuda.is_available():
    available_gpu = get_free_gpu()
    device = torch.device(f"cuda:{available_gpu}" if available_gpu is not None else "cuda:0")
else:
    device = torch.device("cpu")

print("Device:", device)


# ============================================================================
# Data Loading Configuration
# ============================================================================

# TODO: Update these paths to your actual data location
data_folder = './tactile_data'  # Root folder containing demonstrations
demo_folders = [f'demo_{i}' for i in range(10)]  # List of demonstration folders

# CSV configuration
csv_filename = 'data.csv'  # Name of CSV file in each demo folder
trajectory_cols = [0, 1, 2]  # Columns for x, y, z positions
force_cols = [3, 4, 5]  # Columns for fx, fy, fz forces

# Image configuration
img_subfolder = 'img'  # Subfolder containing images
use_feature_extractor = True  # Use CNN features instead of raw images
feature_extractor_name = 'mobilenet_v2'  # Options: 'mobilenet_v2', 'resnet18', 'resnet50'

# Sampling configuration
num_timesteps = 200  # Number of timesteps to sample from each demonstration


# ============================================================================
# Load Data
# ============================================================================

print("Loading tactile dataset...")

# Create feature extractor if needed
img_feature_extractor = None
if use_feature_extractor:
    print(f"Creating feature extractor: {feature_extractor_name}")
    img_feature_extractor = create_feature_extractor(feature_extractor_name, device=device)

# Load dataset
dataset = TactileDataset(
    data_folder=data_folder,
    demo_folders=demo_folders,
    img_subfolder=img_subfolder,
    csv_filename=csv_filename,
    trajectory_cols=trajectory_cols,
    force_cols=force_cols,
    num_timesteps=num_timesteps,
    normalize=True,
    img_feature_extractor=img_feature_extractor,
    device=device
)

# Get all data
trajectories, forces, img_features = dataset.get_all_data()

print(f"\nDataset loaded:")
print(f"  Number of demonstrations: {len(dataset)}")
print(f"  Trajectory shape: {trajectories.shape}")  # (N, T, 3)
print(f"  Force shape: {forces.shape}")  # (N, T, 3)
print(f"  Image features shape: {img_features.shape}")  # (N, T, D)


# ============================================================================
# Split into train/validation
# ============================================================================

num_demos = len(dataset)
train_ratio = 0.8
num_train = int(num_demos * train_ratio)

# Random split
perm_ids = torch.randperm(num_demos)
train_ids = perm_ids[:num_train]
val_ids = perm_ids[num_train:]

train_trajs = trajectories[train_ids]
train_forces = forces[train_ids]
train_img_feats = img_features[train_ids]

val_trajs = trajectories[val_ids]
val_forces = forces[val_ids]
val_img_feats = img_features[val_ids]

print(f"\nTrain/Val split:")
print(f"  Train: {len(train_ids)} demos")
print(f"  Val: {len(val_ids)} demos")


# ============================================================================
# Model Configuration
# ============================================================================

# Dimensions
dx = 1  # Time dimension
dy = 3  # Trajectory dimension (x, y, z)
df = 3  # Force dimension (fx, fy, fz)
dg = img_features.shape[-1]  # Image feature dimension
d_cond = df + dg  # Total conditioning dimension (force + image features)

print(f"\nDimensions:")
print(f"  Time (dx): {dx}")
print(f"  Trajectory (dy): {dy}")
print(f"  Force (df): {df}")
print(f"  Image features (dg): {dg}")
print(f"  Total conditioning (d_cond): {d_cond}")

# Training hyperparameters
batch_size = 8
n_max, m_max = 20, 20  # Max number of observation/target points
t_steps = num_timesteps

# Model hyperparameters
encoder_hidden_dims = [512, 512, 512]
decoder_hidden_dims = [256, 256]

# Create CNMP model
cnmp_ = CNMP(
    input_dim=dx + d_cond,  # Time + force + image features
    output_dim=dy,  # 3D trajectory
    n_max=n_max,
    m_max=m_max,
    encoder_hidden_dims=encoder_hidden_dims,
    decoder_hidden_dims=decoder_hidden_dims,
    batch_size=batch_size,
    device=device
)

optimizer = torch.optim.Adam(lr=3e-4, params=cnmp_.parameters())

def get_parameter_count(model):
    total_num = 0
    for param in model.parameters():
        total_num += param.shape.numel()
    return total_num

print(f"\nModel created:")
print(f"  Parameters: {get_parameter_count(cnmp_):,}")

# Compile model if PyTorch >= 2.0
if torch.__version__ >= "2.0":
    cnmp = torch.compile(cnmp_)
else:
    cnmp = cnmp_


# ============================================================================
# Prepare batched training data
# ============================================================================

# Move data to device
train_trajs = train_trajs.to(device)
train_forces = train_forces.to(device)
train_img_feats = train_img_feats.to(device)
val_trajs = val_trajs.to(device)
val_forces = val_forces.to(device)
val_img_feats = val_img_feats.to(device)

# Pre-allocate tensors for training
obs = torch.zeros((batch_size, n_max, dx+d_cond+dy), dtype=torch.float32, device=device)
tar_x = torch.zeros((batch_size, m_max, dx+d_cond), dtype=torch.float32, device=device)
tar_y = torch.zeros((batch_size, m_max, dy), dtype=torch.float32, device=device)
obs_mask = torch.zeros((batch_size, n_max), dtype=torch.bool, device=device)
tar_mask = torch.zeros((batch_size, m_max), dtype=torch.bool, device=device)


def prepare_masked_batch(traj_ids):
    """Prepare a masked batch for training."""
    obs.fill_(0)
    tar_x.fill_(0)
    tar_y.fill_(0)
    obs_mask.fill_(False)
    tar_mask.fill_(False)
    
    for i, traj_id in enumerate(traj_ids):
        traj = train_trajs[traj_id]
        force = train_forces[traj_id]
        img_feat = train_img_feats[traj_id]
        
        # Random number of observation and target points
        n = torch.randint(1, n_max, (1,)).item()
        m = torch.randint(1, m_max, (1,)).item()
        
        # Random indices
        permuted_ids = torch.randperm(t_steps)
        n_ids = permuted_ids[:n]
        m_ids = permuted_ids[n:n+m]
        
        # Observations: [time, force, img_features, trajectory]
        obs[i, :n, :dx] = (n_ids.float() / t_steps).unsqueeze(1)  # Time
        obs[i, :n, dx:dx+df] = force[n_ids]  # Force
        obs[i, :n, dx+df:dx+d_cond] = img_feat[n_ids]  # Image features
        obs[i, :n, dx+d_cond:] = traj[n_ids]  # Trajectory
        obs_mask[i, :n] = True
        
        # Targets: [time, force, img_features] -> trajectory
        tar_x[i, :m, :dx] = (m_ids.float() / t_steps).unsqueeze(1)  # Time
        tar_x[i, :m, dx:dx+df] = force[m_ids]  # Force
        tar_x[i, :m, dx+df:] = img_feat[m_ids]  # Image features
        tar_y[i, :m] = traj[m_ids]  # Trajectory to predict
        tar_mask[i, :m] = True


# Pre-allocate tensors for validation
val_obs = torch.zeros((batch_size, n_max, dx+d_cond+dy), dtype=torch.float32, device=device)
val_tar_x = torch.zeros((batch_size, t_steps, dx+d_cond), dtype=torch.float32, device=device)
val_tar_y = torch.zeros((batch_size, t_steps, dy), dtype=torch.float32, device=device)
val_obs_mask = torch.zeros((batch_size, n_max), dtype=torch.bool, device=device)


def prepare_masked_val_batch(traj_ids):
    """Prepare a masked batch for validation."""
    val_obs.fill_(0)
    val_tar_x.fill_(0)
    val_tar_y.fill_(0)
    val_obs_mask.fill_(False)
    
    for i, traj_id in enumerate(traj_ids):
        traj = val_trajs[traj_id]
        force = val_forces[traj_id]
        img_feat = val_img_feats[traj_id]
        
        # Random number of observation points
        n = torch.randint(1, n_max, (1,)).item()
        
        # Random observation indices, all points as targets
        permuted_ids = torch.randperm(t_steps)
        n_ids = permuted_ids[:n]
        m_ids = torch.arange(t_steps, device=device)
        
        # Observations
        val_obs[i, :n, :dx] = (n_ids.float() / t_steps).unsqueeze(1)
        val_obs[i, :n, dx:dx+df] = force[n_ids]
        val_obs[i, :n, dx+df:dx+d_cond] = img_feat[n_ids]
        val_obs[i, :n, dx+d_cond:] = traj[n_ids]
        val_obs_mask[i, :n] = True
        
        # Targets (all timesteps)
        val_tar_x[i, :, :dx] = (m_ids.float() / t_steps).unsqueeze(1)
        val_tar_x[i, :, dx:dx+df] = force[m_ids]
        val_tar_x[i, :, dx+df:] = img_feat[m_ids]
        val_tar_y[i] = traj[m_ids]


# ============================================================================
# Training Loop
# ============================================================================

print("\n" + "="*70)
print("Starting training...")
print("="*70)

# Create output directory
import time
timestamp = int(time.time())
root_folder = f'../outputs/tactile/cnmp/{timestamp}/'

os.makedirs(root_folder, exist_ok=True)
os.makedirs(f'{root_folder}saved_models/', exist_ok=True)

# Training configuration
epochs = 100_000
epoch_iter = len(train_ids) // batch_size
v_epoch_iter = max(1, len(val_ids) // batch_size)

val_per_epoch = 1000
min_val_err = float('inf')

mse_loss = torch.nn.MSELoss()

train_losses = []
val_errors = []

for epoch in range(epochs):
    # Training
    epoch_loss = 0
    traj_ids = torch.randperm(len(train_ids))[:batch_size * epoch_iter].chunk(epoch_iter)
    
    for i in range(epoch_iter):
        prepare_masked_batch(traj_ids[i])
        
        optimizer.zero_grad()
        pred = cnmp(obs, tar_x, obs_mask)
        loss = cnmp.loss(pred, tar_y, tar_mask)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    epoch_loss /= epoch_iter
    train_losses.append(epoch_loss)
    
    # Validation
    if epoch % val_per_epoch == 0:
        with torch.no_grad():
            val_err = 0
            v_traj_ids = torch.randperm(len(val_ids))[:batch_size * v_epoch_iter].chunk(v_epoch_iter)
            
            for j in range(v_epoch_iter):
                prepare_masked_val_batch(v_traj_ids[j])
                
                pred = cnmp.val(val_obs, val_tar_x, val_obs_mask)
                pred_means = pred[:, :, :dy]
                val_err += mse_loss(pred_means, val_tar_y).item()
            
            val_err /= v_epoch_iter
            val_errors.append(val_err)
            
            if val_err < min_val_err:
                min_val_err = val_err
                print(f'Epoch {epoch}: New best validation error: {min_val_err:.6f}')
                torch.save(cnmp_.state_dict(), f'{root_folder}saved_models/cnmp_best.pt')
            else:
                print(f'Epoch {epoch}: Loss={epoch_loss:.6f}, Val Err={val_err:.6f}, Best={min_val_err:.6f}')

print("\n" + "="*70)
print("Training completed!")
print(f"Best validation error: {min_val_err:.6f}")
print(f"Model saved to: {root_folder}saved_models/cnmp_best.pt")
print("="*70)

# Save training history
torch.save(torch.tensor(train_losses), f'{root_folder}train_losses.pt')
torch.save(torch.tensor(val_errors), f'{root_folder}val_errors.pt')
