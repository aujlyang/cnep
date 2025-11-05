# Training CNEP/CNMP with Tactile Data

This guide demonstrates how to use tactile images and 3D force as conditional inputs for CNEP and CNMP models to learn robot end-effector trajectory primitives.

## Overview

The implementation allows you to:
- Use tactile images as visual conditioning signals
- Use 3D force measurements as additional conditioning inputs
- Model 3D robot end-effector trajectories
- Train CNEP (with multiple expert decoders) or CNMP models

## Data Structure

Your data should be organized as follows:

```
data_folder/
├── demo_0/
│   ├── img/
│   │   ├── 000.jpg
│   │   ├── 001.jpg
│   │   ├── 002.jpg
│   │   └── ...
│   └── data.csv
├── demo_1/
│   ├── img/
│   │   └── ...
│   └── data.csv
└── ...
```

### CSV File Format

The CSV file should contain the trajectory and force data with the following structure:

```csv
x,y,z,fx,fy,fz
0.1,0.2,0.3,0.5,0.1,0.2
0.11,0.21,0.31,0.52,0.12,0.21
...
```

- Columns 0-2: 3D robot end-effector position (x, y, z)
- Columns 3-5: 3D force measurements (fx, fy, fz)

You can customize which columns to use in the data loader.

### Image Files

- Images should be named sequentially: `000.jpg`, `001.jpg`, `002.jpg`, etc.
- Supported formats: `.jpg`, `.jpeg`, `.png`
- Images are automatically resized and preprocessed
- Features are extracted using pre-trained CNN models (MobileNetV2 by default)

## Usage

### 1. Using the Data Loader

```python
from data.tactile_data_loader import TactileDataset, create_feature_extractor
import torch

# Setup device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create feature extractor
img_feature_extractor = create_feature_extractor('mobilenet_v2', device=device)

# Load dataset
dataset = TactileDataset(
    data_folder='./your_data_folder',
    demo_folders=['demo_0', 'demo_1', 'demo_2'],
    img_subfolder='img',
    csv_filename='data.csv',
    trajectory_cols=[0, 1, 2],  # x, y, z columns in CSV
    force_cols=[3, 4, 5],       # fx, fy, fz columns in CSV
    num_timesteps=200,           # Sample to 200 timesteps
    normalize=True,              # Normalize data to [-1, 1]
    img_feature_extractor=img_feature_extractor,
    device=device
)

# Get data
trajectories, forces, img_features = dataset.get_all_data()
print(f"Trajectories: {trajectories.shape}")  # (N, T, 3)
print(f"Forces: {forces.shape}")              # (N, T, 3)
print(f"Image features: {img_features.shape}") # (N, T, D)
```

### 2. Training CNEP

```bash
cd training_examples
python train_cnep_on_tactile.py
```

Edit the script to configure:
- `data_folder`: Path to your data
- `demo_folders`: List of demonstration folders
- `num_timesteps`: Number of timesteps to sample
- `feature_extractor_name`: 'mobilenet_v2', 'resnet18', or 'resnet50'

### 3. Training CNMP

```bash
cd training_examples
python train_cnmp_on_tactile.py
```

Same configuration options as CNEP training.

## Customization

### Using Different CSV Columns

If your CSV has a different structure, specify the column indices:

```python
dataset = TactileDataset(
    data_folder='./data',
    demo_folders=['demo_0'],
    trajectory_cols=[1, 2, 3],  # Different columns for trajectory
    force_cols=[7, 8, 9],       # Different columns for force
    # ... other parameters
)
```

### Using Different Image Features

Change the feature extractor model:

```python
# Options: 'mobilenet_v2', 'resnet18', 'resnet50'
img_feature_extractor = create_feature_extractor('resnet18', device=device)
```

### Custom Image Transforms

Provide your own image preprocessing:

```python
from torchvision import transforms

custom_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

dataset = TactileDataset(
    # ... other parameters
    img_transform=custom_transform,
)
```

### Without Image Feature Extraction

To use raw image pixels instead of CNN features (not recommended for training):

```python
dataset = TactileDataset(
    # ... other parameters
    img_feature_extractor=None,  # Use raw pixels
)
```

## Model Architecture

### Input Structure

The models receive:
- **Observations**: `[time, force, image_features, trajectory]`
  - Time: Normalized timestamp (0 to 1)
  - Force: 3D force measurements
  - Image features: Extracted visual features
  - Trajectory: 3D position (used for conditioning)

- **Targets**: `[time, force, image_features]` → predicts trajectory

### CNEP vs CNMP

- **CNEP** (Conditional Neural Expert Processes): Uses multiple expert decoders with a gating mechanism, better for multimodal trajectory distributions
- **CNMP** (Conditional Neural Movement Primitives): Single decoder, simpler and faster, suitable for unimodal distributions

## Examples

### Example 1: Simple Training

```python
from data.tactile_data_loader import TactileDataset, create_feature_extractor
from models.cnep import CNEP
import torch

device = 'cuda'

# Load data
feature_extractor = create_feature_extractor('mobilenet_v2', device)
dataset = TactileDataset(
    data_folder='./tactile_data',
    demo_folders=[f'demo_{i}' for i in range(20)],
    num_timesteps=200,
    img_feature_extractor=feature_extractor,
    device=device
)

trajectories, forces, img_features = dataset.get_all_data()

# Create model
input_dim = 1 + 3 + img_features.shape[-1]  # time + force + img_features
output_dim = 3  # 3D trajectory

model = CNEP(
    input_dim=input_dim,
    output_dim=output_dim,
    n_max=20,
    m_max=20,
    encoder_hidden_dims=[512, 512, 512],
    num_decoders=4,
    decoder_hidden_dims=[256, 256],
    batch_size=8,
    device=device
)

# Train model (see training scripts for complete implementation)
```

### Example 2: Inference

```python
# Load trained model
model.load_state_dict(torch.load('saved_model.pt'))
model.eval()

# Prepare observation and target data
# obs: (batch, n_obs, input_dim + output_dim)
# tar: (batch, n_tar, input_dim)
# obs_mask: (batch, n_obs)

with torch.no_grad():
    pred, gate = model.val(obs, tar, obs_mask)
    
    # For CNEP, select best decoder
    dec_id = torch.argmax(gate.squeeze(1), dim=-1)
    pred_traj = pred[dec_id, torch.arange(batch_size), :, :3]
    
    # Denormalize back to original scale
    pred_traj = dataset.denormalize_trajectory(pred_traj)
```

## Troubleshooting

### Out of Memory

- Reduce `batch_size`
- Reduce `num_timesteps`
- Use a smaller feature extractor (e.g., 'mobilenet_v2' instead of 'resnet50')

### Slow Training

- Use smaller hidden dimensions
- Reduce number of decoders (for CNEP)
- Use `torch.compile()` (PyTorch 2.0+)

### Poor Performance

- Increase model capacity (more layers, larger hidden dims)
- Increase number of demonstrations
- Adjust learning rate
- For CNEP: tune loss coefficients

## File Structure

```
cnep/
├── data/
│   └── tactile_data_loader.py    # Data loading utilities
├── models/
│   ├── cnep.py                    # CNEP model
│   └── cnmp.py                    # CNMP model
├── training_examples/
│   ├── train_cnep_on_tactile.py  # CNEP training script
│   ├── train_cnmp_on_tactile.py  # CNMP training script
│   └── TACTILE_TRAINING_GUIDE.md # This file
└── outputs/                       # Training outputs
```

## References

For more information on CNEP and CNMP, see:
- Paper: "Conditional Neural Expert Processes for Learning Movement Primitives From Demonstration"
- Main README.md in the repository root
