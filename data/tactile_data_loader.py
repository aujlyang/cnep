"""
Tactile Data Loader for CNEP/CNMP
Loads robot tactile datasets including:
- Robot end-effector trajectories (3D positions)
- Tactile images (e.g., 000.jpg, 001.jpg, ...)
- 3D force data from CSV files
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import csv
from typing import List, Tuple, Optional


class TactileDataset:
    """
    Dataset class for loading robot tactile data.
    
    Args:
        data_folder: Root folder containing demonstration data
        demo_folders: List of demonstration folder names (e.g., ['demo_0', 'demo_1', ...])
        img_subfolder: Subfolder name containing images (default: 'img')
        csv_filename: Name of CSV file containing trajectory and force data
        trajectory_cols: Column indices in CSV for 3D trajectory (default: [0, 1, 2])
        force_cols: Column indices in CSV for 3D force (default: [3, 4, 5])
        img_transform: Optional image transformation pipeline
        num_timesteps: Number of timesteps to sample from each demo (default: None, use all)
        normalize: Whether to normalize trajectory and force data to [-1, 1]
    """
    
    def __init__(
        self,
        data_folder: str,
        demo_folders: List[str],
        img_subfolder: str = 'img',
        csv_filename: str = 'data.csv',
        trajectory_cols: List[int] = [0, 1, 2],
        force_cols: List[int] = [3, 4, 5],
        img_transform: Optional[transforms.Compose] = None,
        num_timesteps: Optional[int] = None,
        normalize: bool = True,
        img_feature_extractor: Optional[torch.nn.Module] = None,
        device: str = 'cpu'
    ):
        self.data_folder = data_folder
        self.demo_folders = demo_folders
        self.img_subfolder = img_subfolder
        self.csv_filename = csv_filename
        self.trajectory_cols = trajectory_cols
        self.force_cols = force_cols
        self.num_timesteps = num_timesteps
        self.normalize = normalize
        self.device = device
        
        # Default image transform if none provided
        if img_transform is None:
            self.img_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.img_transform = img_transform
        
        self.img_feature_extractor = img_feature_extractor
        if self.img_feature_extractor is not None:
            self.img_feature_extractor.eval()
        
        # Load all demonstrations
        self.trajectories = []
        self.forces = []
        self.img_features = []
        
        self._load_data()
        
        # Normalization parameters
        if self.normalize:
            self._compute_normalization_params()
            self._normalize_data()
    
    def _load_data(self):
        """Load all demonstration data."""
        print(f"Loading {len(self.demo_folders)} demonstrations...")
        
        for demo_folder in self.demo_folders:
            demo_path = os.path.join(self.data_folder, demo_folder)
            
            # Load CSV data (trajectory and force)
            csv_path = os.path.join(demo_path, self.csv_filename)
            traj, force = self._load_csv_data(csv_path)
            
            # Load image features
            img_folder_path = os.path.join(demo_path, self.img_subfolder)
            img_feats = self._load_image_features(img_folder_path, len(traj))
            
            self.trajectories.append(traj)
            self.forces.append(force)
            self.img_features.append(img_feats)
        
        print(f"Loaded {len(self.trajectories)} demonstrations")
        print(f"Trajectory shape: {self.trajectories[0].shape}")
        print(f"Force shape: {self.forces[0].shape}")
        print(f"Image features shape: {self.img_features[0].shape}")
    
    def _load_csv_data(self, csv_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load trajectory and force data from CSV file."""
        traj_data = []
        force_data = []
        
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header if present
            
            for row in reader:
                if len(row) > max(self.trajectory_cols + self.force_cols):
                    traj_point = [float(row[i]) for i in self.trajectory_cols]
                    force_point = [float(row[i]) for i in self.force_cols]
                    traj_data.append(traj_point)
                    force_data.append(force_point)
        
        traj = torch.tensor(traj_data, dtype=torch.float32)
        force = torch.tensor(force_data, dtype=torch.float32)
        
        # Resample if num_timesteps is specified
        if self.num_timesteps is not None and len(traj) != self.num_timesteps:
            indices = torch.linspace(0, len(traj) - 1, self.num_timesteps).long()
            traj = traj[indices]
            force = force[indices]
        
        return traj, force
    
    def _load_image_features(self, img_folder: str, num_frames: int) -> torch.Tensor:
        """Load and extract features from images."""
        # Get all image files
        img_files = sorted([f for f in os.listdir(img_folder) if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        # Sample images to match trajectory length
        if self.num_timesteps is not None:
            num_frames = self.num_timesteps
        
        img_indices = np.linspace(0, len(img_files) - 1, num_frames).astype(int)
        
        img_features = []
        
        for idx in img_indices:
            img_path = os.path.join(img_folder, img_files[idx])
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.img_transform(img).unsqueeze(0)
            
            # Extract features if feature extractor is provided
            if self.img_feature_extractor is not None:
                img_tensor = img_tensor.to(self.device)
                with torch.no_grad():
                    features = self.img_feature_extractor(img_tensor)
                    features = features.flatten()
                img_features.append(features.cpu())
            else:
                # Use raw image as features (flattened)
                img_features.append(img_tensor.flatten())
        
        return torch.stack(img_features)
    
    def _compute_normalization_params(self):
        """Compute min/max for normalization."""
        all_trajs = torch.stack(self.trajectories)
        all_forces = torch.stack(self.forces)
        
        # Compute min/max for each dimension
        self.traj_min = all_trajs.view(-1, all_trajs.shape[-1]).min(dim=0)[0]
        self.traj_max = all_trajs.view(-1, all_trajs.shape[-1]).max(dim=0)[0]
        self.force_min = all_forces.view(-1, all_forces.shape[-1]).min(dim=0)[0]
        self.force_max = all_forces.view(-1, all_forces.shape[-1]).max(dim=0)[0]
    
    def _normalize_data(self):
        """Normalize trajectory and force data to [-1, 1]."""
        for i in range(len(self.trajectories)):
            # Normalize trajectory
            self.trajectories[i] = 2 * (self.trajectories[i] - self.traj_min) / (self.traj_max - self.traj_min + 1e-8) - 1
            # Normalize force
            self.forces[i] = 2 * (self.forces[i] - self.force_min) / (self.force_max - self.force_min + 1e-8) - 1
    
    def denormalize_trajectory(self, traj: torch.Tensor) -> torch.Tensor:
        """Denormalize trajectory from [-1, 1] back to original scale."""
        if not self.normalize:
            return traj
        return (traj + 1) / 2 * (self.traj_max - self.traj_min + 1e-8) + self.traj_min
    
    def denormalize_force(self, force: torch.Tensor) -> torch.Tensor:
        """Denormalize force from [-1, 1] back to original scale."""
        if not self.normalize:
            return force
        return (force + 1) / 2 * (self.force_max - self.force_min + 1e-8) + self.force_min
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        """
        Get a single demonstration.
        
        Returns:
            trajectory: (T, 3) tensor
            force: (T, 3) tensor
            img_features: (T, D) tensor
        """
        return self.trajectories[idx], self.forces[idx], self.img_features[idx]
    
    def get_all_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get all data as tensors.
        
        Returns:
            trajectories: (N, T, 3) tensor
            forces: (N, T, 3) tensor
            img_features: (N, T, D) tensor
        """
        return (
            torch.stack(self.trajectories),
            torch.stack(self.forces),
            torch.stack(self.img_features)
        )


def create_feature_extractor(model_name: str = 'mobilenet_v2', device: str = 'cpu') -> torch.nn.Module:
    """
    Create a pre-trained feature extractor for images.
    
    Args:
        model_name: Name of the model ('mobilenet_v2', 'resnet18', 'resnet50')
        device: Device to load the model on
    
    Returns:
        Feature extractor model
    """
    from torchvision import models
    
    if model_name == 'mobilenet_v2':
        try:
            # Try new API first (torchvision >= 0.13)
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        except (AttributeError, TypeError):
            # Fall back to old API for compatibility
            model = models.mobilenet_v2(pretrained=True)
        model.classifier = torch.nn.Identity()
    elif model_name == 'resnet18':
        try:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except (AttributeError, TypeError):
            model = models.resnet18(pretrained=True)
        model.fc = torch.nn.Identity()
    elif model_name == 'resnet50':
        try:
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        except (AttributeError, TypeError):
            model = models.resnet50(pretrained=True)
        model.fc = torch.nn.Identity()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model.to(device)
    model.eval()
    return model
