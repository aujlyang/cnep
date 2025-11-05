"""
Test script for tactile data loader.

This script tests the TactileDataset class with synthetic data.
Run this after generating example data with generate_example_tactile_data.py
"""

import sys
import torch
import os

# Add data folder to path
data_folder_path = '../data/'
if data_folder_path not in sys.path:
    sys.path.append(data_folder_path)

from tactile_data_loader import TactileDataset, create_feature_extractor


def test_tactile_data_loader():
    """Test the tactile data loader with synthetic data."""
    
    print("="*70)
    print("Testing TactileDataset")
    print("="*70)
    
    # Generate example data first if it doesn't exist
    data_folder = '../data/example_tactile_data'
    if not os.path.exists(data_folder):
        print(f"\nExample data not found at {data_folder}")
        print("Generating synthetic data...")
        sys.path.append('../data/')
        from generate_example_tactile_data import generate_synthetic_tactile_data
        generate_synthetic_tactile_data(
            output_folder=data_folder,
            num_demos=5,
            num_timesteps=50,  # Use fewer timesteps for quick testing
        )
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Test 1: Load data without feature extractor
    print("\n" + "-"*70)
    print("Test 1: Loading data without feature extractor")
    print("-"*70)
    
    try:
        dataset = TactileDataset(
            data_folder=data_folder,
            demo_folders=['demo_0', 'demo_1', 'demo_2'],
            img_subfolder='img',
            csv_filename='data.csv',
            trajectory_cols=[0, 1, 2],
            force_cols=[3, 4, 5],
            num_timesteps=50,
            normalize=True,
            img_feature_extractor=None,
            device=device
        )
        
        trajectories, forces, img_features = dataset.get_all_data()
        
        print(f"✓ Successfully loaded dataset")
        print(f"  Trajectories shape: {trajectories.shape}")
        print(f"  Forces shape: {forces.shape}")
        print(f"  Image features shape: {img_features.shape}")
        print(f"  Trajectory range: [{trajectories.min():.3f}, {trajectories.max():.3f}]")
        print(f"  Force range: [{forces.min():.3f}, {forces.max():.3f}]")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    # Test 2: Load data with MobileNetV2 feature extractor
    print("\n" + "-"*70)
    print("Test 2: Loading data with MobileNetV2 feature extractor")
    print("-"*70)
    
    try:
        feature_extractor = create_feature_extractor('mobilenet_v2', device=device)
        
        dataset = TactileDataset(
            data_folder=data_folder,
            demo_folders=['demo_0', 'demo_1'],
            img_subfolder='img',
            csv_filename='data.csv',
            trajectory_cols=[0, 1, 2],
            force_cols=[3, 4, 5],
            num_timesteps=30,
            normalize=True,
            img_feature_extractor=feature_extractor,
            device=device
        )
        
        trajectories, forces, img_features = dataset.get_all_data()
        
        print(f"✓ Successfully loaded dataset with feature extractor")
        print(f"  Trajectories shape: {trajectories.shape}")
        print(f"  Forces shape: {forces.shape}")
        print(f"  Image features shape: {img_features.shape}")
        print(f"  Feature dimension: {img_features.shape[-1]}")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    # Test 3: Test denormalization
    print("\n" + "-"*70)
    print("Test 3: Testing normalization/denormalization")
    print("-"*70)
    
    try:
        # Get normalized data
        traj_normalized, _, _ = dataset.get_all_data()
        
        # Denormalize
        traj_denormalized = dataset.denormalize_trajectory(traj_normalized)
        
        print(f"✓ Denormalization successful")
        print(f"  Normalized range: [{traj_normalized.min():.3f}, {traj_normalized.max():.3f}]")
        print(f"  Denormalized range: [{traj_denormalized.min():.3f}, {traj_denormalized.max():.3f}]")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    # Test 4: Test individual item access
    print("\n" + "-"*70)
    print("Test 4: Testing individual item access")
    print("-"*70)
    
    try:
        traj, force, img_feat = dataset[0]
        
        print(f"✓ Individual item access successful")
        print(f"  Single trajectory shape: {traj.shape}")
        print(f"  Single force shape: {force.shape}")
        print(f"  Single image feature shape: {img_feat.shape}")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    print("\n" + "="*70)
    print("All tests passed! ✓")
    print("="*70)
    
    return True


if __name__ == '__main__':
    success = test_tactile_data_loader()
    
    if success:
        print("\n✓ TactileDataset is working correctly!")
        print("\nNext steps:")
        print("  1. Generate more data or use your own dataset")
        print("  2. Run train_cnep_on_tactile.py or train_cnmp_on_tactile.py")
    else:
        print("\n✗ Tests failed. Please check the error messages above.")
