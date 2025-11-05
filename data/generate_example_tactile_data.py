"""
Generate synthetic tactile data for testing the tactile data loader and training scripts.

This script creates example data with the expected structure:
- data_folder/demo_X/img/XXX.jpg (tactile images)
- data_folder/demo_X/data.csv (trajectory and force data)
"""

import numpy as np
import os
from PIL import Image
import csv


def generate_synthetic_tactile_data(
    output_folder='./example_tactile_data',
    num_demos=10,
    num_timesteps=200,
    img_size=(224, 224)
):
    """
    Generate synthetic tactile dataset for testing.
    
    Args:
        output_folder: Root folder to save generated data
        num_demos: Number of demonstration trajectories
        num_timesteps: Number of timesteps per demonstration
        img_size: Size of generated images (width, height)
    """
    
    print(f"Generating synthetic tactile data...")
    print(f"  Output folder: {output_folder}")
    print(f"  Number of demos: {num_demos}")
    print(f"  Timesteps per demo: {num_timesteps}")
    
    os.makedirs(output_folder, exist_ok=True)
    
    for demo_idx in range(num_demos):
        demo_folder = os.path.join(output_folder, f'demo_{demo_idx}')
        img_folder = os.path.join(demo_folder, 'img')
        os.makedirs(img_folder, exist_ok=True)
        
        print(f"\nGenerating demo {demo_idx}...")
        
        # Generate trajectory (circular or linear motion with noise)
        t = np.linspace(0, 2*np.pi, num_timesteps)
        
        # Different motion patterns for different demos
        if demo_idx % 3 == 0:
            # Circular motion
            radius = 0.3 + np.random.rand() * 0.2
            x = 0.5 + radius * np.cos(t)
            y = 0.5 + radius * np.sin(t)
            z = 0.3 + 0.1 * np.sin(2*t)
        elif demo_idx % 3 == 1:
            # Linear motion with oscillation
            x = np.linspace(0.2, 0.8, num_timesteps)
            y = 0.5 + 0.1 * np.sin(4*t)
            z = 0.3 + 0.05 * np.cos(3*t)
        else:
            # Figure-8 motion
            scale = 0.2 + np.random.rand() * 0.1
            x = 0.5 + scale * np.sin(t)
            y = 0.5 + scale * np.sin(2*t)
            z = 0.3 + 0.05 * np.cos(2*t)
        
        # Add noise
        x += np.random.randn(num_timesteps) * 0.01
        y += np.random.randn(num_timesteps) * 0.01
        z += np.random.randn(num_timesteps) * 0.01
        
        # Generate forces (correlated with trajectory derivatives)
        dx = np.gradient(x)
        dy = np.gradient(y)
        dz = np.gradient(z)
        
        # Force proportional to acceleration + noise
        fx = -np.gradient(dx) * 10 + np.random.randn(num_timesteps) * 0.5
        fy = -np.gradient(dy) * 10 + np.random.randn(num_timesteps) * 0.5
        fz = -np.gradient(dz) * 10 + np.random.randn(num_timesteps) * 0.5 - 2.0  # gravity
        
        # Save CSV file
        csv_path = os.path.join(demo_folder, 'data.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y', 'z', 'fx', 'fy', 'fz'])
            for i in range(num_timesteps):
                writer.writerow([x[i], y[i], z[i], fx[i], fy[i], fz[i]])
        
        # Generate synthetic tactile images
        # Images simulate tactile sensor data with patterns that change over time
        for img_idx in range(num_timesteps):
            # Create image with patterns based on position and force
            img_array = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
            
            # Background gradient based on position
            for row in range(img_size[1]):
                for col in range(img_size[0]):
                    # Color based on position and force
                    r = int(128 + 127 * np.sin(x[img_idx] * np.pi + row/img_size[1]))
                    g = int(128 + 127 * np.sin(y[img_idx] * np.pi + col/img_size[0]))
                    b = int(128 + 127 * np.cos(z[img_idx] * 2 * np.pi))
                    img_array[row, col] = [r, g, b]
            
            # Add force-dependent features (bright spots)
            force_magnitude = np.sqrt(fx[img_idx]**2 + fy[img_idx]**2 + fz[img_idx]**2)
            num_spots = max(1, int(force_magnitude / 2))
            
            for _ in range(num_spots):
                spot_x = np.random.randint(20, img_size[0] - 20)
                spot_y = np.random.randint(20, img_size[1] - 20)
                spot_radius = np.random.randint(5, 15)
                
                for dy_offset in range(-spot_radius, spot_radius):
                    for dx_offset in range(-spot_radius, spot_radius):
                        if dx_offset**2 + dy_offset**2 <= spot_radius**2:
                            py = np.clip(spot_y + dy_offset, 0, img_size[1] - 1)
                            px = np.clip(spot_x + dx_offset, 0, img_size[0] - 1)
                            intensity = 255 - int(255 * (dx_offset**2 + dy_offset**2) / spot_radius**2)
                            img_array[py, px] = [intensity, intensity, intensity]
            
            # Save image
            img = Image.fromarray(img_array)
            img_path = os.path.join(img_folder, f'{img_idx:03d}.jpg')
            img.save(img_path, quality=95)
        
        print(f"  Generated {num_timesteps} images and CSV file")
    
    print(f"\n✓ Successfully generated {num_demos} demonstrations")
    print(f"✓ Data saved to: {output_folder}")
    print(f"\nYou can now use this data with the training scripts:")
    print(f"  1. Edit train_cnep_on_tactile.py or train_cnmp_on_tactile.py")
    print(f"  2. Set data_folder = '{output_folder}'")
    print(f"  3. Set demo_folders = ['demo_{i}' for i in range({num_demos})]")
    print(f"  4. Run the training script")


if __name__ == '__main__':
    # Generate example data
    generate_synthetic_tactile_data(
        output_folder='./example_tactile_data',
        num_demos=20,
        num_timesteps=200,
        img_size=(224, 224)
    )
