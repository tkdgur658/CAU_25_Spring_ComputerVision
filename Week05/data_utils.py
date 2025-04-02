import os
import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
# import yaml # PyYAML installation needed if you parse data.yaml: pip install PyYAML
import re # Added for using regular expressions
import matplotlib.pyplot as plt

# --- 1. Custom Dataset Class ---
class YOLOv8Dataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform

        # Get list of image files (including case-insensitive extensions)
        img_patterns = ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG']
        self.img_files = []
        for pattern in img_patterns:
            self.img_files.extend(glob.glob(os.path.join(img_dir, pattern)))
        self.img_files = sorted(list(set(self.img_files))) # Remove duplicates and sort

        print(f"Debug: Found {len(self.img_files)} potential image files in {img_dir}")
        if not self.img_files:
             print(f"Debug: No image files found in {img_dir} with patterns {img_patterns}")
             self.label_map = {}
             self.valid_img_files = []
             self.valid_label_files = []
             return # No need to proceed if no images found

        # Extract base names and paths for label files (considering Roboflow format)
        self.label_map = {} # Maps image core_name to label path
        raw_label_files = glob.glob(os.path.join(label_dir, '*.txt'))
        print(f"Debug: Found {len(raw_label_files)} potential label files in {label_dir}")

        for label_path in raw_label_files:
            base_name_with_ext = os.path.basename(label_path)
            base_name = os.path.splitext(base_name_with_ext)[0] # Remove .txt extension
            # Attempt to remove Roboflow hash (e.g., .rf.xxxxxxxx...)
            core_name = re.sub(r'\.rf\.[a-f0-9]+$', '', base_name)
            # Attempt to remove original extension if included in the name (e.g., _jpg)
            core_name = re.sub(r'(_jpg|_png|_jpeg)$', '', core_name, flags=re.IGNORECASE)

            self.label_map[core_name] = label_path # Use core_name as the key for matching

        if not self.label_map:
             print(f"Debug: No label files found or processed in {label_dir}")

        # Match image files with label files (considering Roboflow format)
        self.valid_img_files = []
        self.valid_label_files = [] # Store paths of matched labels
        matched_count = 0
        unmatched_img_examples = []

        for img_path in self.img_files:
            img_base_name_with_ext = os.path.basename(img_path)
            img_base_name = os.path.splitext(img_base_name_with_ext)[0] # Remove extension

            # Attempt to remove hash and potential included extension parts from image name
            img_core_name = re.sub(r'\.rf\.[a-f0-9]+$', '', img_base_name)
            img_core_name = re.sub(r'(_jpg|_png|_jpeg)$', '', img_core_name, flags=re.IGNORECASE)

            # Find the matching core_name in the label map
            if img_core_name in self.label_map:
                self.valid_img_files.append(img_path)
                self.valid_label_files.append(self.label_map[img_core_name]) # Add matched label path
                matched_count += 1
            else:
                if len(unmatched_img_examples) < 5: # Store up to 5 examples of unmatched images
                    unmatched_img_examples.append((img_path, img_core_name))

        print(f"Debug: Matched {matched_count} image-label pairs.")

        if not self.valid_img_files:
             print(f"Warning: Could not find matching label files for images. Check file naming conventions.")
             # Print examples for debugging
             if unmatched_img_examples:
                 print("  Unmatched image examples (path, derived core_name):")
                 for img_ex_path, img_ex_core in unmatched_img_examples:
                      print(f"    - {img_ex_path}, '{img_ex_core}'")
             if self.label_map:
                 example_label_key = list(self.label_map.keys())[0]
                 print(f"  Example label core_name derived: '{example_label_key}' (from '{self.label_map[example_label_key]}')")

        # self.img_files now only contains valid files with matched labels
        self.img_files = self.valid_img_files

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        if idx >= len(self.img_files):
             raise IndexError("Index out of range")

        img_path = self.img_files[idx]
        # Use the label file path that was already matched in __init__
        label_path = self.valid_label_files[idx]

        # Load image (convert to RGB)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error: Failed to load image '{img_path}': {e}")
            return None # To be handled by collate_fn

        # Load labels
        labels = []
        if label_path and os.path.exists(label_path):
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            try:
                                # Class index and coordinates (convert all to float)
                                class_idx = float(parts[0])
                                coords = [float(p) for p in parts[1:]]
                                labels.append([class_idx] + coords)
                            except ValueError:
                                print(f"Warning: Error converting label line to numbers in '{label_path}': '{line.strip()}'")
                        # else: # Can produce too many warnings
                        #     print(f"Warning: Incorrect format in label file '{label_path}': '{line.strip()}'")
            except Exception as e:
                print(f"Error: Failed to read label file '{label_path}': {e}")
                labels = []

        # Convert list to NumPy array then to Tensor
        # shape: (num_objects, 5)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        if not labels: # If label file was empty or had errors
             labels_tensor = torch.empty((0, 5), dtype=torch.float32)

        # Apply image transformations (transform)
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                 print(f"Error: Failed to transform image '{img_path}': {e}")
                 return None # To be handled by collate_fn

        # Final format to be passed to DataLoader (tuple or dictionary)
        return image, labels_tensor

# --- 2. Collate Function for DataLoader (Same as before) ---
def yolo_collate_fn(batch):
    # Filter out samples where __getitem__ returned None
    batch = [item for item in batch if item is not None]
    if not batch:
        # If all items were None, return empty batch or raise exception
        print("Warning: No valid samples to process in collate_fn.")
        return None, None # Or return torch.empty(0), []

    try:
        # Stack images into a batch tensor
        images = torch.stack([item[0] for item in batch], 0)
        # Keep labels as a list (because they can have different numbers of objects)
        labels = [item[1] for item in batch]
        return images, labels
    except RuntimeError as e:
        # Error likely occurs if image tensors in the batch have different shapes
        print(f"Error: Runtime error during batch collation: {e}")
        print("Individual image tensor shapes:")
        for i, item in enumerate(batch):
            if hasattr(item[0], 'shape'):
                print(f"  Item {i} image shape: {item[0].shape}")
            else:
                print(f"  Item {i} image is not a tensor (type: {type(item[0])})")
        # Return None or raise exception if problem persists
        return None, None


# Function to visualize a batch with bounding boxes
def visualize_batch(images, labels, num_images=4, figsize=(15, 15)):
    """
    Visualize a batch of images with their bounding boxes.
    
    Args:
        images: Tensor of shape [batch_size, channels, height, width]
        labels: List of tensors, each with shape [num_objects, 5] where each row is [class, x, y, w, h]
        num_images: Number of images to display
        figsize: Figure size
    """
    batch_size = images.shape[0]
    num_to_show = min(batch_size, num_images)
    
    fig, axes = plt.subplots(1, num_to_show, figsize=figsize)
    if num_to_show == 1:
        axes = [axes]
    
    # Define colors for different classes
    colors = plt.cm.hsv(np.linspace(0, 1, 80)).tolist()  # 80 different colors
    
    for i in range(num_to_show):
        # Convert tensor to numpy and transpose from CHW to HWC
        img = images[i].permute(1, 2, 0).cpu().numpy()
        
        # Normalize image for display if needed
        if img.max() <= 1:
            img = (img * 255).astype(np.uint8)
        
        # Display image
        axes[i].imshow(img)
        
        # Get bounding boxes for this image
        label = labels[i]
        
        # Draw bounding boxes
        for box in label:
            cls_id, x_center, y_center, width, height = box
            
            # Convert YOLO format (x_center, y_center, width, height) to (xmin, ymin, width, height)
            xmin = int((x_center - width/2) * img.shape[1])
            ymin = int((y_center - height/2) * img.shape[0])
            box_width = int(width * img.shape[1])
            box_height = int(height * img.shape[0])
            
            # Get color for this class
            color = colors[int(cls_id) % len(colors)]
            
            # Draw rectangle
            rect = plt.Rectangle((xmin, ymin), box_width, box_height, 
                                 fill=False, edgecolor=color, linewidth=2)
            axes[i].add_patch(rect)
            
            # Optionally add class label
            axes[i].text(xmin, ymin, f'Class {int(cls_id)}', 
                        bbox=dict(facecolor=color, alpha=0.5), fontsize=8)
            
        axes[i].set_title(f"Image {i+1}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()