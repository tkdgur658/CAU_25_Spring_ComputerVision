import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import os
from torchvision.datasets import VOCDetection
from PIL import Image
import glob
import xml.etree.ElementTree as ET
import random
import shutil
import yaml
import cv2
import matplotlib.pyplot as plt
import sys
import io
import seaborn as sns
from sklearn.metrics import confusion_matrix

def visualize_voc_sample(dataset, sample_idx=0, figsize=(6, 4), show_info=True):
    """
    Visualize a sample from VOCDetection dataset with bounding boxes and labels.
    
    Args:
        dataset: VOCDetection dataset object
        sample_idx: Index of the sample to visualize (default: 0)
        figsize: Figure size as (width, height) tuple (default: (12, 9))
        show_info: Whether to print additional information about the sample (default: True)
        
    Returns:
        fig, ax: The matplotlib figure and axis objects
    """
    # Define VOC color map for different classes
    VOC_COLORMAP = {
        'person': 'red',
        'dog': 'blue',
        'cat': 'green',
        'car': 'yellow',
        'bicycle': 'purple',
        'boat': 'orange',
        'bird': 'cyan',
        'chair': 'magenta',
        'bottle': 'brown',
        'sofa': 'pink',
        'bus': 'lime',
        'train': 'coral',
        'motorbike': 'gold',
        'aeroplane': 'violet',
        'tvmonitor': 'teal',
        'horse': 'salmon',
        'cow': 'sienna',
        'sheep': 'olivedrab',
        'diningtable': 'darkturquoise',
        'pottedplant': 'indigo'
    }
    DEFAULT_COLOR = 'gray'  # Default color for classes not in the map
    
    # Get a single sample
    img, annotation = dataset[sample_idx]
    
    # Print annotation structure if show_info is True
    if show_info:
        print("Annotation structure:")
        print(annotation)
    
    # Get bounding boxes and object names
    objects = annotation['annotation']['object']
    if not isinstance(objects, list):
        objects = [objects]  # Handle case where there's only one object
    
    # Create figure and axis
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(img)
    
    # Draw bounding boxes
    for obj in objects:
        obj_name = obj['name']
        bbox = obj['bndbox']
        xmin = float(bbox['xmin'])
        ymin = float(bbox['ymin'])
        xmax = float(bbox['xmax'])
        ymax = float(bbox['ymax'])
        
        # Calculate width and height of box
        width = xmax - xmin
        height = ymax - ymin
        
        # Get color for this class
        color = VOC_COLORMAP.get(obj_name, DEFAULT_COLOR)
        
        # Create rectangle patch
        rect = patches.Rectangle(
            (xmin, ymin), width, height, 
            linewidth=2, edgecolor=color, facecolor='none'
        )
        
        # Add rectangle to plot
        ax.add_patch(rect)
        
        # Add label
        plt.text(
            xmin, ymin-5, 
            obj_name, 
            color='white', 
            fontsize=12, 
            bbox=dict(facecolor=color, alpha=0.7)
        )
    
    plt.title(f"VOC 2007 Sample {sample_idx}: {annotation['annotation']['filename']}")
    plt.axis('off')
    plt.tight_layout()
    
    # Print additional information about the image if show_info is True
    if show_info:
        print(f"Image size: {img.size}")
        print(f"Image filename: {annotation['annotation']['filename']}")
        print(f"Image folder: {annotation['annotation']['folder']}")
        print(f"Total objects in this image: {len(objects)}")
    
    plt.show()  # 직접 화면에 표시
    return fig, ax


voc_classes = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


# Function to convert VOC XML to YOLO text
def convert_voc_to_yolo(xml_file, output_dir):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)
    
    filename = root.find("filename").text
    base_name = os.path.splitext(filename)[0]
    
    yolo_lines = []
    
    for obj in root.findall("object"):
        # Get class name and ID
        class_name = obj.find("name").text
        if class_name not in voc_classes:
            continue
            
        class_id = voc_classes.index(class_name)
        
        # Check difficult flag (optional)
        difficult = obj.find("difficult")
        if difficult is not None and int(difficult.text) == 1:
            continue
        
        # Get bounding box coordinates
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)
        
        # Convert to YOLO format (center x, y, width, height) - all values between 0~1
        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height
        
        # Save in YOLO format
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
    
    # Save converted labels
    if yolo_lines:  # Only save if there are valid objects
        with open(os.path.join(output_dir, f"{base_name}.txt"), "w") as f:
            f.write("\n".join(yolo_lines))
        return True
    return False
    
def setup_yolo_dataset(yolo_dataset):
    voc_root = "VOCdevkit/VOC2007"
    voc_annotations = os.path.join(voc_root, "Annotations")
    voc_images = os.path.join(voc_root, "JPEGImages")
    # Create directories
    for split in ["train", "val", "test"]:
        for folder in ["images", "labels"]:
            os.makedirs(os.path.join(yolo_dataset, folder, split), exist_ok=True)
    
    # YOLO label save directory
    temp_labels_dir = os.path.join(yolo_dataset, "temp_labels")
    os.makedirs(temp_labels_dir, exist_ok=True)
    
    # Label conversion for all XML files
    xml_files = glob.glob(os.path.join(voc_annotations, "*.xml"))
    valid_images = []
    
    for xml_file in xml_files:
        base_name = os.path.splitext(os.path.basename(xml_file))[0]
        img_file = os.path.join(voc_images, f"{base_name}.jpg")
        
        # Only valid if image file exists and label conversion is successful
        if os.path.exists(img_file) and convert_voc_to_yolo(xml_file, temp_labels_dir):
            valid_images.append(base_name)
    
    print(f"Found {len(valid_images)} valid images and labels.")
    
    # train/val/test split (60/20/20)
    random.seed(42)
    random.shuffle(valid_images)
    train_split = int(len(valid_images) * 0.6)
    val_split = int(len(valid_images) * 0.8)
    
    train_images = valid_images[:train_split]
    val_images = valid_images[train_split:val_split]
    test_images = valid_images[val_split:]
    
    print(f"Training: {len(train_images)} images ({len(train_images)/len(valid_images)*100:.1f}%)")
    print(f"Validation: {len(val_images)} images ({len(val_images)/len(valid_images)*100:.1f}%)")
    print(f"Testing: {len(test_images)} images ({len(test_images)/len(valid_images)*100:.1f}%)")
    
    # Copy image and label files
    for img_set, subset in [(train_images, "train"), (val_images, "val"), (test_images, "test")]:
        for img_name in img_set:
            # Copy image
            src_img = os.path.join(voc_images, f"{img_name}.jpg")
            dst_img = os.path.join(yolo_dataset, "images", subset, f"{img_name}.jpg")
            shutil.copy(src_img, dst_img)
            
            # Copy label
            src_label = os.path.join(temp_labels_dir, f"{img_name}.txt")
            dst_label = os.path.join(yolo_dataset, "labels", subset, f"{img_name}.txt")
            if os.path.exists(src_label):
                shutil.copy(src_label, dst_label)
    
    # Delete temporary label directory
    shutil.rmtree(temp_labels_dir)
    
    # Create dataset.yaml file for YOLOv5 training
    yaml_content = {
        'path': os.path.abspath(yolo_dataset),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(voc_classes),
        'names': voc_classes
    }
    
    with open(os.path.join(yolo_dataset, 'dataset.yaml'), 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"Created dataset.yaml in {yolo_dataset}")

def run_individual_inference(trained_model, test_image_dir, output_dir, conf=0.5):
    """
    주어진 테스트 이미지 디렉토리의 모든 이미지에 대해 모델 추론을 수행하고, 결과를 output_dir에 저장합니다.
    
    Args:
        trained_model: 추론에 사용할 YOLO 모델 객체.
        test_image_dir (str): 테스트 이미지가 저장된 디렉토리 경로.
        output_dir (str): 추론 결과를 저장할 디렉토리 경로.
        conf (float, optional): 추론 시 사용할 신뢰도 임계값. 기본값은 0.5.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    original_stdout = sys.stdout
    sys.stdout = io.StringIO()  # 불필요한 출력 숨기기
    
    for img_file in os.listdir(test_image_dir):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(test_image_dir, img_file)
            results = trained_model(img_path, conf=conf, verbose=False)
            
            for r in results:
                output_path = os.path.join(output_dir, img_file)
                r.save(filename=output_path)
    
    sys.stdout = original_stdout

def plot_confusion_matrix(test_results, output_dir=None, class_names=voc_classes):
    """
    Function to visualize the confusion matrix.
    
    Args:
        test_results: The validation results object from the YOLO model.
        output_dir: Directory to save the resulting images (default: None).
        class_names: Optional list of class names to use for labeling (default: uses test_results.names)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    
    print("Generating confusion matrix...")
    
    # Use provided class names if given, otherwise fall back to test_results.names
    if class_names is None:
        class_names = test_results.names
    
    # Obtain confusion matrix data
    conf_matrix = test_results.confusion_matrix.matrix
    
    # Compute normalized confusion matrix
    conf_matrix_norm = conf_matrix / (conf_matrix.sum(axis=1, keepdims=True) + 1e-10)
    
    # Create plot for normalized confusion matrix
    plt.figure(figsize=(10,8))
    sns.heatmap(conf_matrix_norm, annot=True, cmap="Blues", fmt=".2f",
                xticklabels=class_names, yticklabels=class_names)
    
    plt.xlabel('True Class')
    plt.ylabel('Predicted Class')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    
    # Save the normalized confusion matrix if an output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        norm_path = os.path.join(output_dir, "confusion_matrix_norm.png")
        plt.savefig(norm_path, dpi=300)
        print(f"Normalized confusion matrix saved: {norm_path}")
    
    # Display the normalized confusion matrix inline
    plt.show()
    
    # Create plot for the raw confusion matrix (counts)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt=".0f",
                xticklabels=class_names, yticklabels=class_names)
    
    plt.xlabel('True Class')
    plt.ylabel('Predicted Class')
    plt.title('Confusion Matrix (Raw Counts)')
    plt.tight_layout()
    
    # Save the raw confusion matrix if an output directory is provided
    if output_dir:
        raw_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(raw_path, dpi=300)
        print(f"Raw confusion matrix saved: {raw_path}")
    
    # Display the raw confusion matrix inline
    plt.show()
    
    # Calculate class-wise precision and recall
    recall = np.diag(conf_matrix) / (conf_matrix.sum(axis=0) + 1e-10)
    precision = np.diag(conf_matrix) / (conf_matrix.sum(axis=1) + 1e-10)
    
    # Print class-wise performance metrics
    print("\nClass-wise performance metrics:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}")
    
    return conf_matrix, conf_matrix_norm

