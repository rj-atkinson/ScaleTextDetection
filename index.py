import os
import argparse
from ultralytics import YOLO
from lxml import etree
import shutil
import torch
import time

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 using VOC XML annotations.")
    parser.add_argument('--images', type=str, required=True, help='Path to the folder containing images.')
    parser.add_argument('--annotations', type=str, required=True, help='Path to the folder containing annotations (VOC XML format).')
    parser.add_argument('--dataset', type=str, required=True, help='Name for the YOLO dataset directory.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--model_name', type=str, default='yolo11n', help='YOLO model to use (e.g., yolov8n, yolov8s, etc.)')
    return parser.parse_args()

# Convert VOC XML to YOLO TXT format and create empty annotation files for images without annotations
def convert_voc_to_yolo(annotations_dir, images_dir, train_labels_dir, train_images_dir):
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]  # List of image files
    
    for image_filename in image_files:
        xml_file = os.path.join(annotations_dir, os.path.splitext(image_filename)[0] + '.xml')
        img_path = os.path.join(images_dir, image_filename)
        shutil.copy(img_path, os.path.join(train_images_dir, image_filename))  # Copy image to YOLO dataset folder

        label_file = os.path.join(train_labels_dir, os.path.splitext(image_filename)[0] + '.txt')
        
        if os.path.exists(xml_file):
            # Parse the XML file
            tree = etree.parse(xml_file)
            root = tree.getroot()

            with open(label_file, 'w') as f:
                for obj in root.findall('object'):
                    class_name = 'digits'
                    class_id = get_class_id(class_name)  # Define your own function for class mapping
                    
                    # Extract bounding box coordinates
                    bbox = obj.find('bndbox')
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)
                    
                    # Convert to YOLO format (normalized values)
                    img_width = int(root.find('size/width').text)
                    img_height = int(root.find('size/height').text)
                    x_center = (xmin + xmax) / 2.0 / img_width
                    y_center = (ymin + ymax) / 2.0 / img_height
                    width = (xmax - xmin) / img_width
                    height = (ymax - ymin) / img_height
                    
                    # Write the converted annotation
                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
        else:
            # Create an empty .txt file for background images
            open(label_file, 'w').close()

def get_class_id(class_name):
    # You need to define a mapping from class names to class IDs
    class_map = {'digits': 0}  # Replace with your classes
    return class_map[class_name]

# Train the YOLOv8 model
def train_yolo(model_name, yaml_file, epochs, batch_size):
    # Optimize for M1 Max, Metal Performance Shaders
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # Load the YOLO model and train
    model = YOLO(f'{model_name}.pt').to(device)  # Load a YOLOv8 model, e.g., yolov8n (nano), yolov8m (medium)
    model.train(data=yaml_file, epochs=epochs, batch=batch_size)

    # Save the trained model to a new file
    unix_time = int(time.time())
    file_name = f'models/scale_text_detection_{model_name}_{unix_time}'

    model.save(f'{file_name}.pt')
    """
    Note: better to use coreml folder and pipenv shell for proper
    dependency versions that have been validated by coremltools

    # Save .mlpackage version
    model.overrides['nms'] = True
    model.export(format="coreml")  # export the model to CoreML format
    """

if __name__ == '__main__':
    args = parse_args()

    yolo_data_dir = args.dataset

    # Create directories for YOLO dataset format
    train_images_dir = os.path.join("datasets", yolo_data_dir, "images/train")
    train_labels_dir = os.path.join("datasets", yolo_data_dir, "labels/train")
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)

    # Convert VOC to YOLO format and handle background images
    convert_voc_to_yolo(args.annotations, args.images, train_labels_dir, train_images_dir)

    # Create a data.yaml file for the YOLO training configuration
    data_yaml_path = 'yolo_dataset.yaml'  # The path for the YAML file to be passed to the model
    with open(data_yaml_path, 'w') as f:
        f.write(f"""
        path: {yolo_data_dir}
        train: images/train
        val: images/train  # You can split or provide a validation set separately
        nc: 1  # Replace with the number of your classes
        names: ['digits']  # Replace with your class names
        """)

    # Train YOLO model using the correct YAML file path
    train_yolo(args.model_name, data_yaml_path, args.epochs, args.batch_size)
