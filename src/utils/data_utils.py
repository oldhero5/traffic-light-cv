"""
Data utility functions for traffic perception system.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

import albumentations as A
import cv2
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)


def prepare_dataset(output_yaml: str) -> None:
    """
    Prepare dataset configuration file.

    This function creates a dataset configuration YAML file for training.

    Args:
        output_yaml: Path to output YAML file
    """
    logger.info(f"Preparing dataset configuration at {output_yaml}")

    # Get base directory
    base_dir = Path(output_yaml).parent.parent.resolve()
    data_dir = base_dir / "data"

    # Create paths
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    # Create basic dataset configuration
    config = {
        "path": str(data_dir),
        "train": str(train_dir),
        "val": str(val_dir),
        "test": str(test_dir),
        "nc": 7,  # Number of classes
        "names": [
            "traffic_light_unknown",
            "traffic_light_red",
            "traffic_light_yellow",
            "traffic_light_green",
            "traffic_camera",
            "speed_camera",
            "red_light_camera",
        ],
    }

    # Create directories if they don't exist
    os.makedirs(train_dir / "images", exist_ok=True)
    os.makedirs(train_dir / "labels", exist_ok=True)
    os.makedirs(val_dir / "images", exist_ok=True)
    os.makedirs(val_dir / "labels", exist_ok=True)
    os.makedirs(test_dir / "images", exist_ok=True)
    os.makedirs(test_dir / "labels", exist_ok=True)

    # Write configuration to YAML file
    os.makedirs(os.path.dirname(output_yaml), exist_ok=True)

    with open(output_yaml, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(f"Dataset configuration created at {output_yaml}")


def convert_annotations(
    input_file: str,
    output_dir: str,
    format_type: str = "yolo",
    image_dir: str | None = None,
) -> None:
    """
    Convert annotations to the specified format.

    Args:
        input_file: Path to input annotation file
        output_dir: Directory to save converted annotations
        format_type: Annotation format type ('yolo', 'coco', 'pascal')
        image_dir: Directory containing images (for validation)
    """
    logger.info(f"Converting annotations from {input_file} to {format_type} format")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load annotations
    annotations = load_annotations(input_file)

    if not annotations:
        logger.error("No annotations loaded")
        return

    # Process annotations
    if format_type.lower() == "yolo":
        convert_to_yolo(annotations, output_dir, image_dir)
    elif format_type.lower() == "coco":
        convert_to_coco(annotations, output_dir, image_dir)
    elif format_type.lower() == "pascal":
        convert_to_pascal(annotations, output_dir, image_dir)
    else:
        logger.error(f"Unsupported format type: {format_type}")
        return

    logger.info(f"Annotations converted to {format_type} format in {output_dir}")


def load_annotations(annotation_file: str) -> list[dict]:
    """
    Load annotations from file.

    Args:
        annotation_file: Path to annotation file

    Returns:
        List of annotation dictionaries
    """
    ext = os.path.splitext(annotation_file)[1].lower()

    try:
        if ext == ".json":
            with open(annotation_file) as f:
                data = json.load(f)

            # Handle different JSON formats
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                if "annotations" in data:
                    return data["annotations"]
                elif "images" in data:
                    # COCO format
                    return convert_coco_to_list(data)

            logger.error("Unsupported JSON format")
            return []

        elif ext == ".yaml" or ext == ".yml":
            with open(annotation_file) as f:
                data = yaml.safe_load(f)

            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "annotations" in data:
                return data["annotations"]

            logger.error("Unsupported YAML format")
            return []

        else:
            logger.error(f"Unsupported annotation file format: {ext}")
            return []

    except Exception as e:
        logger.error(f"Error loading annotations: {e}")
        return []


def convert_coco_to_list(coco_data: dict) -> list[dict]:
    """
    Convert COCO format annotations to a list of annotations.

    Args:
        coco_data: COCO format data

    Returns:
        List of annotation dictionaries
    """
    result = []

    # Create image lookup
    image_lookup = {img["id"]: img for img in coco_data.get("images", [])}

    # Create category lookup
    category_lookup = {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}

    # Process annotations
    for ann in coco_data.get("annotations", []):
        image_id = ann.get("image_id")
        category_id = ann.get("category_id")

        if image_id is None or category_id is None:
            continue

        image_info = image_lookup.get(image_id)
        category_name = category_lookup.get(category_id)

        if not image_info or not category_name:
            continue

        bbox = ann.get("bbox", [0, 0, 0, 0])

        # COCO format is [x, y, width, height]
        # Convert to [x1, y1, x2, y2]
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h

        result.append(
            {
                "image": image_info.get("file_name"),
                "image_id": image_id,
                "image_width": image_info.get("width"),
                "image_height": image_info.get("height"),
                "category": category_name,
                "category_id": category_id,
                "bbox": [x1, y1, x2, y2],
                "area": ann.get("area"),
                "iscrowd": ann.get("iscrowd", 0),
            }
        )

    return result


def convert_to_yolo(
    annotations: list[dict],
    output_dir: str,
    image_dir: str | None = None,
) -> None:
    """
    Convert annotations to YOLO format.

    Args:
        annotations: List of annotation dictionaries
        output_dir: Directory to save YOLO annotations
        image_dir: Directory containing images (for validation)
    """
    # Create class mapping
    class_names = set()
    for ann in annotations:
        if "category" in ann:
            class_names.add(ann["category"])

    class_mapping = {name: i for i, name in enumerate(sorted(class_names))}

    # Write class names file
    with open(os.path.join(output_dir, "classes.txt"), "w") as f:
        f.writelines(f"{name}\n" for name in sorted(class_names))

    # Group annotations by image
    image_annotations = {}
    for ann in annotations:
        image_name = ann.get("image")
        if not image_name:
            continue

        if image_name not in image_annotations:
            image_annotations[image_name] = []

        image_annotations[image_name].append(ann)

    # Process each image
    for image_name, anns in tqdm(image_annotations.items(), desc="Converting to YOLO"):
        # Use first annotation to get image dimensions
        img_width = anns[0].get("image_width")
        img_height = anns[0].get("image_height")

        # If image dimensions not provided, try to get from image file
        if not img_width or not img_height:
            if image_dir:
                img_path = os.path.join(image_dir, image_name)
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        img_height, img_width = img.shape[:2]

            # If still no dimensions, skip
            if not img_width or not img_height:
                logger.warning(f"Image dimensions not available for {image_name}, skipping")
                continue

        # Create YOLO annotation file
        basename = os.path.splitext(image_name)[0]
        yolo_path = os.path.join(output_dir, f"{basename}.txt")

        with open(yolo_path, "w") as f:
            for ann in anns:
                category = ann.get("category")
                bbox = ann.get("bbox")

                if not category or not bbox:
                    continue

                # Get class ID
                class_id = class_mapping.get(category, 0)

                # Convert bbox to YOLO format (x_center, y_center, width, height)
                # Input bbox is [x1, y1, x2, y2]
                x1, y1, x2, y2 = bbox
                x_center = (x1 + x2) / 2 / img_width
                y_center = (y1 + y2) / 2 / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height

                # Write to file
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


def convert_to_coco(
    annotations: list[dict],
    output_dir: str,
    image_dir: str | None = None,
) -> None:
    """
    Convert annotations to COCO format.

    Args:
        annotations: List of annotation dictionaries
        output_dir: Directory to save COCO annotations
        image_dir: Directory containing images (for validation)
    """
    # Create COCO structure
    coco_data = {
        "info": {
            "description": "Traffic Light and Camera Dataset",
            "version": "1.0",
            "year": 2023,
            "contributor": "Traffic Perception System",
            "date_created": "2023/01/01",
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": "",
            }
        ],
        "images": [],
        "annotations": [],
        "categories": [],
    }

    # Create class mapping
    class_names = set()
    for ann in annotations:
        if "category" in ann:
            class_names.add(ann["category"])

    # Create categories
    for i, name in enumerate(sorted(class_names)):
        coco_data["categories"].append(
            {
                "id": i + 1,  # COCO uses 1-indexed category IDs
                "name": name,
                "supercategory": "traffic",
            }
        )

    # Create class mapping
    class_mapping = {name: i + 1 for i, name in enumerate(sorted(class_names))}

    # Group annotations by image
    image_annotations = {}
    for ann in annotations:
        image_name = ann.get("image")
        if not image_name:
            continue

        if image_name not in image_annotations:
            image_annotations[image_name] = []

        image_annotations[image_name].append(ann)

    # Process each image
    image_id = 1
    annotation_id = 1

    for image_name, anns in tqdm(image_annotations.items(), desc="Converting to COCO"):
        # Use first annotation to get image dimensions
        img_width = anns[0].get("image_width")
        img_height = anns[0].get("image_height")

        # If image dimensions not provided, try to get from image file
        if not img_width or not img_height:
            if image_dir:
                img_path = os.path.join(image_dir, image_name)
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        img_height, img_width = img.shape[:2]

            # If still no dimensions, skip
            if not img_width or not img_height:
                logger.warning(f"Image dimensions not available for {image_name}, skipping")
                continue

        # Add image to COCO data
        coco_data["images"].append(
            {
                "id": image_id,
                "file_name": image_name,
                "width": img_width,
                "height": img_height,
                "license": 1,
            }
        )

        # Process annotations
        for ann in anns:
            category = ann.get("category")
            bbox = ann.get("bbox")

            if not category or not bbox:
                continue

            # Get category ID
            category_id = class_mapping.get(category, 1)

            # Convert bbox to COCO format [x, y, width, height]
            # Input bbox is [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1

            coco_data["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x1, y1, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                }
            )

            annotation_id += 1

        image_id += 1

    # Write COCO annotation file
    coco_path = os.path.join(output_dir, "coco_annotations.json")

    with open(coco_path, "w") as f:
        json.dump(coco_data, f, indent=2)


def convert_to_pascal(
    annotations: list[dict],
    output_dir: str,
    image_dir: str | None = None,
) -> None:
    """
    Convert annotations to Pascal VOC format.

    Args:
        annotations: List of annotation dictionaries
        output_dir: Directory to save Pascal VOC annotations
        image_dir: Directory containing images (for validation)
    """
    # Create output directory
    annotations_dir = os.path.join(output_dir, "Annotations")
    os.makedirs(annotations_dir, exist_ok=True)

    # Group annotations by image
    image_annotations = {}
    for ann in annotations:
        image_name = ann.get("image")
        if not image_name:
            continue

        if image_name not in image_annotations:
            image_annotations[image_name] = []

        image_annotations[image_name].append(ann)

    # Process each image
    for image_name, anns in tqdm(image_annotations.items(), desc="Converting to Pascal VOC"):
        # Use first annotation to get image dimensions
        img_width = anns[0].get("image_width")
        img_height = anns[0].get("image_height")

        # If image dimensions not provided, try to get from image file
        if not img_width or not img_height:
            if image_dir:
                img_path = os.path.join(image_dir, image_name)
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        img_height, img_width = img.shape[:2]

            # If still no dimensions, skip
            if not img_width or not img_height:
                logger.warning(f"Image dimensions not available for {image_name}, skipping")
                continue

        # Create XML content
        xml_content = f"""
        <annotation>
            <folder>images</folder>
            <filename>{image_name}</filename>
            <path>{image_name}</path>
            <source>
                <database>Unknown</database>
            </source>
            <size>
                <width>{img_width}</width>
                <height>{img_height}</height>
                <depth>3</depth>
            </size>
            <segmented>0</segmented>
        """

        # Add objects
        for ann in anns:
            category = ann.get("category")
            bbox = ann.get("bbox")

            if not category or not bbox:
                continue

            # Convert bbox to Pascal VOC format
            # Input bbox is [x1, y1, x2, y2]
            x1, y1, x2, y2 = [int(c) for c in bbox]

            xml_content += f"""
            <object>
                <name>{category}</name>
                <pose>Unspecified</pose>
                <truncated>0</truncated>
                <difficult>0</difficult>
                <bndbox>
                    <xmin>{x1}</xmin>
                    <ymin>{y1}</ymin>
                    <xmax>{x2}</xmax>
                    <ymax>{y2}</ymax>
                </bndbox>
            </object>
            """

        xml_content += """
        </annotation>
        """

        # Write XML file
        basename = os.path.splitext(image_name)[0]
        xml_path = os.path.join(annotations_dir, f"{basename}.xml")

        with open(xml_path, "w") as f:
            f.write(xml_content)


def augment_dataset(
    input_dir: str,
    output_dir: str,
    num_augmentations: int = 3,
) -> None:
    """
    Augment dataset with various transformations.

    Args:
        input_dir: Directory containing input images and labels
        output_dir: Directory to save augmented dataset
        num_augmentations: Number of augmentations per image
    """
    logger.info(f"Augmenting dataset from {input_dir} to {output_dir}")

    # Create directories
    images_input_dir = os.path.join(input_dir, "images")
    labels_input_dir = os.path.join(input_dir, "labels")

    images_output_dir = os.path.join(output_dir, "images")
    labels_output_dir = os.path.join(output_dir, "labels")

    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(labels_output_dir, exist_ok=True)

    # Define augmentation pipeline
    transform = A.Compose(
        [
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.5),
            A.CLAHE(p=0.5),
            A.HueSaturationValue(p=0.3),
            A.RGBShift(p=0.3),
            A.RandomFog(p=0.2),
            A.RandomRain(p=0.2),
            A.RandomShadow(p=0.2),
            A.RandomSnow(p=0.1),
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.GaussianBlur(blur_limit=3, p=0.1),
            A.GaussNoise(p=0.3),
            A.RandomSunFlare(p=0.1),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )

    # Get image files
    image_files = [
        f for f in os.listdir(images_input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # Copy original files
    for img_file in image_files:
        # Copy image
        src_img = os.path.join(images_input_dir, img_file)
        dst_img = os.path.join(images_output_dir, img_file)
        shutil.copy2(src_img, dst_img)

        # Copy corresponding label if exists
        label_file = os.path.splitext(img_file)[0] + ".txt"
        src_label = os.path.join(labels_input_dir, label_file)
        dst_label = os.path.join(labels_output_dir, label_file)

        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)

    # Augment images
    for img_file in tqdm(image_files, desc="Augmenting images"):
        # Load image
        img_path = os.path.join(images_input_dir, img_file)
        img = cv2.imread(img_path)

        if img is None:
            logger.warning(f"Failed to load image: {img_path}")
            continue

        # Load labels
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(labels_input_dir, label_file)

        bboxes = []
        class_labels = []

        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])

                        bboxes.append([x_center, y_center, width, height])
                        class_labels.append(class_id)

        # Generate augmentations
        for i in range(num_augmentations):
            # Skip augmentation if no bboxes
            if not bboxes:
                continue

            augmented = transform(image=img, bboxes=bboxes, class_labels=class_labels)

            aug_img = augmented["image"]
            aug_bboxes = augmented["bboxes"]
            aug_class_labels = augmented["class_labels"]

            # Save augmented image
            aug_img_file = (
                f"{os.path.splitext(img_file)[0]}_aug_{i + 1}{os.path.splitext(img_file)[1]}"
            )
            aug_img_path = os.path.join(images_output_dir, aug_img_file)
            cv2.imwrite(aug_img_path, aug_img)

            # Save augmented labels
            aug_label_file = f"{os.path.splitext(img_file)[0]}_aug_{i + 1}.txt"
            aug_label_path = os.path.join(labels_output_dir, aug_label_file)

            with open(aug_label_path, "w") as f:
                for j, bbox in enumerate(aug_bboxes):
                    class_id = aug_class_labels[j]
                    x_center, y_center, width, height = bbox
                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

    logger.info(
        f"Dataset augmentation complete. Augmented {len(image_files)} images with {num_augmentations} variations each."
    )
