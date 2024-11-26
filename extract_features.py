from feature_extractor import FeatureExtractor
from typing import Optional
from data_classes import Batch, ImageLabels
from read_file import read_csv_file, read_csv_file_from_fraction, read_txt_file
import os

def extract_all_features(dataset, path_images, annotations_file, CROP_SIZE, model_name, weights_file):
    if (type(annotations_file) or type (path_images)) != str:
        raise TypeError("Expected a string for the annotations file or the image path")
    if dataset == 'RIO':
        full_batch = create_full_batch_RIO(path_images, annotations_file)
    elif dataset == 'MLC':
        full_batch = create_full_batch_MLC(path_images)
    elif dataset == 'TasCPC':
        full_batch = create_full_batch_TASCPC(path_images, annotations_file)
    extract_features(full_batch, CROP_SIZE, path_images, model_name, dataset, weights_file)

def extract_features(batch, crop_size, input_folder, model_name, dataset, weights_file: Optional):
    batch_size = batch.batch_size()
    if model_name == 'efficientnet-b0':
        feature_extractor = FeatureExtractor(model_name, weights_file=weights_file)
    else:
        feature_extractor = FeatureExtractor(model_name)

    features, labels = feature_extractor.get_features(batch, crop_size, input_folder, split=dataset)

    return features, labels

def create_full_batch_MLC(input_folder: str):
    batch = Batch()

    # Use os.walk to recursively traverse the directory and its subdirectories
    for root, dirs, files in os.walk(input_folder):
        # Select only files with extensions .JPG or .jpg
        image_files = [file for file in files if file.endswith(('.JPG', '.jpg'))]

        for file_name in image_files:
            # Create a full path for each image file
            relative_image_path = os.path.relpath(os.path.join(root, file_name), input_folder)
            print(relative_image_path)
            image_label = ImageLabels(relative_image_path)
            batch.add_image(image_label)

            # Create a full path for the corresponding .txt file for the image
            txt_file_name = file_name + '.txt'
            full_txt_path = os.path.join(root, txt_file_name)

            # Read and add annotations if the .txt file exists
            if os.path.exists(full_txt_path):
                points_labels = read_txt_file(root, txt_file_name)

                # Add all existing annotations from the .txt file
                for annotation in points_labels:
                    normalized_label = annotation[2].capitalize()
                    annotation = (annotation[0], annotation[1], normalized_label)
                    image_label.add_annotations(annotation)

                # If there are fewer than 200 annotations, add [0, 0, 'FALSE']
                num_annotations = len(points_labels)
                if num_annotations < 200:
                    num_missing = 200 - num_annotations
                    for _ in range(num_missing):
                        # Add the annotation [0, 0, 'FALSE']
                        image_label.add_annotations((0, 0, 'NULL'))

    return batch


def create_full_batch_RIO(input_folder: str, annotations_file: str):
    batch = Batch()
    files = [file for file in os.listdir(input_folder) if file.endswith(('.JPG', '.jpg'))]

    for file_name in files:
        image_label = ImageLabels(file_name)
        batch.add_image(image_label)

    batch = read_csv_file(annotations_file, batch)

    return batch

def create_full_batch_TASCPC(input_folder: str, annotations_file: str):
    batch = Batch()
    files = [
        file for file in os.listdir(input_folder)
        if file.endswith(('.JPG', '.jpg', '.png')) and not file.startswith('._')
    ]

    for file_name in files:

        image_label = ImageLabels(file_name)
        batch.add_image(image_label)

    batch = read_csv_file_from_fraction(annotations_file, batch, 1360, 1024)

    return batch
