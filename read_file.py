from data_classes import Batch
import pandas as pd
import os

def read_csv_file(annotations_file: str, batch: Batch) -> Batch:
    """
    Fill the ImageLabels Object in the batch with the annotations from a csv file
    :param annotations_file: path to csv file
    :param batch: batch with all images name
    :return: list of images and their corresponding annotations 'image_name' : '[(row1, col1, label1), (row2, col2, label2), ...]'
    """
    # 'image_name' : '[(row1, col1, label1), (row2, col2, label2), ...]'
    df = pd.read_csv(annotations_file)

    for index, row in df.iterrows():
        image_name = row['Name']
        row_image = int(row['Row'])
        column_image = int(row['Column'])
        label_image = str(row['Label'])

        # Check if images already in the batch
        #if image_name not in batch:
        #print('No image {}'.format(image_name))
        image_label = batch.get_image_labels(image_name)
        if image_label is not None:
            image_label.add_annotations((row_image, column_image, label_image))

    return batch


def read_csv_file_from_fraction(annotations_file:str, batch: Batch, image_width, image_height) -> Batch:
    """
    Fill the ImageLabels Object in the batch with the annotations from a csv file
    :param annotations_file: path to csv file
    :param batch: batch with all images name
    :return: list of images and their corresponding annotations 'image_name' : '[(row1, col1, label1), (row2, col2, label2), ...]'
    """
    df = pd.read_csv(annotations_file)
    for index, row in df.iterrows():
        if row['label_number'] not in [1,2,3,4,5]:
            image_name = row['left_image_name'] + '.png'
            fraction_from_left = row['fraction_from_image_left']
            fraction_from_top = row['fraction_from_image_top']
            row_image = from_fraction_to_pixel(fraction_from_left, image_width)
            column_image = from_fraction_to_pixel(fraction_from_top, image_height)
            label_image = row['cpc_label']

            image_label = batch.get_image_labels(image_name)

            if image_label is not None:
                image_label.add_annotations((row_image, column_image, label_image))
    return batch


def from_fraction_to_pixel(fraction, size):
    return int(fraction * size)


def read_txt_file(input_folder, txt_file):
    points_labels = []

    # Combine the input folder and file name to get the full path
    file_path = os.path.join(input_folder, txt_file)

    try:
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('#'):
                    continue
                parts = line.split(';')
                row = int(parts[0].strip())
                col = int(parts[1].strip())
                label = parts[2].strip()
                points_labels.append((row, col, label))
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    except Exception as e:
        raise Exception(f"An error occurred while reading the file: {e}")

    return points_labels
