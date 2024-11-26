from typing import List, Tuple, Optional

import torch
import os
#import params as p

#label_to_line_number = p.label_to_line_number

class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __str__(self) -> str:
        return f'({self.x}, {self.y})'

    def from_tuple(self, coords: Tuple[int, int]) -> 'Point':
        self.x = coords[0]
        self.y = coords[1]
        return self

    def __eq__(self, other: 'Point') -> bool:
        return self.x == other.x and self.y == other.y


class Annotation:
    def __init__(self, point: Point, label: str):
        self.point = point
        self.label = label


class Patch:
    def __init__(self, point: Point, name: str, label: str):
        self.point = point
        self.name = name
        self.label = label
        #self.patch = patch
        self.feature = torch.tensor([])
        #self.label_id = label_to_line_number[label]

    def add_features(self, features: torch.Tensor):
        device = features.device.type
        self.feature = self.feature.to(device)
        self.feature = torch.cat((self.feature, features), dim=0)

    def get_patch(self) -> torch.Tensor:
        return self.patch

    def __str__(self) -> str:
        patch_info = f"Patch at ({self.point.x}, {self.point.y}) - Name: {self.name}, Label: {self.label},"
        if hasattr(self, 'patch'):
            return f"{patch_info} Patch: Exists, Features: {self.feature.shape}"
        else:
            return f"{patch_info} Patch: Does not exist, Features: {self.feature.shape}"


class ImageLabels:
    def __init__(self, image_name: str):
        self.image_name = image_name
        #[(row1, col1, label1), (row2, col2, label2), ...]
        self.annotations: List[Annotation] = []
        self.patches: List[Patch] = []
        self.features: List[torch.Tensor] = []

    def add_image(self, image_key: str, annotations: List[Tuple[int, int, str]]):
        points = [Point(x, y) for x, y, _ in annotations]
        labels = [label for _, _, label in annotations]
        annotations_objs = [Annotation(point, label) for point, label in zip(points, labels)]
        self.annotations = annotations_objs
        self.image_name = image_key

    def add_annotations(self, annotation: Tuple[int, int, str]):
        point = Point(annotation[0], annotation[1])
        label = annotation[2]
        annotation_obj = Annotation(point, label)
        self.annotations.append(annotation_obj)

    def add_features(self, features: List[torch.Tensor]):
        self.features = features

    def add_patch(self, patch: Patch):
        self.patches.append(patch)

    def count_annotations(self) -> int:
        return len(self.annotations)

    def get_label_at_point(self, point: Point) -> Optional[str]:
        for annotation in self.annotations:
            if annotation.point == point:
                return annotation.label
        return None

    def get_list_patches(self):
        return self.patches

    def get_points_per_image(self) -> Tuple[str, List[Point]]:
        return self.image_name, [annotation.point for annotation in self.annotations]

    def __str__(self):
        annotations_str = ', '.join([f"({ann.point.x}, {ann.point.y}, {ann.label})" for ann in self.annotations])
        return f"{self.image_name}: {annotations_str}"

    def count_labels(self):
        label_count = {}
        for annotation in self.annotations:
            label = annotation.label
            if label in label_count:
                label_count[label] += 1
            else:
                label_count[label] = 1
        #print(label_count)
        return label_count


class Batch:
    def __init__(self, image_labels_list: List[ImageLabels] = None):
        if image_labels_list is None:
            self.image_labels_list = []
        else:
            self.image_labels_list = image_labels_list

    def add_image(self, image_labels: ImageLabels):
        self.image_labels_list.append(image_labels)

    def get_image_labels(self, image_name: str) -> Optional[ImageLabels]:
        for image_labels in self.image_labels_list:
            if image_labels.image_name == image_name:
                return image_labels
        return None

    def get_image_names_wtht_jpg(self) -> List[str]:
        image_names = []
        for image in self.image_labels_list:
            name, _ = os.path.splitext(image.image_name)
            image_names.append(name)
        return image_names

    def __str__(self):
        return '\n'.join(str(image_labels) for image_labels in self.image_labels_list)

    def __contains__(self, image_name: str):
        return any(image_name == image_label.image_name for image_label in self.image_labels_list)

    def __len__(self):
        return len(self.image_labels_list)

    def batch_size(self) -> int:
        return len(self.image_labels_list)

    def label_count(self)-> int:
        total_count = 0
        for image_label in self.image_labels_list:
            total_count += image_label.count_annotations()
        return total_count

    def labels_count_per_image(self):
        label_count_per_image = {}
        for image_label in self.image_labels_list:
            label_count = image_label.count_labels()
            label_count_per_image[image_label.image_name] = label_count
        #print("LABEL COUNT PER IMAGE ", label_count_per_image)
        return label_count_per_image

    def count_labels_in_batch(self):
        total_label_count = {}
        images_with_label_count = {}
        for image_labels in self.image_labels_list:
            label_count = image_labels.count_labels()
            for label, count in label_count.items():
                if label in total_label_count:
                    total_label_count[label] += count
                    images_with_label_count[label] += 1
                else:
                    total_label_count[label] = count
                    images_with_label_count[label] = 1
        return total_label_count, images_with_label_count

    def __add__(self, other):
        new_image_labels_list = self.image_labels_list + other.image_labels_list
        return Batch(new_image_labels_list)


class CaseInsensitiveDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Convertir toutes les clés existantes en minuscules
        self._convert_keys()

    def _convert_keys(self):
        """Convertit toutes les clés du dictionnaire en minuscules."""
        for key in list(self.keys()):
            value = super().pop(key)
            super().__setitem__(key.lower(), value)

    def __getitem__(self, key):
        return super().__getitem__(key.lower())

    def __setitem__(self, key, value):
        super().__setitem__(key.lower(), value)

    def __delitem__(self, key):
        super().__delitem__(key.lower())

    def __contains__(self, key):
        return super().__contains__(key.lower())
