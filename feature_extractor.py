import os
import torch
from crop_images import crop_images_in_patches
import timm
import pickle
import sys
from collections import OrderedDict
from io import BytesIO
from typing import Any, Optional
import models
from data_classes import Batch

# import sys

# def get_list_memory_size(lst):
#     total_memory = sys.getsizeof(lst)  # Memory of the list structure itself
#     for item in lst:
#         total_memory += sys.getsizeof(item)  # Add memory of each item
#     return total_memory / (1024**3)

def get_tensors_memory_size(tensors_list):
    total_memory = 0
    for tensor in tensors_list:
        # Get the size of a single element in bytes
        element_size = tensor.element_size()
        # Multiply by the number of elements to get the total size of the tensor
        tensor_size = tensor.numel() * element_size
        # Add to the total memory
        total_memory += tensor_size

    return total_memory / (1024**3)


def get_lists_memory_size(lists_list):
    total_memory = 0
    for item in lists_list:
        # Get the size of the item in bytes
        item_size = sys.getsizeof(item)
        # Add to the total memory
        total_memory += item_size

    return total_memory / (1024**3)


class FeatureExtractor:
    def __init__(self, model_name, weights_file=None, save_path="./.features/"):
        self.model_name = model_name
        if model_name not in timm.list_models(pretrained=True) and model_name != 'efficientnet-b0' and model_name != 'benthicnet-vit':
            print("Error, '%s' is not a valid model name for timm library. For a list of available pretrained models, "
                  "run: \n'timm.list_models(pretrained=True)'" % model_name)
            return

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_file = weights_file

        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        #model = timm.create_model(self.model_name, pretrained=False, num_classes=0)
        #data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
        #self.model_transform = timm.data.create_transform(**data_cfg)
        #self.num_params = sum(p.numel() for p in model.parameters())
        #self.model_size_bytes = sum(p.numel() for p in model.parameters()) * 4

        self.model = self.load_model()
        self.file_name = None

    @staticmethod
    def _load_weights(model: Any,
                     weights_datastream: BytesIO) -> Any:
        """
        Load model weights, original weight saved with DataParallel
        Create new OrderedDict that does not contain `module`.
        :param model: Currently support EfficientNet
        :param weights_datastream: model weights, already loaded from storage
        :return: well trained model
        """
        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load weights
        state_dicts = torch.load(weights_datastream,
                                 map_location=device)

        new_state_dicts = OrderedDict()
        for k, v in state_dicts['net'].items():
            name = k[7:]
            new_state_dicts[name] = v
        model.load_state_dict(new_state_dicts)

        for param in model.parameters():
            param.requires_grad = False
        return model

    @staticmethod
    def _load_ckpt_weights(model: Any, ckpt_path: str) -> Any:
        """
        Load model weights from a checkpoint file.
        :param model: The model to load the weights into.
        :param ckpt_path: Path to the checkpoint file.
        :return: Model with loaded weights.
        """
        print(f"Loading checkpoint weights from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))

        # Assume the state dict is under a "state_dict" key, or directly load it
        state_dict = checkpoint.get('state_dict', checkpoint)

        # Remove any "module." prefix if saved with DataParallel
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_k = k[7:] if k.startswith("module.") else k
            new_state_dict[new_k] = v

        model.load_state_dict(new_state_dict)

        # Freeze model parameters
        for param in model.parameters():
            param.requires_grad = False
        return model


    def load_model(self):
        if self.weights_file is not None and self.weights_file.endswith('.pt'):
            print("Loading pretrained model ", self.model_name, ' from ', self.weights_file)
            # load efficient net
            efficientnet = models.get_model('efficientnet', self.model_name, 1275)

            with open(self.weights_file, 'rb') as f:
                weights_datastream = BytesIO(f.read())
                self.model = self._load_weights(efficientnet, weights_datastream)
        elif self.weights_file is not None and self.weights_file.endswith('.cpkt'):
            print("Loading pretrained model ", self.model_name, ' from ', self.weights_file)

            #load VIT
            vit = models.get_model('vit', self.model_name, 1275)
        else:
            print("Loading model ", self.model_name)
            self.model = timm.create_model(self.model_name, pretrained=True, num_classes=0)
        self.model.to(self.device)
        self.model.eval()
        return self.model

    def get_features(self, image_labels_batch: Optional[Batch] = None, crop_size: Optional[int] = None,
                     path_images_batch: Optional[str] = None, split: Optional[str] = None):
        if self.weights_file is not None and split is not None:
            file_name = f"{self.model_name}_pretrained_features/{split}"
            file_path = os.path.join(self.save_path, file_name)
        elif split is not None:
            file_name = f"{self.model_name}_features/{split}"
            file_path = os.path.join(self.save_path, file_name)
        else:
            file_name = f"{self.model_name}_features"
            file_path = os.path.join(self.save_path, file_name)

        if (os.path.exists(file_path) and image_labels_batch is not None
                and len(os.listdir(file_path)) >= len(image_labels_batch)):
            print("Loading features in ", file_path)
            features, flattened_labels, _ = self._load_features(file_path, image_labels_batch)
        elif image_labels_batch is None and os.path.exists(file_path):
            print("Loading features in ", file_path)
            features, flattened_labels, _ = self._load_features(file_path)
        else:
            os.makedirs(file_path, exist_ok=True)
            print("Extracting features in ", file_path)
            if self.model is None:
                self.model = self.load_model()

                #data_cfg = timm.data.resolve_data_config(self.model.pretrained_cfg)
                #self.model_transform = timm.data.create_transform(**data_cfg)
            if split == 'test':
                path_images_batch = path_images_batch + '/Test'
            #print("extract extract")
            features, labels, image_names = self._extract_features(image_labels_batch, crop_size, path_images_batch)
            #print("EXTRACT FEATURES SHAPE", features.shape)
            self._save_features(features, labels, image_names, file_path, split)
            flattened_labels = [item for sublist in labels for item in sublist]
            features = features.reshape(-1, features.shape[2])

        return features, flattened_labels

    def get_features_wth_image(self):
        file_name = f"{self.model_name}_features"
        file_path = os.path.join(self.save_path, file_name)
        if os.path.exists(file_path):
            print("Loading features in ", file_path)
            features, flattened_labels, image_list = self._load_features(file_path)
        else:
            raise Exception("You need to extract features before to use this method")

        return features, flattened_labels, image_list

    def _extract_features(self, image_labels_batch, crop_size, path_images_batch):
        features, labels, image_names = [], [], []
        print("Extracting ", len(image_labels_batch.image_labels_list), "features")
        for image in image_labels_batch.image_labels_list:
            with torch.no_grad():
                patches_tensor, patch_labels_list = crop_images_in_patches(image.image_name, image_labels_batch,
                                                                           crop_size, path_images_batch)
                #print("Patches tensor in extraction ", patches_tensor.shape) -> (25,3,224,224)
                if self.model_name == 'efficientnet-b0':
                    feature = self.model.extract_features(patches_tensor)
                else:
                    feature = self.model(patches_tensor)
                #(25,1024) expected
                #print("feature of tensors cropped", feature.shape)
                features.append(feature.detach())
                labels.append(patch_labels_list)
                image_names.append(image.image_name)

        features_stacked = torch.stack(features, dim=0)
        #print('FINAL FEATURES STACKED', features_stacked.shape)
        #features_tensor = features_stacked.reshape(-1, features_stacked.shape[2])
        #print("all the features in one tensor", features_tensor.shape)
        #flatten label here ?
        return features_stacked.detach(), labels, image_names

    @staticmethod
    def _save_features(features, labels, image_names, file_path, split):
        """Saves extracted features to file."""
        for i in range(len(image_names)):
            file_name, _ = os.path.splitext(image_names[i])
            file_name = f"{file_name}.pt"
            file_full_path = os.path.join(file_path, file_name)
            print(file_full_path)
            feature = features[i].detach().cpu()
            label = labels[i]
            directory = os.path.dirname(file_full_path)  # Get the directory from the file path
            if not os.path.exists(directory):
                os.makedirs(directory)  # Create the directory and any necessary parent directories

            print(directory)
            torch.save({'features': feature, 'labels': label}, file_full_path, pickle_module=pickle)
            #file_size_bytes = os.path.getsize(file_full_path)
            #file_size_mb = file_size_bytes / (1024 * 1024)
            # print("Taille du fichier sauvegardé :", file_size_mb, "Mo")

    @staticmethod
    def _load_features(file_path, image_labels_batch=None):
        """Loads features from batch."""
        features_list = []
        labels_list = []
        # TODO : make it clean so if there is not all the features corresponding to the batch :
        #  extract the missing features
        for filename in os.listdir(file_path):
            image_name, _ = os.path.splitext(filename)
            if isinstance(image_labels_batch, Batch) and (image_name in image_labels_batch.get_image_names_wtht_jpg()):
                if filename.endswith(".pt"):
                    file_full_path = os.path.join(file_path, filename)
                    data = torch.load(file_full_path, pickle_module=pickle)
                    features_list.append(data['features'])
                    labels_list.append(data['labels'])
            elif image_labels_batch is None:
                if filename.endswith(".pt"):
                    file_full_path = os.path.join(file_path, filename)
                    data = torch.load(file_full_path, pickle_module=pickle)
                    features_list.append(data['features'])
                    labels_list.append(data['labels'])
            else :
                raise ValueError(f"{filename} is not extracted from features")
        features_tensor = torch.stack(features_list, dim=0)
        features_tens = features_tensor.reshape(-1, features_tensor.shape[2])
        flattened_labels_list = [item for sublist in labels_list for item in sublist]
        return features_tens.cuda(), flattened_labels_list, os.listdir(file_path)
