import os
import torch
import numpy as np
import random
from collections import Counter
from sklearn.model_selection import train_test_split

def get_features(folder_path, nb_image, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    all_files = []

    # Parcours récursif des dossiers et sous-dossiers
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.pt'):
                all_files.append(os.path.join(root, file))

    random.shuffle(all_files)

    selected_features = []
    selected_labels = []

    for filename in all_files[:nb_image]:
        data = torch.load(filename)

        features = data['features'].tolist()
        labels = data['labels']

        features_np = np.array([np.array(tensor) for tensor in features])

        selected_features.extend(features_np)
        selected_labels.extend(labels)

    selected_features_np = np.array(selected_features)
    return selected_features_np, selected_labels


def load_data(folder_path, nb_image=None):
    if nb_image is None:
        nb_image = len(os.listdir(folder_path))
    features, labels = get_features(folder_path, nb_image)
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=None)
    return x_train, x_test, y_train, y_test


def load_representative_data(folder_path, nb_image=None):
    if nb_image is None:
        nb_image = len(os.listdir(folder_path))
    features, labels = get_features(folder_path, nb_image)
    label_counter = Counter(labels)
    len_test_set = len(labels)*0.1

    label_indices = {label: np.where(np.array(labels) == label)[0] for label in label_counter.keys()}

    test_indices = []
    nb_singleton_max = 5
    nb_singleton = 0
    for label, indices in label_indices.items():
        if len(indices) > 1:
            num_to_select = int(len(indices)*0.1)
            test_indices.extend(random.sample(indices.tolist(), num_to_select))
        elif len(indices) == 1:
            if nb_singleton < nb_singleton_max:
                test_indices.extend(indices.tolist())
                nb_singleton += 1

    print("TEST SIZE (patches) ", len(test_indices), " ideal test size : ", len_test_set)

    test_indices = list(set(test_indices))
    train_indices = list(set(range(len(labels))) - set(test_indices))

    features_train = [features[i] for i in train_indices]
    labels_train = [labels[i] for i in train_indices]
    features_test = [features[i] for i in test_indices]
    labels_test = [labels[i] for i in test_indices]

    return features_train, features_test, labels_train, labels_test



def load_representative_data_mlc(folder_path, nb_image=None):
    subfolders = [os.path.join(folder_path, d) for d in os.listdir(folder_path) if
                  os.path.isdir(os.path.join(folder_path, d))]

    all_features = []
    all_labels = []

    # Si nb_image n'est pas spécifié, calculer le nombre total d'images disponibles
    if nb_image is None:
        nb_image = sum([len(os.listdir(subfolder)) for subfolder in subfolders])
        print('Number of images found: ', nb_image)
        for i, subfolder in enumerate(subfolders):
            images_to_load = len(os.listdir(subfolder))

            features, labels = get_features(subfolder, images_to_load)

            all_features.extend(features)
            all_labels.extend(labels)

    elif nb_image is not None:
        # Distribuer le nombre d'images à charger par sous-dossier
        images_per_folder = nb_image // len(subfolders)
        remaining_images = nb_image % len(subfolders)
        for i, subfolder in enumerate(subfolders):
            if i == len(subfolders) - 1:
                images_to_load = images_per_folder + remaining_images
            else:
                images_to_load = images_per_folder

            features, labels = get_features(subfolder, images_to_load)

            all_features.extend(features)
            all_labels.extend(labels)

    all_labels_counter = Counter(all_labels)
    #plot_counter(all_labels_counter)

    # Filtrer les caractéristiques et labels pour éliminer ceux avec le label 'Off' ou 'Null'
    filtered_indices = [i for i, label in enumerate(all_labels) if label != 'Off' and label != 'Null' and label != 'OFF']
    filtered_features = [all_features[i] for i in filtered_indices]
    filtered_labels = [all_labels[i] for i in filtered_indices]

    label_counter = Counter(filtered_labels)
    len_test_set = len(filtered_labels) * 0.1

    label_indices = {label: np.where(np.array(filtered_labels) == label)[0] for label in label_counter.keys()}
    #plot_counter(label_counter)
    test_indices = []
    nb_singleton_max = 5
    nb_singleton = 0
    for label, indices in label_indices.items():
        if len(indices) > 1:
            num_to_select = int(len(indices) * 0.1)
            test_indices.extend(random.sample(indices.tolist(), num_to_select))
        elif len(indices) == 1:
            if nb_singleton < nb_singleton_max:
                test_indices.extend(indices.tolist())
                nb_singleton += 1

    print("TEST SIZE (patches) ", len(test_indices), " ideal test size : ", len_test_set)

    test_indices = list(set(test_indices))
    train_indices = list(set(range(len(filtered_labels))) - set(test_indices))

    features_train = [filtered_features[i] for i in train_indices]
    labels_train = [filtered_labels[i] for i in train_indices]
    features_test = [filtered_features[i] for i in test_indices]
    labels_test = [filtered_labels[i] for i in test_indices]

    return features_train, features_test, labels_train, labels_test

