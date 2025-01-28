import argparse
import yaml
import os
from training import main_train, MLPClassifier
from extract_features import extract_all_features
from visualize_data import visualize_data
from utils_file import save_pickle
import warnings


def build_classifiers(classifiers_config):
    classifiers_dict = {}
    for classifier in classifiers_config:
        model_type = classifier['type']
        params = classifier['params']

        if 'hidden_layer_sizes' in params:
            params['hidden_layer_sizes'] = tuple(params['hidden_layer_sizes'])

        # Support only MLPClassifier for now
        classifiers_dict[model_type] = MLPClassifier(**params)

    return classifiers_dict

def extract_features_from_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    dataset_name = config.get('dataset')
    image_path = config.get('image_path')
    annotations_file = config.get('annotations_file')
    crop_size = config.get('crop_size')
    model_name = config.get('model_extraction_name')
    weights_file = config.get('weights_file')
    result_file = config.get('result_folder')
    result_dataset_folder = os.path.join(result_file, dataset_name)

    if not os.path.exists(result_dataset_folder):
        os.makedirs(result_dataset_folder)
        print(f"Folder {result_dataset_folder} has been created.")

    print(f"Extracting features for dataset: {dataset_name} from {image_path}")
    clean_emissions = extract_all_features(dataset_name, image_path, annotations_file, crop_size,  model_name, weights_file)
    save_pickle(clean_emissions, result_dataset_folder + '/extract_features.pkl')
    print(f"Features extracted if you want to train a model use the train command")
    pass


def train_model_from_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    dataset_name = config.get('dataset')
    features_path = config.get('features_path')
    folder_path = features_path + '/efficientnet-b0_pretrained_features/' + dataset_name
    list_nb_training_patches = config.get('list_nb_training_patches')
    result_folder = config.get('result_folder')
    result_dataset_folder = os.path.join(result_folder, dataset_name)

    if not os.path.exists(result_dataset_folder):
        os.makedirs(result_dataset_folder)
        print(f"Folder {result_dataset_folder} has been created.")


    if dataset_name == 'RIO':
        intermediate_labels = ['Bleached', 'Corals', 'Algae', 'Non Calcified', 'Calcified', 'Hard', 'Soft',
                               'Substrates', 'Rock', 'SHAD', 'Unk']
    elif dataset_name == 'MLC':
        intermediate_labels = ['Corals', 'Algae', 'Hard', 'Soft', 'Sand']
    elif dataset_name == 'TasCPC':
        intermediate_labels = ['Sponges', 'Cnidaria', 'Algae', 'Echinoderms', 'Bryozoa']
    else:
        print('Invalid dataset')
        raise ValueError('Invalid dataset')

    flat_classifiers = build_classifiers(config['classifiers']['flat_classifiers'])
    hierarchical_classifiers = build_classifiers(config['classifiers']['hierarchical_classifiers'])
    warnings.filterwarnings("ignore")
    print(f"Training model for dataset: {dataset_name} using features from {features_path}")
    print(f"Flat models: {flat_classifiers}; hierarchical models: {hierarchical_classifiers}")
    main_train(
        folder_path=folder_path,
        flat_classifiers=flat_classifiers,
        hierarchical_classifiers=hierarchical_classifiers,
        list_nb_training_patches=list_nb_training_patches,
        intermediate_labels=intermediate_labels,
        result_file=result_dataset_folder,
        config_path = config_path
    )
    pass

def visualize_data_from_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    dataset_name = config.get('dataset')
    image_path = config.get('image_path')
    annotations_file = config.get('annotations_file')
    crop_size = config.get('crop_size')
    model_name = config.get('model_extraction_name')
    weights_file = config.get('weights_file')

    print(f"Visualing data for dataset: {dataset_name} from {image_path}")
    visualize_data(config_path)
    pass

def get_config_path(cli_path):
    """Find the path of config file"""
    if cli_path:
        return cli_path
    default_path = "params/params.yaml"
    if os.path.exists(default_path):
        return default_path
    raise FileNotFoundError(
        "No configuration file provided, and 'params.yaml' not found in the current directory."
    )


def check_config(config_path):
    """
    Verify that all paths in the configuration file exist.
    :param config_path: Path to the YAML configuration file.
    :raises FileNotFoundError: If any path in the configuration does not exist.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    if config.get('dataset') == 'RIO':
    # Paths to verify
        paths_to_check = [
            config.get('image_path'),
            config.get('features_path'),
            config.get('annotations_file'),
            config.get('weights_file'),
            config.get('labelset_file'),
        ]
    elif config.get('dataset') == 'MLC':
        paths_to_check = [
            config.get('image_path'),
            config.get('features_path'),
            config.get('weights_file'),
        ]
    elif config.get('dataset') == 'TasCPC':
        paths_to_check = [
            config.get('image_path'),
            config.get('features_path'),
            config.get('annotations_file'),
            config.get('weights_file'),
        ]
    else :
        raise ValueError('Dataset Invalid')
    # Check each path
    for path in paths_to_check:
        if path and not os.path.exists(path):  # Check only non-empty paths
            raise FileNotFoundError(f"Path not found: {path}")

    print("All paths in the configuration file exist.")

def main():
    parser = argparse.ArgumentParser(description="Pipeline for feature extraction and training.")

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Extract Command
    parser_extract = subparsers.add_parser('extract', help='Extract features based on configuration file.')
    parser_extract.add_argument('config_path', nargs='?', type=str, help='Path to the YAML configuration file.')

    # Train Command
    parser_train = subparsers.add_parser('train', help='Train a model using extracted features.')
    parser_train.add_argument('config_path', nargs='?', type=str, help='Path to the YAML configuration file.')

    # Visualize Command
    parser_train = subparsers.add_parser('visualize', help='Visualize the dataset')
    parser_train.add_argument('config_path', nargs='?', type=str, help='Path to the YAML configuration file.')

    args = parser.parse_args()

    try:
        config_path = get_config_path(args.config_path)
        check_config(config_path)
        if args.command == 'extract':
            extract_features_from_config(config_path)
        elif args.command == 'train':
            train_model_from_config(config_path)
        elif args.command == 'visualize':
            visualize_data_from_config(config_path)
        else:
            parser.print_help()
    except FileNotFoundError as e:
        print(e)


if __name__ == "__main__":
    main()
