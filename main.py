import argparse
import yaml
import os
from training import main_train, MLPClassifier
from extract_features import extract_all_features
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

    print(f"Extracting features for dataset: {dataset_name} from {image_path}")
    features = extract_all_features(dataset_name, image_path, annotations_file, crop_size,  model_name, weights_file)
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


def get_config_path(cli_path):
    """Find the path of config file"""
    if cli_path:
        return cli_path
    default_path = "params.yaml"
    if os.path.exists(default_path):
        return default_path
    raise FileNotFoundError(
        "No configuration file provided, and 'params.yaml' not found in the current directory."
    )


def main():
    parser = argparse.ArgumentParser(description="Pipeline for feature extraction and training.")

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Extract Command
    parser_extract = subparsers.add_parser('extract', help='Extract features based on configuration file.')
    parser_extract.add_argument('config_path', nargs='?', type=str, help='Path to the YAML configuration file.')

    # Train Command
    parser_train = subparsers.add_parser('train', help='Train a model using extracted features.')
    parser_train.add_argument('config_path', nargs='?', type=str, help='Path to the YAML configuration file.')

    args = parser.parse_args()

    try:
        config_path = get_config_path(args.config_path)

        if args.command == 'extract':
            extract_features_from_config(config_path)
        elif args.command == 'train':
            train_model_from_config(config_path)
        else:
            parser.print_help()
    except FileNotFoundError as e:
        print(e)


if __name__ == "__main__":
    main()
