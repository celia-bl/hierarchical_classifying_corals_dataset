import pickle
import numpy as np
import os

def calculate_means(metric_report, classifier_name, nb_image, final_metric_report):
    #print("Calculate means ", classifier_name, metric_report[classifier_name])
    precision_values = [report['precision'] for report in metric_report[classifier_name]]
    recall_values = [report['recall'] for report in metric_report[classifier_name]]
    f1_values = [report['f1-score'] for report in metric_report[classifier_name]]
    support_values = [report['support'] for report in metric_report[classifier_name]]

    precision_mean = np.mean(precision_values)
    recall_mean = np.mean(recall_values)
    f1_mean = np.mean(f1_values)

    precision_std = np.std(precision_values)
    recall_std = np.std(recall_values)
    f1_std = np.std(f1_values)

    final_metric_report[classifier_name][nb_image] = {
        'precision': precision_mean,
        'recall': recall_mean,
        'f1-score': f1_mean,
        'std_precision': precision_std,
        'std_recall': recall_std,
        'std_f1-score': f1_std
    }
    return final_metric_report


def calculate_intermediate_means(metric_report, classifier_name, nb_image, final_metric_report):
    #print("Intermediate report", metric_report)
    results = {}

    # Obtenir les labels
    labels = set(label for report in metric_report[classifier_name] for label in report.keys())

    # Parcourir les labels dans le rapport de métriques
    for label in labels:
        precision_values = []
        recall_values = []
        f1_values = []
        test_support_values = []
        train_support_values = []
        test_pred_values = []

        for report in metric_report[classifier_name]:
            if label in report:
                precision_values.append(report[label]['precision'])
                recall_values.append(report[label]['recall'])
                f1_values.append(report[label]['f1_score'])
                test_support_values.append(report[label]['test_support'])
                train_support_values.append(report[label]['train_support'])
                test_pred_values.append(report[label]['test_pred'])

        precision_mean = np.mean(precision_values)
        recall_mean = np.mean(recall_values)
        f1_mean = np.mean(f1_values)
        test_support_mean = np.mean(test_support_values)
        train_support_mean = np.mean(train_support_values)
        test_pred_mean = np.mean(test_pred_values)

        precision_std = np.std(precision_values)
        recall_std = np.std(recall_values)
        f1_std = np.std(f1_values)

        results[label] = {
            'precision': precision_mean,
            'recall': recall_mean,
            'f1-score': f1_mean,
            'std_precision': precision_std,
            'std_recall': recall_std,
            'std_f1-score': f1_std,
            'train_support': train_support_mean,
            'test_support': test_support_mean,
            'test_pred': test_pred_mean
        }

    # Ajouter les résultats au rapport final
    final_metric_report[classifier_name][nb_image] = results

    return final_metric_report


def extract_last_element(list_of_lists):
    return [sublist[-1] for sublist in list_of_lists]


def count_label_occurrences(y_train, label):
    label_counter = 0
    for labels_list in y_train:
        if label in labels_list:
            label_counter += 1
    return label_counter


def save_pickle(obj, filename):
    with open(filename, 'wb') as file :
        pickle.dump(obj, file)


def load_pickle(filename):
    """Charge un objet Python depuis un fichier pickle."""
    with open(filename, 'rb') as file:
        return pickle.load(file)


def load_all_pickles(directory, target_module):
    for filename in os.listdir(directory):
        if filename.endswith('.pkl'):
            var_name = filename[:-4]  # Retirer le ".pkl"
            file_path = os.path.join(directory, filename)

            data = load_pickle(file_path)  # Charger le fichier

            setattr(target_module, var_name, data)  # Stocker la variable dynamiquement
            print(f"Chargé : {var_name}")


def calculate_means_emissions(flat_emissions_run_reports, classifier_name, nb_image, flat_emissions_reports):
    """
    Calculate mean of emissions of the same classifier across several runs
    :param flat_emissions_run_reports: dict with lists of emissions for each runs
    :param classifier_name: name of the classifier
    :param nb_image: nb_image of the run
    :param flat_emissions_reports: dict to store the mean
    """
    means_emissions = {}
    keys_to_exclude = {'run_id', 'country_name', 'country_iso_code', 'region', 'cloud_provider', 'cloud_region', 'os',
                       'python_version', 'codecarbon_version', 'cpu_count', 'cpu_model', 'gpu_count', 'gpu_model',
                       'ram_total_size', 'tracking_mode', 'on_cloud', 'pue'}
    count = len(flat_emissions_run_reports[classifier_name])

    # Calculate totals for each key
    for entry in flat_emissions_run_reports[classifier_name]:
        for key, value in entry.items():
            if key not in keys_to_exclude and isinstance(value, (int, float)):
                if key not in means_emissions:
                    means_emissions[key] = 0
                means_emissions[key] += value

    # Calculate average
    means_emissions = {key: total / count for key, total in means_emissions.items()}
    flat_emissions_reports[classifier_name][nb_image] = means_emissions

    return flat_emissions_reports


def eq_patch_emission_extraction(extract_features, nb_total_patches):
    emission_per_patch = extract_features['emissions']/nb_total_patches
    return emission_per_patch

def nb_total_patches(dataset, folder_images):
    image_extensions = ('.jpg', '.png')
    count = 0
    for root, _, files in os.walk(folder_images):
        count += sum(1 for file in files if file.lower().endswith(image_extensions))
    if dataset == 'RIO':
        count=count*25
    elif dataset == 'MLC':
        count=count*200
    elif dataset == 'TasCPC':
        count=count*50
    return count