import numpy as np
from load_data import load_representative_data_mlc, load_representative_data, load_data
from hiclass import LocalClassifierPerParentNode, LocalClassifierPerNode, LocalClassifierPerLevel
from hiclass.HierarchicalClassifier import make_leveled
from analyze_data import plot_avg_distrib, plot_label_distribution
from collections import defaultdict
from hierarchy_utils import convert_labels_to_hierarchy, convert_label_to_hierarchy
from hierarchy_params import get_hierarchy_info
from metrics import classification_report, hierarchical_metrics_from_flat, metrics_intermediate_labels, normal_metrics, get_hierarchical_scores_from_hiclass
from plot_logic import plot_metrics_by_nb_images, plot_intermediate_metrics
from utils_file import calculate_means, calculate_intermediate_means, save_pickle, calculate_means_emissions
from utils.emissions import tracker, clean_emissions_data
import yaml

from sklearn.neural_network import MLPClassifier

def train_and_evaluate(classifier, x_train, y_train, x_test, y_test, hierarchy, dic=None, int_labels=None):
    tracker.start_task("training-flat")
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    emissions_data = tracker.stop_task()
    clean_emissions = clean_emissions_data(emissions_data)
    if dic is not None:
        y_pred = decode_labels(y_pred, dic)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    flat_hi_report = hierarchical_metrics_from_flat(y_pred, y_test, hierarchy)

    mispredicted_vectors = {}
    for i, (pred, true) in enumerate(zip(y_pred, y_test)):
        if pred != true:

            mispredicted_vectors[tuple(x_test[i])] = {
                'predicted': pred,
                'true_label': true
            }

    if int_labels is not None:
        y_hier_train = convert_label_to_hierarchy(y_train, hierarchy)
        y_pred_hierarchized, y_true_hierarchized = convert_labels_to_hierarchy(y_pred, y_test, hierarchy)
        intermediate_metrics = metrics_intermediate_labels(y_true_hierarchized, y_pred_hierarchized, y_hier_train, int_labels)
    else:
        intermediate_metrics = None
    return report, flat_hi_report, intermediate_metrics, mispredicted_vectors, clean_emissions

def train_and_evaluate_hierarchical(classifier, x_train, x_test, y_hier_train, y_hier_test, int_labels=None):
    tracker.start_task("training-hi")
    classifier.fit(x_train, y_hier_train)
    y_hier_pred = classifier.predict(x_test)
    emissions_data = tracker.stop_task()
    clean_emissions = clean_emissions_data(emissions_data)

    # normal scores from normal metrics with weighted avg
    hi_flat_report = normal_metrics(y_hier_pred, y_hier_test)['weighted avg']
    hi_report = get_hierarchical_scores_from_hiclass(y_hier_pred, y_hier_test)

    mispredicted_vectors = {}
    y_hier_test_hi = make_leveled(y_hier_test)
    for i, (pred, true) in enumerate(zip(y_hier_pred, y_hier_test_hi)):
        if pred[-1] != true[-1]:
            mispredicted_vectors[tuple(x_test[i])] = {
                'predicted': tuple(pred),
                'true_label': true
            }

    if int_labels is not None:

        intermediate_metrics = metrics_intermediate_labels(y_hier_test, y_hier_pred, y_hier_train, int_labels)
    else:
        intermediate_metrics = None
    return hi_report, hi_flat_report, intermediate_metrics, mispredicted_vectors, clean_emissions

def decode_labels(encoded_labels, label_to_index):
    if label_to_index is None:
        decoded_labels = encoded_labels
    else:
        index_to_label = {index: label for label, index in label_to_index.items()}
        decoded_labels = [index_to_label.get(index, "ERR") for index in encoded_labels]
    return decoded_labels

def encode_labels(y_train, y_test, clf_name, label_to_line_number):
    dict_index_label = None
    if clf_name == 'MLPHead-147':
        dict_index_label = label_to_line_number
        y_train = [label_to_line_number[label] for label in y_train]
        y_test = [label_to_line_number[label] for label in y_test]
    elif clf_name == 'MLPHead-40':
        all_labels = set(y_train + y_test)
        if len(all_labels) >= 40:
            raise ValueError("You cannot use MLPHead-30 for your dataset there is more than 40 different labels in it")

        label_to_index = {label: index for index, label in enumerate(all_labels)}
        dict_index_label = label_to_index
        y_train = [label_to_index[label] for label in y_train]
        y_test = [label_to_index[label] for label in y_test]

    return y_train, y_test, dict_index_label

def main_train(folder_path, flat_classifiers, hierarchical_classifiers, list_nb_training_patches, intermediate_labels, result_file, config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    dataset = config.get('dataset')
    hierarchy, label_to_line_number, code_label_full_name, line_number_to_label = get_hierarchy_info(config_path)
    flat_reports = {}
    hi_flat_reports = {}
    flat_emissions_reports = {}
    hi_reports = {}
    flat_hi_reports = {}
    hi_intermediate_reports = {}
    hi_emissions_reports = {}
    flat_intermediate_reports = {}
    mispredicted_vectors_flat = {}
    mispredicted_vectors_hi = {}

    if dataset == 'MLC':
        x_train, x_test, y_train, y_test = load_representative_data_mlc(folder_path)
    elif dataset == 'RIO':
        # x_train, x_test, y_train, y_test = load_data(folder_path)
        x_train, x_test, y_train, y_test = load_representative_data(folder_path)
        # x_train, x_test, y_train, y_test = load_representative_data_mlc(folder_path)
        # x_train, x_test, y_train, y_test = load_representative_uniform_data(folder_path)
    elif dataset == 'TasCPC':
        x_train, x_test, y_train, y_test = load_data(folder_path)

    plot_label_distribution(y_test)

    for classifier_name in flat_classifiers.keys():
        flat_reports[classifier_name] = {}
        flat_hi_reports[classifier_name] = {}
        flat_intermediate_reports[classifier_name] = {}
        mispredicted_vectors_flat[classifier_name] = {}
        flat_emissions_reports[classifier_name] = {}

    for classifier_name in hierarchical_classifiers.keys():
        hi_reports[classifier_name] = {}
        hi_flat_reports[classifier_name] = {}
        hi_intermediate_reports[classifier_name] = {}
        mispredicted_vectors_hi[classifier_name] = {}
        hi_emissions_reports[classifier_name] = {}

    for nb_image in list_nb_training_patches:
        if nb_image < 20000:
            n_runs = 10
        else:
            n_runs = 1
        flat_run_reports = {classifier_name: [] for classifier_name in flat_classifiers.keys()}
        flat_hi_run_reports = {classifier_name: [] for classifier_name in flat_classifiers.keys()}
        flat_intermediate_run_reports = {classifier_name: [] for classifier_name in flat_classifiers.keys()}
        flat_emissions_run_reports = {classifier_name: [] for classifier_name in flat_classifiers.keys()}

        hi_run_reports = {classifier_name: [] for classifier_name in hierarchical_classifiers.keys()}
        hi_flat_run_reports = {classifier_name: [] for classifier_name in hierarchical_classifiers.keys()}
        hi_intermediate_run_reports = {classifier_name: [] for classifier_name in hierarchical_classifiers.keys()}
        hi_emissions_run_reports = {classifier_name: [] for classifier_name in hierarchical_classifiers.keys()}

        label_counts = defaultdict(lambda: [])
        for _ in range(n_runs):
            # shuffle training set randomly
            indices = np.random.permutation(range(len(x_train)))

            x_train_shuffled = [x_train[i] for i in indices]
            y_train_shuffled = [y_train[i] for i in indices]
            x_train_subset = x_train_shuffled[:nb_image]
            y_train_subset = y_train_shuffled[:nb_image]


            #load representative data
            #x_train_subset, y_train_subset = load_representative_training_set(x_train, y_train, y_test, nb_image)
            #x_train_subset, y_train_subset = load_uniform_training_data(x_train, y_train, nb_image)

            unique_labels, counts = np.unique(y_train_subset, return_counts=True)
            for label, count in zip(unique_labels, counts):
                label_counts[label].append(count)

            y_hier_train, y_hier_test = convert_labels_to_hierarchy(y_train_subset, y_test, hierarchy)
            y_train_encoded, y_test_encoded, dict_encode = encode_labels(y_train_subset, y_test, 'MLPHead-147', label_to_line_number)

            for classifier_name, classifier in flat_classifiers.items():
                print(f'Processing {classifier_name}')


                if 'MLPHead' in classifier_name:
                    y_train_subset = y_train_encoded

                (flat_report, flat_hi_report, flat_intermediate_metric,
                 mispredicted_vectors_flat_run,
                 emissions_data_flat) = train_and_evaluate(classifier,
                                                           x_train_subset, y_train_subset,
                                                           x_test, y_test,
                                                           hierarchy=hierarchy,  int_labels=intermediate_labels)

                flat_run_reports[classifier_name].append(flat_report['weighted avg'])
                flat_hi_run_reports[classifier_name].append(flat_hi_report)
                flat_intermediate_run_reports[classifier_name].append(flat_intermediate_metric)
                flat_emissions_run_reports[classifier_name].append(emissions_data_flat)

                for mispredicted_feature in mispredicted_vectors_flat_run.keys():
                    if mispredicted_feature not in mispredicted_vectors_flat[classifier_name]:
                        mispredicted_vectors_flat[classifier_name][mispredicted_feature] = {
                            'predicted': [],
                            'true_label': mispredicted_vectors_flat_run[mispredicted_feature]['true_label']
                        }
                    mispredicted_vectors_flat[classifier_name][mispredicted_feature]['predicted'].append(
                        mispredicted_vectors_flat_run[mispredicted_feature]['predicted'])

                print(f'Finish processing {classifier_name}')

            for classifier_name, node_classifier in hierarchical_classifiers.items():
                print(f'Processing {classifier_name}')

                if 'LCPPN' in classifier_name:
                    classifier = LocalClassifierPerParentNode(local_classifier=node_classifier)
                elif 'LCPN' in classifier_name:
                    classifier = LocalClassifierPerNode(local_classifier=node_classifier)
                elif 'LCPL' in classifier_name:
                    classifier = LocalClassifierPerLevel(local_classifier=node_classifier)
                else:
                    raise ValueError(f'Unknown classifier {classifier_name}, specify LCPPN or LCPN')

                (hi_report, hi_flat_report,
                 hi_intermediate_metric, mispredicted_vectors_hi_run,
                 hi_emissions) = train_and_evaluate_hierarchical(classifier,
                                                                 x_train_subset, x_test,
                                                                 y_hier_train, y_hier_test,
                                                                 int_labels=intermediate_labels)
                hi_run_reports[classifier_name].append(hi_report)
                hi_flat_run_reports[classifier_name].append(hi_flat_report)
                hi_intermediate_run_reports[classifier_name].append(hi_intermediate_metric)
                hi_emissions_run_reports[classifier_name].append(hi_emissions)

                for mispredicted_feature in mispredicted_vectors_hi_run.keys():
                    if mispredicted_feature not in mispredicted_vectors_hi[classifier_name]:
                        mispredicted_vectors_hi[classifier_name][mispredicted_feature] = {
                            'predicted': [],
                            'true_label': mispredicted_vectors_hi_run[mispredicted_feature]['true_label']
                        }
                    mispredicted_vectors_hi[classifier_name][mispredicted_feature]['predicted'].append(
                        mispredicted_vectors_hi_run[mispredicted_feature]['predicted'])

                print(f'Finish processing {classifier_name}')

        #print(flat_run_reports)
        #print(flat_hi_run_reports)
        average_label_counts = {label: np.mean(counts) for label, counts in label_counts.items()}
        plot_avg_distrib(average_label_counts)
        # Compute mean and standard deviation for flat classifiers
        for classifier_name in flat_classifiers.keys():
            flat_reports = calculate_means(flat_run_reports, classifier_name, nb_image, flat_reports)
            flat_hi_reports = calculate_means(flat_hi_run_reports, classifier_name, nb_image, flat_hi_reports)
            flat_intermediate_reports = calculate_intermediate_means(flat_intermediate_run_reports, classifier_name, nb_image, flat_intermediate_reports)
            flat_emissions_reports = calculate_means_emissions(flat_emissions_run_reports, classifier_name, nb_image, flat_emissions_reports)

        # Compute mean and standard deviation for hierarchical classifiers
        for classifier_name in hierarchical_classifiers.keys():
            hi_reports = calculate_means(hi_run_reports, classifier_name, nb_image, hi_reports)
            hi_flat_reports = calculate_means(hi_flat_run_reports, classifier_name, nb_image, hi_flat_reports)
            hi_intermediate_reports = calculate_intermediate_means(hi_intermediate_run_reports, classifier_name, nb_image, hi_intermediate_reports)
            hi_emissions_reports = calculate_means_emissions(hi_emissions_run_reports, classifier_name, nb_image, hi_emissions_reports)

    #print(hi_intermediate_reports)
    #print(flat_intermediate_reports)
    #print('Final dict ', flat_reports)
    #print('Final dict ', flat_hi_reports)
    #print('Final dict ', hi_flat_reports)
    #print('Final dict ', hi_reports)
    #print('Intermediate dict ', flat_intermediate_reports)


    save_pickle(flat_reports, result_file + '/flat_reports.pkl')
    save_pickle(flat_hi_reports, result_file + '/flat_hi_reports.pkl')
    save_pickle(hi_flat_reports, result_file + '/hi_flat_reports.pkl')
    save_pickle(hi_reports, result_file + '/hi_reports.pkl')
    save_pickle(flat_intermediate_reports, result_file + '/flat_intermediate_reports.pkl')
    save_pickle(hi_intermediate_reports, result_file + '/hi_intermediate_reports.pkl')
    save_pickle(y_test, result_file + '/y_test.pkl')
    save_pickle(intermediate_labels, result_file + '/intermediate_labels.pkl')

    print(flat_emissions_reports)
    plot_metrics_by_nb_images(flat_reports, flat_hi_reports, hi_flat_reports, hi_reports)
    plot_intermediate_metrics(flat_intermediate_reports, hi_intermediate_reports)
    #comparison_cover(flat_intermediate_reports, hi_intermediate_reports, y_test, intermediate_labels)
    #plot_mispredict_vectors(mispredicted_vectors_hi, mispredicted_vectors_flat, folder_path, annotations_file, path_images_batches)

