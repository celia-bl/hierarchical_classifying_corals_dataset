#from plot_logic import plot_hierarchy_graphviz, plot_hierarchy_sunburst, plot_hierarchy_sunburst_balanced, plot_occurences
from plot_logic import plot_hierarchy_sunburst, plot_hierarchy_sunburst_balanced, plot_occurences, plot_metrics_by_nb_images, plot_emissions_vs_performance, plot_emissions, plot_f1_vs_emission_diff
import yaml
from hierarchy_params import get_hierarchy_info
from load_data import get_features
import numpy as np
from collections import Counter
from utils_file import load_all_pickles, nb_total_patches, eq_patch_emission_extraction
import sys


def visualize_data(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    dataset_name = config.get('dataset')
    features_path = config.get('features_path')
    folder_path = features_path + '/efficientnet-b0_pretrained_features/' + dataset_name
    features, labels = get_features(folder_path)
    label_count = Counter(labels)

    hierarchy, label_to_line_number, code_label_full_name, line_number_to_label = get_hierarchy_info(config_path)
    #plot_hierarchy_graphviz(hierarchy, root_name='Root')
    plot_hierarchy_sunburst(hierarchy, root='Root')
    plot_hierarchy_sunburst_balanced(hierarchy, label_count, root='Root')
    plot_occurences(label_count)
    #plot_hierarchy(hierarchy, root='root')

def visualize_results(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    dataset_name = config.get('dataset')
    result_folder = config.get('result_folder') + '/' + dataset_name
    image_folder = config.get('image_path')
    load_all_pickles(result_folder, sys.modules[__name__])
    plot_metrics_by_nb_images(flat_reports, flat_hi_reports, hi_flat_reports, hi_reports)
    plot_emissions_vs_performance(flat_emissions, hi_emissions, flat_reports, flat_hi_reports, hi_flat_reports, hi_reports, extract_features)
    nb_total_patchs = nb_total_patches(dataset_name, image_folder)
    emmission_per_patch = eq_patch_emission_extraction(extract_features, nb_total_patchs)
    plot_emissions(flat_emissions, hi_emissions, emmission_per_patch)
    plot_f1_vs_emission_diff(flat_emissions, hi_emissions, flat_reports, hi_reports)