from plot_logic import plot_hierarchy_graphviz, plot_hierarchy_sunburst, plot_hierarchy_sunburst_balanced, plot_occurences
import yaml
from hierarchy_params import get_hierarchy_info
from load_data import get_features
import numpy as np
from collections import Counter


def visualize_data(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    dataset_name = config.get('dataset')
    features_path = config.get('features_path')
    folder_path = features_path + '/efficientnet-b0_pretrained_features/' + dataset_name
    features, labels = get_features(folder_path)
    label_count = Counter(labels)

    hierarchy, label_to_line_number, code_label_full_name, line_number_to_label = get_hierarchy_info(config_path)
    plot_hierarchy_graphviz(hierarchy, root_name='Root')
    plot_hierarchy_sunburst(hierarchy, root='Root')
    plot_hierarchy_sunburst_balanced(hierarchy, label_count, root='Root')
    plot_occurences(label_count)
    #plot_hierarchy(hierarchy, root='root')