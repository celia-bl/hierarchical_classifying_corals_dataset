from hierarchy_utils import convert_labels_to_hierarchy, convert_hierarchy_to_label
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
from hiclass.metrics import f1, precision, recall
from utils_file import extract_last_element, count_label_occurrences


def hierarchical_metrics_from_flat(y_pred, y_true, hierarchy):
    y_pred_hierarchized, y_true_hierarchized = convert_labels_to_hierarchy(y_pred, y_true, hierarchy)
    flat_hi_report = get_hierarchical_scores_from_hiclass(y_pred_hierarchized, y_true_hierarchized)
    return flat_hi_report


def normal_metrics(y_hier_pred, y_hier_true):
    new_y_pred = []
    for list_label in y_hier_pred :
        new_list_label = convert_hierarchy_to_label(list_label)
        #print(list_label, new_list_label)
        new_y_pred.append(new_list_label.tolist())

    #print(new_y_pred)
    #print(len(new_y_pred))
    #new_y_pred = np.array(new_y_pred, dtype=object)
    y_true_last = extract_last_element(y_hier_true)
    #print(y_true_last)
    #print(len(y_true_last))
    # Combine y_true_last and y_pred_last to get all unique classes
    all_classes = set(y_true_last + new_y_pred)
    #print(len(all_classes))

    # Sort the classes alphabetically
    class_names = sorted(all_classes)

    # Calculate accuracy
    accuracy = accuracy_score(y_true_last, new_y_pred)
    print("Accuracy:", accuracy)

    # Generate classification report
    report = classification_report(y_true_last, new_y_pred, target_names=class_names, zero_division=0, output_dict=True)

    #print("Classification Report:\n", report)

    #plt.figure(figsize=(15, 15))
    #ConfusionMatrixDisplay.from_predictions(y_true_last, y_pred_last, ax=plt.gca(), xticks_rotation=90)

    #plt.show()
    return report


def get_hierarchical_scores_from_hiclass(y_hier_pred, y_hier_true):
    f1_score_val = f1(y_hier_true, y_hier_pred)
    precision_score_val = precision(y_hier_true, y_hier_pred)
    recall_score_val = recall(y_hier_true, y_hier_pred)
    hi_report = {
        'precision': precision_score_val,
        'recall': recall_score_val,
        'f1-score': f1_score_val,
        'support': len(y_hier_pred)
    }
    return hi_report


def metrics_intermediate_labels(y_true, y_pred, y_train, intermediate_labels):
    intermediate_metrics = {}

    for label in intermediate_labels:
        intermediate_metrics[label] = {}
        intermediate_metrics[label]['y_true'] = []
        intermediate_metrics[label]['y_pred'] = []

    for true_labels, pred_labels in zip(y_true, y_pred):
        for label in intermediate_labels:
            if label in true_labels or label in pred_labels:
                intermediate_metrics[label]['y_true'].append(label in true_labels)
                intermediate_metrics[label]['y_pred'].append(label in pred_labels)

    for label in intermediate_labels:
        y_true_intermediate = intermediate_metrics[label]['y_true']
        y_pred_intermediate = intermediate_metrics[label]['y_pred']

        intermediate_metrics[label]['accuracy'] = accuracy_score(y_true_intermediate, y_pred_intermediate)
        intermediate_metrics[label]['recall'] = recall_score(y_true_intermediate, y_pred_intermediate)
        intermediate_metrics[label]['precision'] = precision_score(y_true_intermediate, y_pred_intermediate, zero_division=0)
        intermediate_metrics[label]['f1_score'] = f1_score(y_true_intermediate, y_pred_intermediate)
        intermediate_metrics[label]['test_support'] = y_true_intermediate.count(True)
        intermediate_metrics[label]['test_pred'] = count_label_occurrences(y_pred, label)
        intermediate_metrics[label]['train_support'] = count_label_occurrences(y_train, label)

        del intermediate_metrics[label]['y_true']
        del intermediate_metrics[label]['y_pred']

    return intermediate_metrics