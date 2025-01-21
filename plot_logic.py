import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#import graphviz
import plotly.express as px
import random
from load_data import get_image_labels_from_feature, get_features_from_label_type


# ----------------------- METRICS ------------------------------------------
def plot_metrics_by_nb_images(flat_reports, flat_hi_reports, hi_flat_reports, hi_reports):
    plt.figure(figsize=(15, 9))

    for reports, report_type in zip([flat_reports, hi_flat_reports], ['Flat Classifier (baseline)', 'Hierarchical Classifier (ours)']):
        for classifier_name, nb_images in reports.items():
            nb_images_sorted = sorted(nb_images.items())
            nb_images_list, f1_scores, f1_stds = zip(
                *[(nb_image, metrics.get('f1-score', None), metrics.get('std_f1-score', None)) for nb_image, metrics in
                  nb_images_sorted])
            nb_images_list = [x for x in nb_images_list]

            plt.errorbar(nb_images_list, f1_scores, yerr=f1_stds, label=f'{report_type}',
                         marker='o', markersize=8, capsize=5, linewidth=4)
    plt.xlabel('Number of training patches', fontsize=36)
    plt.ylabel('F1-score'.capitalize(), fontsize=36)
    #plt.title(f'{"F1-score".capitalize()} Bottom-up vs Top-down')
    plt.legend(fontsize=34)
    plt.grid(True)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.tight_layout()
    plt.savefig('f1_score_plot_rio.png')
    plt.show(block=False)

    plt.figure(figsize=(15, 9))
    for reports, report_type in zip([flat_hi_reports, hi_reports], ['Flat Classifier (baseline)', 'Hierarchical Classifier (ours)']):
        for classifier_name, nb_images in reports.items():
            nb_images_sorted = sorted(nb_images.items())
            nb_images_list, f1_scores, f1_stds = zip(
                *[(nb_image, metrics.get('f1-score', None), metrics.get('std_f1-score', None)) for nb_image, metrics in
                  nb_images_sorted])
            nb_images_list = [x for x in nb_images_list]

            plt.errorbar(nb_images_list, f1_scores, yerr=f1_stds, label=f'{report_type}',
                         marker='o', markersize=8, capsize=5, linewidth=4)
    plt.xlabel('Number of training patches', fontsize=36)
    plt.ylabel('Hierarchical F1-score'.capitalize(), fontsize=36)
    #plt.title(f'{"Hierarchical F1-score".capitalize()} Bottom-up vs top-down')
    plt.legend(fontsize=34)
    plt.grid(True)

    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)

    plt.tight_layout()
    plt.savefig('hierarchical_fi_score_plot_rio.png')
    plt.show(block=False)


def plot_intermediate_metrics(flat_intermediate_reports, hi_intermediate_reports):
    # Créer une liste pour collecter les données
    rows = []

    # Itérer sur data_dict_flat
    for classifier_flat, data_flat in flat_intermediate_reports.items():
        for image_count, metrics_flat in data_flat.items():
            for label, metrics_data in metrics_flat.items():
                row = {
                    'Classifier': classifier_flat,
                    'Image Count': image_count,
                    'Label': label,
                    'Precision': metrics_data['precision'],
                    'Recall': metrics_data['recall'],
                    'F1-score': metrics_data['f1-score'],
                    'Std Precision': metrics_data['std_precision'],
                    'Std Recall': metrics_data['std_recall'],
                    'Std F1-score': metrics_data['std_f1-score']
                }
                rows.append(row)

    # Itérer sur data_dict_2
    for classifier_2, data_2 in hi_intermediate_reports.items():
        for image_count, metrics_2 in data_2.items():
            for label, metrics_data in metrics_2.items():
                row = {
                    'Classifier': classifier_2,
                    'Image Count': image_count,
                    'Label': label,
                    'Precision': metrics_data['precision'],
                    'Recall': metrics_data['recall'],
                    'F1-score': metrics_data['f1-score'],
                    'Std Precision': metrics_data['std_precision'],
                    'Std Recall': metrics_data['std_recall'],
                    'Std F1-score': metrics_data['std_f1-score']
                }
                rows.append(row)

    # Créer un DataFrame à partir des données collectées
    df = pd.DataFrame(rows)

    # Tri des classifiers uniques
    classifiers = df['Classifier'].unique()

    # Boucle sur chaque nombre d'images
    for image_count in sorted(set(df['Image Count'])):
        fig, axes = plt.subplots(3, 1, figsize=(15, 25), sharey=True)

        # Boucle sur chaque métrique
        for i, metric in enumerate(['Precision', 'Recall', 'F1-score']):
            ax = axes[i]

            # Scatter plot pour chaque classifier
            for classifier in classifiers:
                df_filtered = df[(df['Image Count'] == image_count) & (df['Classifier'] == classifier)]

                sns.scatterplot(ax=ax, data=df_filtered, x='Label', y=metric, label=classifier, markers=True)

                # Ajout des barres d'erreur
                ax.errorbar(x=df_filtered['Label'], y=df_filtered[metric],
                            yerr=df_filtered[f'Std {metric}'], fmt='o', capsize=5)

            ax.set_title(f'{metric} Comparison (Images: {image_count})')
            ax.set_xlabel('Label')
            ax.set_ylabel(metric)
            ax.legend(title='Classifier', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True)

        plt.tight_layout()
        plt.show(block=False)


# ---------------------------------------- VISUALIZE HIERARCHY AND DATASET ------------------------------------------------
'''
def plot_hierarchy_graphviz(hierarchy, root_name="Root"):
    """
    Affiche la hiérarchie avec Graphviz, en ajoutant un nœud racine explicite s'il n'est pas déjà présent.

    Args:
        hierarchy (dict): Dictionnaire représentant la hiérarchie.
        root_name (str): Nom du nœud racine.
    """
    # Création du graphe
    dot = graphviz.Digraph(format="png")
    dot.attr(rankdir="LR")  # Orientation gauche-droite ou changez par "TB" pour top-bottom

    # Ajout du nœud racine
    if root_name not in hierarchy.values():
        dot.node(root_name)

    # Ajouter tous les nœuds et arêtes
    for child, parent in hierarchy.items():
        if parent is None:  # Si le nœud n'a pas de parent, connectez-le à la racine
            dot.edge(root_name, child)
        else:
            dot.edge(parent, child)

    # Afficher ou sauvegarder
    dot.render("hierarchy_graph", view=True)  # Sauvegarde et ouvre le fichier "hierarchy_graph.png"
'''

def plot_hierarchy_sunburst(hierarchy, root="root"):
    """
    Plotte une hiérarchie de deux façons :
    1. Un arbre hiérarchique non-orienté avec Graphviz.
    2. Un diagramme en sunburst avec Plotly Express.

    Arguments :
    - hierarchy : dict, la hiérarchie sous forme de dictionnaire {noeud: parent}.
    - root : str, le noeud racine pour Graphviz (facultatif).
    """
    # Convertir la hiérarchie en une liste de parent-enfant
    edges = [(parent, child) for child, parent in hierarchy.items() if parent is not None]
    # Préparer les données pour Plotly
    data = [{"parent": parent if parent else root, "child": child} for child, parent in hierarchy.items()]
    df = pd.DataFrame(data)

    # Créer le sunburst
    fig = px.sunburst(
        df,
        names="child",
        parents="parent",
        title="Diagramme Sunburst de la hiérarchie",
    )
    fig.show(block=False)


def plot_hierarchy_sunburst_balanced(hierarchy, occurrences, root="root"):
    """
    Arguments :
    - hierarchy : dict, la hiérarchie sous forme de dictionnaire {noeud: parent}.
    - occurrences : dict, le nombre d'occurrences pour chaque label, sous la forme {label: nombre_occurences}.
    - root : str, le noeud racine pour Graphviz (facultatif).
    """
    # Convertir la hiérarchie en une liste de parent-enfant
    edges = [(parent, child) for child, parent in hierarchy.items() if parent is not None]

    # Préparer les données pour Plotly en incluant le nombre d'occurrences
    data = []
    for child, parent in hierarchy.items():
        data.append({
            "parent": parent if parent else root,
            "child": child,
            "size": occurrences.get(child, 0)  # Utiliser 0 si l'occurrence n'est pas dans le dictionnaire
        })

    df = pd.DataFrame(data)

    # Créer le Sunburst avec les tailles proportionnelles aux occurrences
    fig = px.sunburst(
        df,
        names="child",
        parents="parent",
        values="size",  # Attribuer la taille des catégories en fonction des occurrences
        title="Diagramme Sunburst de la hiérarchie",
    )

    fig.show()


def plot_occurences(occurences):
    # Décomposer les labels et leurs occurrences
    labels, counts = zip(*occurences.items())

    # Trier les labels par nombre d'occurrences
    sorted_indices = sorted(range(len(counts)), key=lambda i: counts[i], reverse=True)
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_counts = [counts[i] for i in sorted_indices]

    # Créer l'histogramme
    plt.figure(figsize=(25, 10))
    bars = plt.bar(sorted_labels, sorted_counts, color='skyblue')
    plt.xlabel('Labels')
    plt.ylabel('Occurrences')
    plt.title('Histogramme des Labels')
    plt.xticks(ticks=range(len(sorted_labels)), labels=sorted_labels, rotation=45)

    # Ajouter le nombre au-dessus de chaque barre
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval),
                 ha='center', va='bottom')  # Centre le texte et le place au-dessus de la barre

    plt.tight_layout()  # Ajuster les marges
    plt.show(block=False)
