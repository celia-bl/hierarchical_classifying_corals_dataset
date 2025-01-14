import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_label_distribution(y):
    # Convertir y en un tableau numpy pour faciliter le traitement
    y_array = np.array(y)

    # Calculer la distribution des labels
    unique_labels, counts = np.unique(y_array, return_counts=True)

    # Trier les labels par nombre d'occurrences (de manière décroissante)
    sorted_indices = np.argsort(counts)[::-1]  # Tri décroissant
    sorted_labels = unique_labels[sorted_indices]
    sorted_counts = counts[sorted_indices]

    # Calculer la skewness et la kurtosis
    skewness = stats.skew(sorted_counts)
    kurtosis = stats.kurtosis(sorted_counts)

    # Créer l'histogramme
    plt.figure(figsize=(20, 6))
    plt.bar(sorted_labels, sorted_counts, color='skyblue', edgecolor='black')

    # Ajouter des annotations pour skewness et kurtosis
    textstr = '\n'.join((
        f'Skewness: {skewness:.2f}',
        f'Kurtosis: {kurtosis:.2f}'
    ))

    # Ajouter une annotation dans le graphique
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
                   fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.xlabel('Labels')
    plt.ylabel('Number of Occurrences')
    plt.title('Distribution of Labels in y_test')
    plt.xticks(sorted_labels)  # Afficher les labels sur l'axe des x
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show(block=False)


def plot_avg_distrib(average_label_counts):
    # Extraire les labels et les moyennes des occurrences
    labels = list(average_label_counts.keys())
    avg_counts = list(average_label_counts.values())

    # Convertir en arrays pour faciliter le traitement
    labels_array = np.array(labels)
    avg_counts_array = np.array(avg_counts)

    # Trier les labels par nombre moyen d'occurrences (de manière décroissante)
    sorted_indices = np.argsort(avg_counts_array)[::-1]  # Tri décroissant
    sorted_labels = labels_array[sorted_indices]
    sorted_avg_counts = avg_counts_array[sorted_indices]

    # Calculer la skewness et la kurtosis
    skewness = stats.skew(sorted_avg_counts)
    kurtosis = stats.kurtosis(sorted_avg_counts)

    # Créer l'histogramme
    plt.figure(figsize=(20, 6))
    plt.bar(sorted_labels, sorted_avg_counts, color='skyblue', edgecolor='black')

    # Ajouter des annotations pour skewness et kurtosis
    textstr = '\n'.join((
        f'Skewness: {skewness:.2f}',
        f'Kurtosis: {kurtosis:.2f}'
    ))

    # Ajouter une annotation dans le graphique
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
                   fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.xlabel('Labels')
    plt.ylabel('Average Number of Occurrences')
    plt.title('Average Distribution of Labels Across Runs')
    plt.xticks(sorted_labels)  # Afficher les labels sur l'axe des x
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show(block=False)