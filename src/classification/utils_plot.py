import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

def plot_heatmap(data: pd.DataFrame, limit: int = None):
    # Create unique labels by appending `ili` to `example_en_lemma`
    data['unique_label'] = data['example_en_lemma'] + " (" + data['ili'] + ")"

    # Pivot the data for plotting using unique labels as the index, languages as the columns,
    # and basicness_score as the values of the columns
    pivoted_data = data.pivot(index='unique_label', columns='Language', values='basicness_score')

    if limit is not None and limit < pivoted_data.shape[0]:
        pivoted_data = pivoted_data.head(limit)

    n_rows = pivoted_data.shape[0]

    if n_rows < 30:
        figure_height = 0.5 * n_rows
    else:
        figure_height = 0.25 * n_rows

    # Plotting the heatmap
    plt.figure(figsize=(12,figure_height))  # Adjust the figure size for readability
    sns.heatmap(pivoted_data, annot=True, cmap="Blues", cbar_kws={'label': 'Basicness Score'}, vmin=0, vmax=1)
    plt.title('Basicness Score for Synsets Grouped by ILI (Unique Labels) Across Languages')
    plt.xlabel('Language')
    plt.ylabel('Example English Lemma (with ILI)')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_spaghettiplot(data, limit=None, show_legend=True):
    """
    Plots a spaghetti plot of basicness scores across languages for each ILI in the data.

    Parameters:
    - data: The dataframe containing 'Language', 'basicness_score', 'ili', and 'example_en_lemma' columns.
    - limit: The maximum number of ILIs to plot. If None, plot all ILIs, Default None.
    - show_legend: Whether to display the legend. Default True.
    """
    # Set ordered categories for Language to maintain consistent order
    ordered_languages = ["en", "it", "es", "nb"]
    data['Language'] = pd.Categorical(data['Language'], categories=ordered_languages, ordered=True)

    # Sort data to ensure each ILI is in order of 'Language'
    data = data.sort_values(['ili', 'Language'])

    # Limit the number of ILIs if a limit is specified
    unique_ilis = data['ili'].unique()
    if limit is not None and limit < len(unique_ilis):
        ilis_to_plot = unique_ilis[:limit]
        data = data[data['ili'].isin(ilis_to_plot)]

    plt.figure(figsize=(12, 8))

    # Plot points and lines for each unique `ili`, using `example_en_lemma` for the legend
    for ili, ili_data in data.groupby('ili'):
        # Get the `example_en_lemma` for the legend
        example_en_lemma = ili_data['example_en_lemma'].iloc[0]

        # Ensure the data includes all languages, filling missing data with NaN
        ili_data = ili_data.set_index('Language').reindex(ordered_languages).reset_index()

        # Plot the line with label as the `example_en_lemma`
        plt.plot(ili_data['Language'], ili_data['basicness_score'], marker='o', linestyle='-', label=example_en_lemma,
                 alpha=0.6)

    plt.xlabel("Language")
    plt.ylabel("Basicness Score")
    plt.title("Basicness Score Across Languages by ILI (Ordered with Gaps)")

    # Conditionally display the legend
    if show_legend:
        plt.legend(title="Example English Lemma", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.show()

def plot_clustermap(data: pd.DataFrame, limit: int = None, use_rank: bool = True):
    # Create unique labels by appending `ili` to `example_en_lemma`
    data['unique_label'] = data['example_en_lemma'] + " (" + data['ili'] + ")"

    if use_rank:
        column_name = 'basicness_rank'
        label = 'Basicness Rank'
        subtitle_word = 'Rank'
        vmin, vmax = 1, 4  # Rank ranges from 1 to 4
    else:
        column_name = 'basicness_score'
        label = 'Basicness Score'
        subtitle_word = 'Score'
        vmin, vmax = data[column_name].min(), data[column_name].max()  # Adjust based on actual score range

    # Pivot the data for plotting using unique labels as the index, languages as the columns,
    # and basicness_rank as the values of the columns
    pivoted_data = data.pivot(index='unique_label', columns='Language', values=column_name)

    # Apply the limit if specified
    if limit is not None and limit < pivoted_data.shape[0]:
        pivoted_data = pivoted_data.head(limit)

    n_rows = pivoted_data.shape[0]

    # Adjust figure size dynamically based on the number of rows for readability
    if n_rows < 30:
        figure_height = 0.5 * n_rows
        dendrogram_ratio = (0.1, 0.2)
        cbar_pos = (-0.1, 0.8, 0.02, 0.30)
    else:
        figure_height = 0.30 * n_rows
        dendrogram_ratio = (0.2, 0.04)
        cbar_pos = (-0.1, 0.9, 0.02, 0.10)

    # Plotting the clustermap with adjusted size using ranks
    sns.clustermap(pivoted_data, cmap="Blues", annot=True, cbar_kws={'label': label},
                   vmin=vmin, vmax=vmax, cbar_pos=cbar_pos,
                   dendrogram_ratio=dendrogram_ratio, figsize=(12, figure_height))
    plt.suptitle(f'Clustered Basicness {subtitle_word} for Synsets Grouped by ILI Across Languages', y=1.05)
    plt.show()