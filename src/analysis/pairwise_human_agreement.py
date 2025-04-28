"""
Author: Federico Torrielli
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
import itertools
import os

# Configuration
HUMAN_PATHS = [
    "culture_annotation_datasets_annotated_human_it.csv",  # Italian
    "culture_annotation_datasets_annotated_human_nb.csv",  # Norwegian
    "culture_annotation_datasets_annotated_human_en.csv",  # English
    "culture_annotation_datasets_annotated_human_es.csv",  # Spanish
]
OUTPUT_DIR = "pairwise_human_agreement_results"
OUTPUT_LATEX = os.path.join(OUTPUT_DIR, "pairwise_human_agreement_table.tex")
OUTPUT_SUMMARY = os.path.join(OUTPUT_DIR, "pairwise_human_agreement_summary.png")

# Language colors for visualizations
LANGUAGE_COLORS = {
    "it": "#008C45",  # Italian - Green
    "nb": "#00205B",  # Norwegian - Dark Blue
    "en": "#CF142B",  # English - Red
    "es": "#F1BF00",  # Spanish - Yellow
}

# Language full names
LANGUAGE_NAMES = {
    "it": "Italian",
    "nb": "Norwegian",
    "en": "English",
    "es": "Spanish",
}


def load_human_data():
    """
    Load all human annotation datasets and combine them into a single dataframe
    with language identifiers
    """
    all_data = {}
    
    for path in HUMAN_PATHS:
        # Extract language code from filename
        lang_code = path.split('_')[-1].split('.')[0]
        
        # Load data
        df = pd.read_csv(path)
        
        # Create a language super-annotator by averaging scores
        df['super_annotator'] = df[['annotator_1', 'annotator_2', 'annotator_3']].mean(axis=1)
        
        # Store data with language code
        all_data[lang_code] = df
    
    return all_data


def combine_data_for_pairwise(all_data):
    """
    Combine all language datasets into a single DataFrame for pairwise comparison,
    merging on the ILI identifier to match concepts across languages.
    """
    # Start with a DataFrame containing just ILI identifiers
    combined_df = pd.DataFrame({'ili': next(iter(all_data.values()))['ili']})
    
    # Add super-annotator scores for each language
    for lang, df in all_data.items():
        # Add super-annotator scores with language prefix
        combined_df = combined_df.merge(
            df[['ili', 'super_annotator']].rename(
                columns={'super_annotator': f'score_{lang}'}
            ),
            on='ili',
            how='outer'  # Use outer join to include all concepts
        )
    
    return combined_df


def calculate_pairwise_metrics(combined_df):
    """
    Calculate pairwise agreement metrics between all language pairs
    """
    # Get score columns
    score_cols = [col for col in combined_df.columns if col.startswith('score_')]
    
    # Get all possible pairs of languages
    pairs = list(itertools.combinations(score_cols, 2))
    
    results = []
    
    for col1, col2 in pairs:
        # Extract language codes from column names
        lang1 = col1.replace('score_', '')
        lang2 = col2.replace('score_', '')
        
        # Extract ratings with no missing values
        pair_df = combined_df[[col1, col2]].dropna()
        
        if len(pair_df) < 5:  # Skip pairs with too few items
            continue
            
        # Calculate Cohen's Kappa (with linear weights for ordinal data)
        # Need to round to integers for kappa calculation
        kappa = cohen_kappa_score(
            pair_df[col1].round().astype(int), 
            pair_df[col2].round().astype(int), 
            weights="linear"
        )
        
        # Calculate Spearman correlation
        corr = spearmanr(pair_df[col1], pair_df[col2])[0]
        
        # Calculate binary agreement (1-2 → 0, 3-4 → 1)
        pair_df_binary = pair_df.copy()
        pair_df_binary[col1] = pair_df_binary[col1].apply(lambda x: 0 if x <= 2 else 1)
        pair_df_binary[col2] = pair_df_binary[col2].apply(lambda x: 0 if x <= 2 else 1)
        
        # Binary kappa
        binary_kappa = cohen_kappa_score(
            pair_df_binary[col1].astype(int), 
            pair_df_binary[col2].astype(int)
        )
        
        # Raw agreement percentage
        binary_agreement = np.mean(pair_df_binary[col1] == pair_df_binary[col2])
        
        # Mean squared error
        mse = np.mean((pair_df[col1] - pair_df[col2])**2)
        
        results.append({
            "lang1": lang1,
            "lang2": lang2,
            "kappa": kappa,
            "correlation": corr,
            "binary_kappa": binary_kappa,
            "binary_agreement": binary_agreement,
            "mse": mse,
            "n_items": len(pair_df)
        })
    
    return pd.DataFrame(results)


def generate_latex_table(results_df):
    """Generate a LaTeX table from the results dataframe"""
    # Create full document with proper packages
    latex = "% This is a standalone LaTeX document\n"
    latex += "% To include in another document, copy everything between 'begin{table}' and 'end{table}'\n"
    latex += "% and ensure your document has \\usepackage{booktabs} in the preamble\n"
    latex += "\\documentclass{article}\n"
    latex += "\\usepackage[utf8]{inputenc}\n"
    latex += "\\usepackage{booktabs}\n"
    latex += "\\usepackage{amsmath}\n"
    latex += "\\begin{document}\n\n"
    
    # Create table
    latex += "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += "\\caption{Pairwise Inter-Language Agreement Metrics}\n"
    latex += "\\label{tab:pairwise_language_agreement}\n"
    latex += "\\begin{tabular}{lcccc}\n"
    latex += "\\toprule\n"
    latex += "Language Pair & Cohen's $\\kappa$ & Spearman $\\rho$ & Agreement & $n$ \\\\\n"
    latex += "\\midrule\n"
    
    # Add rows
    for _, row in results_df.iterrows():
        # Use full language names
        lang1 = LANGUAGE_NAMES.get(row['lang1'], row['lang1'])
        lang2 = LANGUAGE_NAMES.get(row['lang2'], row['lang2'])
        lang_pair = f"{lang1} -- {lang2}"
        
        # Escape percentages and format data
        binary_agreement = f"{row['binary_agreement']:.1%}".replace("%", "\\%")
        
        latex += f"{lang_pair} & {row['kappa']:.3f} & {row['correlation']:.3f} & {binary_agreement} & {row['n_items']} \\\\\n"
    
    # Add footer
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n\n"
    
    # Close document
    latex += "\\end{document}\n"
    
    # Also save a standalone version for easier compilation testing
    with open(os.path.join(OUTPUT_DIR, "latex_table_standalone.tex"), "w") as f:
        f.write(latex)
    
    # For the main return value, just return the table portion without the document wrapper
    table_only = "\n".join(latex.split('\n')[8:-3])  # Extract just the table part
    return table_only


def create_heatmap(results_df, metric, title, ax=None):
    """Create a heatmap for a specific metric"""
    # Get unique languages
    languages = sorted(list(set(results_df['lang1'].unique()) | set(results_df['lang2'].unique())))
    n_languages = len(languages)
    
    # Create matrix for heatmap
    matrix = np.zeros((n_languages, n_languages))
    
    # Fill matrix with metric values
    for _, row in results_df.iterrows():
        i = languages.index(row['lang1'])
        j = languages.index(row['lang2'])
        matrix[i, j] = row[metric]
        matrix[j, i] = row[metric]  # Mirror since it's symmetric
    
    # Set diagonal to 1 (or appropriate value)
    for i in range(n_languages):
        if metric in ['kappa', 'correlation', 'binary_kappa', 'binary_agreement']:
            matrix[i, i] = 1.0
        elif metric == 'mse':
            matrix[i, i] = 0.0
    
    # Use provided axes or create a new figure
    if ax is None:
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
    
    if metric == 'mse':
        # For MSE, lower is better
        cmap = 'YlOrRd_r'
        vmin, vmax = 0, max(0.1, matrix.max())
        avg_text = f"Mean MSE: {np.mean(matrix[~np.eye(n_languages, dtype=bool)]):.3f}"
    else:
        # For other metrics, higher is better
        cmap = 'YlGnBu'
        vmin, vmax = matrix.min() if matrix.min() > 0 else 0, 1
        
        # Calculate average excluding diagonal
        mask = ~np.eye(n_languages, dtype=bool)  # Mask to exclude diagonal
        avg_value = np.mean(matrix[mask])
        
        if metric == 'binary_agreement':
            avg_text = f"Mean: {avg_value:.1%}"
        else:
            avg_text = f"Mean: {avg_value:.3f}"
    
    # Enhanced title with average value
    full_title = f"{title}\n{avg_text}"
    
    # Convert language codes to full names
    pretty_labels = [LANGUAGE_NAMES.get(lang, lang) for lang in languages]
    
    # Create heatmap
    hm = sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f" if metric != 'binary_agreement' else ".1%",
        cmap=cmap,
        xticklabels=pretty_labels,
        yticklabels=pretty_labels,
        vmin=vmin,
        vmax=vmax,
        ax=ax
    )
    
    # Improve readability of labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    # Add title with average value
    ax.set_title(full_title)
    
    # Adjust colorbar (optional)
    cbar = hm.collections[0].colorbar
    if metric == 'binary_agreement':
        cbar.set_label('Agreement %')
    elif metric == 'mse':
        cbar.set_label('Mean Squared Error')
    else:
        cbar.set_label(title)
    
    return ax


def create_radar_chart(results_df, metric, title):
    """Create a radar chart comparing languages on a specific metric"""
    # Get unique languages
    languages = sorted(list(set(results_df['lang1'].unique()) | set(results_df['lang2'].unique())))
    n_languages = len(languages)
    
    # Calculate average metric value for each language
    lang_metrics = {}
    for lang in languages:
        # Get all rows involving this language
        rows = results_df[(results_df['lang1'] == lang) | (results_df['lang2'] == lang)]
        lang_metrics[lang] = rows[metric].mean()
    
    # Convert to full language names
    lang_names = [LANGUAGE_NAMES.get(lang, lang) for lang in languages]
    metrics = [lang_metrics[lang] for lang in languages]
    
    # Create radar chart
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # Compute angle for each language
    angles = np.linspace(0, 2*np.pi, n_languages, endpoint=False)
    
    # Close the polygon
    metrics = np.concatenate((metrics, [metrics[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    lang_names = lang_names + [lang_names[0]]
    
    # Plot data
    ax.plot(angles, metrics, 'o-', linewidth=2)
    
    # Fill area
    ax.fill(angles, metrics, alpha=0.25)
    
    # Set labels
    ax.set_thetagrids(angles[:-1] * 180/np.pi, lang_names[:-1])
    
    # Set title
    ax.set_title(title)
    
    # Set y limits based on metric
    if metric == 'mse':
        ax.set_ylim(0, max(metrics) * 1.1)  # Lower is better for MSE
    else:
        ax.set_ylim(0, 1)  # Higher is better for other metrics
    
    plt.tight_layout()
    return fig


def create_bar_chart(results_df, metric, title):
    """Create a bar chart comparing language pairs on a specific metric"""
    plt.figure(figsize=(12, 6))
    
    # Convert language codes to full names for the x-axis
    results_df = results_df.copy()
    results_df['pair'] = results_df.apply(
        lambda row: f"{LANGUAGE_NAMES.get(row['lang1'], row['lang1'])} - {LANGUAGE_NAMES.get(row['lang2'], row['lang2'])}",
        axis=1
    )
    
    # Sort by metric value for better visualization
    results_df = results_df.sort_values(by=metric, ascending=(metric == 'mse'))
    
    # Create color mapping based on language pairs
    colors = []
    for _, row in results_df.iterrows():
        # Mix colors of the two languages
        color1 = LANGUAGE_COLORS.get(row['lang1'], '#CCCCCC')
        color2 = LANGUAGE_COLORS.get(row['lang2'], '#CCCCCC')
        colors.append(color1 if row.name % 2 == 0 else color2)
    
    # Create bar chart
    ax = sns.barplot(x='pair', y=metric, data=results_df, palette=colors)
    
    # Add value labels on top of bars
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        if metric == 'binary_agreement':
            label = f"{height:.1%}"
        else:
            label = f"{height:.3f}"
        ax.text(p.get_x() + p.get_width()/2., height + 0.01,
                label, ha="center", va="bottom")
    
    # Rotate x labels for readability
    plt.xticks(rotation=45, ha='right')
    
    # Add title and labels
    plt.title(title)
    plt.ylabel(metric.replace('_', ' ').title())
    plt.xlabel('Language Pair')
    
    plt.tight_layout()
    return plt.gcf()


def generate_readme(results_df):
    """Generate a README.md file explaining the analysis and files"""
    readme_text = """# Pairwise Inter-Language Agreement Analysis

This directory contains the results of pairwise agreement analysis between human annotators grouped by language.

## Approach

For each language (English, Italian, Spanish, Norwegian), we:
1. Averaged the scores of the three annotators to create a "super-annotator" for that language
2. Calculated agreement metrics between each pair of languages
3. Generated visualizations to compare agreement patterns

## Files Overview

- **pairwise_human_agreement_summary.png**: Combined visualization of all agreement metrics
- **heatmap_kappa.png**: Heatmap of Cohen's Kappa scores between language pairs
- **heatmap_correlation.png**: Heatmap of Spearman correlation coefficients between language pairs
- **heatmap_binary_agreement.png**: Heatmap of binary agreement percentages (after converting 1-2→0, 3-4→1)
- **heatmap_mse.png**: Heatmap of Mean Squared Error between language pairs
- **radar_chart_kappa.png**: Radar chart showing each language's average agreement with others
- **bar_chart_kappa.png**: Bar chart comparing language pairs by agreement metrics
- **pairwise_human_agreement_table.tex**: LaTeX code for a table of pairwise metrics
- **latex_table_standalone.tex**: Complete LaTeX document containing the table (can be compiled independently)
- **summary_statistics.csv**: CSV file with summary statistics across all language pairs
- **full_results.csv**: Complete results for all language pairs with all metrics

## Summary Statistics

"""
    # Add summary statistics
    readme_text += f"- Number of language pairs analyzed: {len(results_df)}\n"
    readme_text += f"- Average Cohen's Kappa: {results_df['kappa'].mean():.3f}\n"
    readme_text += f"- Average Spearman Correlation: {results_df['correlation'].mean():.3f}\n"
    readme_text += f"- Average Agreement: {results_df['binary_agreement'].mean():.1%}\n\n"
    
    # Add language pairs ranked by agreement
    readme_text += "## Language Pairs Ranked by Agreement (Kappa)\n\n"
    readme_text += "| Language Pair | Cohen's Kappa | Spearman Correlation | Agreement |\n"
    readme_text += "|---------------|--------------|----------------------|------------------|\n"
    
    sorted_results = results_df.sort_values(by='kappa', ascending=False)
    for _, row in sorted_results.iterrows():
        lang1 = LANGUAGE_NAMES.get(row['lang1'], row['lang1'])
        lang2 = LANGUAGE_NAMES.get(row['lang2'], row['lang2'])
        readme_text += f"| {lang1} - {lang2} | {row['kappa']:.3f} | {row['correlation']:.3f} | {row['binary_agreement']:.1%} |\n"
    
    # Add explanation of metrics
    readme_text += """
## Metrics Explanation

- **Cohen's Kappa**: Measures inter-rater agreement taking into account chance agreement. Values range from -1 to 1, with higher values indicating better agreement. Values above 0.8 are considered very good agreement.

- **Spearman Correlation**: Measures how well the relationship between two variables can be described using a monotonic function. Values range from -1 to 1, with higher values indicating stronger positive correlation.

- **Agreement**: Percentage of items where both languages give the same binary classification after converting ratings (1-2→0, 3-4→1).

- **Mean Squared Error (MSE)**: Average of the squared differences between corresponding elements. Lower values indicate better agreement.

## Notes on Interpretation

- High agreement between languages suggests consistent conceptual understanding across cultures.
- Binary agreement often shows higher values than the more stringent Kappa or correlation metrics, as it's less sensitive to exact rating differences.
- Differences in agreement may reflect cultural differences in basic-level categorization.
"""
    
    # Write to file
    readme_path = os.path.join(OUTPUT_DIR, "README.md")
    with open(readme_path, 'w') as f:
        f.write(readme_text)
    
    return readme_path


def main():
    """Main function to run the pairwise agreement analysis"""
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")
    
    print("Loading human annotation datasets...")
    all_data = load_human_data()
    
    print("Combining data for pairwise comparison...")
    combined_df = combine_data_for_pairwise(all_data)
    
    print("Calculating pairwise metrics...")
    results_df = calculate_pairwise_metrics(combined_df)
    
    # Display summary
    print("\nPairwise Agreement Summary:")
    print(f"Number of language pairs: {len(results_df)}")
    print(f"Average Cohen's Kappa: {results_df['kappa'].mean():.3f}")
    print(f"Average Spearman Correlation: {results_df['correlation'].mean():.3f}")
    print(f"Average Agreement: {results_df['binary_agreement'].mean():.1%}")
    
    # Save summary statistics as CSV
    summary_df = pd.DataFrame({
        'metric': ['Average Cohen\'s Kappa', 'Average Spearman Correlation', 'Average Agreement', 'Pairs Analyzed'],
        'value': [results_df['kappa'].mean(), results_df['correlation'].mean(), results_df['binary_agreement'].mean(), len(results_df)]
    })
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'summary_statistics.csv'), index=False)
    
    # Save full results as CSV for further analysis
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'full_results.csv'), index=False)
    
    # Generate LaTeX table
    latex_table = generate_latex_table(results_df)
    with open(OUTPUT_LATEX, 'w') as f:
        f.write(latex_table)
    print(f"\nLaTeX table saved to {OUTPUT_LATEX}")
    
    # Create summary visualization with all metrics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    create_heatmap(results_df, 'kappa', "Cohen's Kappa", axes[0, 0])
    create_heatmap(results_df, 'correlation', "Spearman Correlation", axes[0, 1])
    create_heatmap(results_df, 'binary_agreement', "Agreement", axes[1, 0])
    create_heatmap(results_df, 'mse', "Mean Squared Error", axes[1, 1])
    
    plt.suptitle("Pairwise Agreement Between Languages", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for super title
    plt.savefig(OUTPUT_SUMMARY, dpi=300)
    print(f"Summary visualizations saved to {OUTPUT_SUMMARY}")
    
    # Save individual heatmaps in higher resolution
    metrics = {
        'kappa': "Cohen's Kappa",
        'correlation': "Spearman Correlation",
        'binary_agreement': "Agreement (%)",
        'mse': "Mean Squared Error"
    }
    
    for metric, title in metrics.items():
        fig, ax = plt.subplots(figsize=(10, 8))
        create_heatmap(results_df, metric, title, ax)
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, f"heatmap_{metric}.png")
        plt.savefig(output_path, dpi=300)
        print(f"Saved {metric} heatmap to {output_path}")
    
    # Generate radar charts
    for metric, title in metrics.items():
        radar_fig = create_radar_chart(results_df, metric, f"Language Comparison - {title}")
        output_path = os.path.join(OUTPUT_DIR, f"radar_chart_{metric}.png")
        radar_fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved radar chart for {metric} to {output_path}")
    
    # Generate bar charts
    for metric, title in metrics.items():
        bar_fig = create_bar_chart(results_df, metric, f"Language Pair Comparison - {title}")
        output_path = os.path.join(OUTPUT_DIR, f"bar_chart_{metric}.png")
        bar_fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved bar chart for {metric} to {output_path}")
    
    # Generate and save README
    readme_path = generate_readme(results_df)
    print(f"Generated README file at {readme_path}")
    
    print(f"\nAll results have been saved to the '{OUTPUT_DIR}' directory")


if __name__ == "__main__":
    main() 