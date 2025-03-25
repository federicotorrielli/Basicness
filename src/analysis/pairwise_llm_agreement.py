import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

# Configuration
SILVER_PATH = "culture_annotation_datasets_silver_annotated_simple.csv"
OUTPUT_DIR = "pairwise_agreement_results"
OUTPUT_LATEX = os.path.join(OUTPUT_DIR, "pairwise_agreement_table.tex")
OUTPUT_SUMMARY = os.path.join(OUTPUT_DIR, "pairwise_agreement_summary.png")


def load_silver_data(path):
    """Load silver dataset and extract rater columns"""
    df = pd.read_csv(path)
    rater_cols = [c for c in df.columns if c.startswith("basicness_score_")]
    return df, rater_cols


def pretty_model_name(name):
    """Convert model names to more readable format"""
    # Replace underscores with spaces and capitalize words
    pretty_name = name.replace("_", " ").title()
    return pretty_name


def calculate_pairwise_metrics(df, rater_cols):
    """Calculate pairwise agreement metrics between all pairs of raters"""
    # Get all possible pairs of raters
    pairs = list(itertools.combinations(rater_cols, 2))

    results = []

    for col1, col2 in pairs:
        # Extract ratings with no missing values
        pair_df = df[[col1, col2]].dropna()

        if len(pair_df) < 5:  # Skip pairs with too few items
            continue

        # Extract model names from column names (assuming format "basicness_score_{model}")
        model1 = col1.replace("basicness_score_", "")
        model2 = col2.replace("basicness_score_", "")

        # Calculate Cohen's Kappa (with linear weights for ordinal data)
        kappa = cohen_kappa_score(pair_df[col1], pair_df[col2], weights="linear")

        # Calculate Spearman correlation
        corr = spearmanr(pair_df[col1], pair_df[col2])[0]

        # Calculate binary agreement (1-2 → 0, 3-4 → 1)
        pair_df_binary = pair_df.copy()
        pair_df_binary[col1] = pair_df_binary[col1].apply(lambda x: 0 if x <= 2 else 1)
        pair_df_binary[col2] = pair_df_binary[col2].apply(lambda x: 0 if x <= 2 else 1)

        # Binary kappa
        binary_kappa = cohen_kappa_score(pair_df_binary[col1], pair_df_binary[col2])

        # Raw agreement percentage
        binary_agreement = np.mean(pair_df_binary[col1] == pair_df_binary[col2])

        # Mean squared error
        mse = np.mean((pair_df[col1] - pair_df[col2]) ** 2)

        results.append(
            {
                "model1": model1,
                "model2": model2,
                "kappa": kappa,
                "correlation": corr,
                "binary_kappa": binary_kappa,
                "binary_agreement": binary_agreement,
                "mse": mse,
                "n_items": len(pair_df),
            }
        )

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
    latex += "\\caption{Pairwise Inter-Model Agreement Metrics}\n"
    latex += "\\label{tab:pairwise_agreement}\n"
    latex += "\\begin{tabular}{lcccc}\n"
    latex += "\\toprule\n"
    latex += "Model Pair & Cohen's $\\kappa$ & Spearman $\\rho$ & Binary Agreement & $n$ \\\\\n"
    latex += "\\midrule\n"

    # Add rows
    for _, row in results_df.iterrows():
        # Escape any special characters in model names (like underscores)
        model1 = row["model1"].replace("_", "\\_")
        model2 = row["model2"].replace("_", "\\_")
        model_pair = f"{model1} -- {model2}"

        # Escape percentages and format data
        binary_agreement = f"{row['binary_agreement']:.1%}".replace("%", "\\%")

        latex += f"{model_pair} & {row['kappa']:.3f} & {row['correlation']:.3f} & {binary_agreement} & {row['n_items']} \\\\\n"

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
    table_only = "\n".join(latex.split("\n")[8:-3])  # Extract just the table part
    return table_only


def create_heatmap(results_df, metric, title, ax=None):
    """Create a heatmap for a specific metric"""
    # Get unique models
    models = sorted(
        list(set(results_df["model1"].unique()) | set(results_df["model2"].unique()))
    )
    n_models = len(models)

    # Create matrix for heatmap
    matrix = np.zeros((n_models, n_models))

    # Fill matrix with metric values
    for _, row in results_df.iterrows():
        i = models.index(row["model1"])
        j = models.index(row["model2"])
        matrix[i, j] = row[metric]
        matrix[j, i] = row[metric]  # Mirror since it's symmetric

    # Set diagonal to 1 (or appropriate value)
    for i in range(n_models):
        if metric in ["kappa", "correlation", "binary_kappa", "binary_agreement"]:
            matrix[i, i] = 1.0
        elif metric == "mse":
            matrix[i, i] = 0.0

    # Use provided axes or create a new figure
    if ax is None:
        plt.figure(figsize=(10, 8))
        ax = plt.gca()

    if metric == "mse":
        # For MSE, lower is better
        cmap = "YlOrRd_r"
        vmin, vmax = 0, max(0.1, matrix.max())
        avg_text = f"Mean MSE: {np.mean(matrix[~np.eye(n_models, dtype=bool)]):.3f}"
    else:
        # For other metrics, higher is better
        cmap = "YlGnBu"
        vmin, vmax = matrix.min() if matrix.min() > 0 else 0, 1

        # Calculate average excluding diagonal
        mask = ~np.eye(n_models, dtype=bool)  # Mask to exclude diagonal
        avg_value = np.mean(matrix[mask])

        if metric == "binary_agreement":
            avg_text = f"Mean: {avg_value:.1%}"
        else:
            avg_text = f"Mean: {avg_value:.3f}"

    # Enhanced title with average value
    full_title = f"{title}\n{avg_text}"

    # Convert model names to more readable format
    pretty_labels = [pretty_model_name(model) for model in models]

    # Create heatmap
    hm = sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f" if metric != "binary_agreement" else ".1%",
        cmap=cmap,
        xticklabels=pretty_labels,
        yticklabels=pretty_labels,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
    )

    # Improve readability of labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Add title with average value
    ax.set_title(full_title)

    # Adjust colorbar (optional)
    cbar = hm.collections[0].colorbar
    if metric == "binary_agreement":
        cbar.set_label("Agreement %")
    elif metric == "mse":
        cbar.set_label("Mean Squared Error")
    else:
        cbar.set_label(title)

    return ax


def generate_readme(results_df, model_stats_df):
    """Generate a README.md file explaining the analysis and files"""
    readme_text = """# Pairwise Inter-Model Agreement Analysis

This directory contains the results of pairwise agreement analysis between different models in the silver dataset.

## Files Overview

- **pairwise_agreement_summary.png**: Combined visualization of all agreement metrics
- **heatmap_kappa.png**: Heatmap of Cohen's Kappa scores between model pairs
- **heatmap_correlation.png**: Heatmap of Spearman correlation coefficients between model pairs
- **heatmap_binary_agreement.png**: Heatmap of binary agreement percentages (after converting 1-2→0, 3-4→1)
- **heatmap_mse.png**: Heatmap of Mean Squared Error between model pairs
- **pairwise_agreement_table.tex**: LaTeX code for a table of pairwise metrics
- **latex_table_standalone.tex**: Complete LaTeX document containing the table (can be compiled independently)
- **summary_statistics.csv**: CSV file with summary statistics across all model pairs
- **full_results.csv**: Complete results for all model pairs with all metrics
- **model_rankings.csv**: Models ranked by their average agreement with other models

## Summary Statistics

"""
    # Add summary statistics
    readme_text += f"- Number of model pairs analyzed: {len(results_df)}\n"
    readme_text += f"- Average Cohen's Kappa: {results_df['kappa'].mean():.3f}\n"
    readme_text += (
        f"- Average Spearman Correlation: {results_df['correlation'].mean():.3f}\n"
    )
    readme_text += (
        f"- Average Agreement: {results_df['binary_agreement'].mean():.1%}\n\n"
    )

    # Add model rankings
    readme_text += "## Model Rankings (by average agreement)\n\n"
    readme_text += "| Model | Avg Kappa | Avg Correlation | Avg Agreement |\n"
    readme_text += "|-------|-----------|-----------------|----------------------|\n"

    for _, row in model_stats_df.iterrows():
        model = pretty_model_name(row["model"])
        readme_text += f"| {model} | {row['avg_kappa']:.3f} | {row['avg_correlation']:.3f} | {row['avg_binary_agreement']:.1%} |\n"

    # Add explanation of metrics
    readme_text += """
## Metrics Explanation

- **Cohen's Kappa**: Measures inter-rater agreement taking into account chance agreement. Values range from -1 to 1, with higher values indicating better agreement. Values above 0.8 are considered very good agreement.

- **Spearman Correlation**: Measures how well the relationship between two variables can be described using a monotonic function. Values range from -1 to 1, with higher values indicating stronger positive correlation.

- **Agreement**: Percentage of items where both models give the same binary classification after converting ratings (1-2→0, 3-4→1).

- **Mean Squared Error (MSE)**: Average of the squared differences between corresponding elements. Lower values indicate better agreement.

## Notes on Interpretation

- The heatmaps display all metrics between each pair of models, with the diagonal set to 1.0 (or 0 for MSE).
- Each metric provides different insights into model agreement. High agreement suggests models are capturing similar patterns.
- Binary agreement often shows higher values than the more stringent Kappa or correlation metrics, as it's less sensitive to exact rating differences.
"""

    # Write to file
    readme_path = os.path.join(OUTPUT_DIR, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme_text)

    return readme_path


def main():
    """Main function to run the pairwise agreement analysis"""
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    print("Loading silver dataset...")
    df, rater_cols = load_silver_data(SILVER_PATH)

    print(f"Calculating pairwise metrics for {len(rater_cols)} models...")
    results_df = calculate_pairwise_metrics(df, rater_cols)

    # Display summary
    print("\nPairwise Agreement Summary:")
    print(f"Number of pairs: {len(results_df)}")
    print(f"Average Cohen's Kappa: {results_df['kappa'].mean():.3f}")
    print(f"Average Spearman Correlation: {results_df['correlation'].mean():.3f}")
    print(f"Average Agreement: {results_df['binary_agreement'].mean():.1%}")

    # Save summary statistics as CSV
    summary_df = pd.DataFrame(
        {
            "metric": [
                "Average Cohen's Kappa",
                "Average Spearman Correlation",
                "Average Agreement",
                "Pairs Analyzed",
            ],
            "value": [
                results_df["kappa"].mean(),
                results_df["correlation"].mean(),
                results_df["binary_agreement"].mean(),
                len(results_df),
            ],
        }
    )
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary_statistics.csv"), index=False)

    # Save full results as CSV for further analysis
    results_df.to_csv(os.path.join(OUTPUT_DIR, "full_results.csv"), index=False)

    # Generate LaTeX table
    latex_table = generate_latex_table(results_df)
    with open(OUTPUT_LATEX, "w") as f:
        f.write(latex_table)
    print(f"\nLaTeX table saved to {OUTPUT_LATEX}")

    # Create summary visualization with all metrics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    create_heatmap(results_df, "kappa", "Cohen's Kappa", axes[0, 0])
    create_heatmap(results_df, "correlation", "Spearman Correlation", axes[0, 1])
    create_heatmap(results_df, "binary_agreement", "Agreement", axes[1, 0])
    create_heatmap(results_df, "mse", "Mean Squared Error", axes[1, 1])

    plt.tight_layout()
    plt.savefig(OUTPUT_SUMMARY, dpi=300)
    print(f"Summary visualizations saved to {OUTPUT_SUMMARY}")

    # Also display individual models' average agreement with others
    model_stats = []
    for model in set(results_df["model1"].unique()) | set(
        results_df["model2"].unique()
    ):
        model_rows = results_df[
            (results_df["model1"] == model) | (results_df["model2"] == model)
        ]
        model_stats.append(
            {
                "model": model,
                "avg_kappa": model_rows["kappa"].mean(),
                "avg_correlation": model_rows["correlation"].mean(),
                "avg_binary_agreement": model_rows["binary_agreement"].mean(),
            }
        )

    model_stats_df = pd.DataFrame(model_stats).sort_values("avg_kappa", ascending=False)
    print("\nModels Ranked by Average Agreement with Others:")
    print(model_stats_df.to_string(index=False, float_format="%.3f"))

    # Save model stats to CSV
    model_stats_df.to_csv(os.path.join(OUTPUT_DIR, "model_rankings.csv"), index=False)

    # Save individual heatmaps in higher resolution
    metrics = {
        "kappa": "Cohen's Kappa",
        "correlation": "Spearman Correlation",
        "binary_agreement": "Agreement (%)",
        "mse": "Mean Squared Error",
    }

    for metric, title in metrics.items():
        fig, ax = plt.subplots(figsize=(10, 8))
        create_heatmap(results_df, metric, title, ax)
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, f"heatmap_{metric}.png")
        plt.savefig(output_path, dpi=300)
        print(f"Saved {metric} heatmap to {output_path}")

    # Generate and save README
    readme_path = generate_readme(results_df, model_stats_df)
    print(f"Generated README file at {readme_path}")

    print(f"\nAll results have been saved to the '{OUTPUT_DIR}' directory")


if __name__ == "__main__":
    main()
