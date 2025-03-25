# Pairwise Inter-Language Agreement Analysis

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

- Number of language pairs analyzed: 6
- Average Cohen's Kappa: 0.294
- Average Spearman Correlation: 0.372
- Average Agreement: 94.8%

## Language Pairs Ranked by Agreement (Kappa)

| Language Pair | Cohen's Kappa | Spearman Correlation | Agreement |
|---------------|--------------|----------------------|------------------|
| Italian - English | 0.626 | 0.582 | 93.2% |
| Italian - Norwegian | 0.495 | 0.704 | 90.7% |
| Norwegian - English | 0.375 | 0.324 | 96.1% |
| English - Spanish | 0.138 | 0.481 | 98.9% |
| Italian - Spanish | 0.093 | 0.190 | 93.2% |
| Norwegian - Spanish | 0.039 | -0.048 | 96.4% |

## Metrics Explanation

- **Cohen's Kappa**: Measures inter-rater agreement taking into account chance agreement. Values range from -1 to 1, with higher values indicating better agreement. Values above 0.8 are considered very good agreement.

- **Spearman Correlation**: Measures how well the relationship between two variables can be described using a monotonic function. Values range from -1 to 1, with higher values indicating stronger positive correlation.

- **Agreement**: Percentage of items where both languages give the same binary classification after converting ratings (1-2→0, 3-4→1).

- **Mean Squared Error (MSE)**: Average of the squared differences between corresponding elements. Lower values indicate better agreement.

## Notes on Interpretation

- High agreement between languages suggests consistent conceptual understanding across cultures.
- Binary agreement often shows higher values than the more stringent Kappa or correlation metrics, as it's less sensitive to exact rating differences.
- Differences in agreement may reflect cultural differences in basic-level categorization.
