# Pairwise Inter-Model Agreement Analysis

This directory contains the results of pairwise agreement analysis between different models in the silver dataset.

## Files Overview

- **pairwise_agreement_summary.png**: Combined visualization of Cohen's Kappa and MSE
- **heatmap_kappa.png**: Heatmap of Cohen's Kappa scores between model pairs
- **heatmap_mse.png**: Heatmap of Mean Squared Error between model pairs
- **heatmap_corr_vs_agreement.png**: Combined heatmap showing Spearman Correlation (lower triangle) and Binary Agreement (upper triangle)
- **pairwise_agreement_table.tex**: LaTeX code for a table of pairwise metrics
- **latex_table_standalone.tex**: Complete LaTeX document containing the table (can be compiled independently)
- **summary_statistics.csv**: CSV file with summary statistics across all model pairs
- **full_results.csv**: Complete results for all model pairs with all metrics
- **model_rankings.csv**: Models ranked by their average agreement with other models

## Summary Statistics

- Number of model pairs analyzed: 36
- Average Cohen's Kappa: 0.291
- Average Spearman Correlation: 0.589
- Average Agreement: 86.2%

## Model Rankings (by average agreement)

| Model | Avg Kappa | Avg Correlation | Avg Agreement |
|-------|-----------|-----------------|----------------------|
| Gpt 4O | 0.342 | 0.638 | 87.8% |
| Gpt O3 Mini High | 0.340 | 0.594 | 89.4% |
| Claude 3 7 Sonnet | 0.312 | 0.585 | 86.4% |
| Gemini 2 0 Flash | 0.312 | 0.694 | 84.4% |
| Grok 2 | 0.307 | 0.656 | 87.5% |
| Gpt O1 | 0.275 | 0.643 | 80.0% |
| Deepseek R1 | 0.250 | 0.560 | 87.2% |
| Mistral Medium | 0.248 | 0.494 | 85.8% |
| Llama 3 70B | 0.232 | 0.439 | 87.8% |

## Metrics Explanation

- **Cohen's Kappa**: Measures inter-rater agreement taking into account chance agreement. Values range from -1 to 1, with higher values indicating better agreement. Values above 0.8 are considered very good agreement.

- **Spearman Correlation**: Measures how well the relationship between two variables can be described using a monotonic function. Values range from -1 to 1, with higher values indicating stronger positive correlation.

- **Agreement**: Percentage of items where both models give the same binary classification after converting ratings (1-2→0, 3-4→1).

- **Mean Squared Error (MSE)**: Average of the squared differences between corresponding elements. Lower values indicate better agreement.

## Notes on Interpretation

- The heatmaps display all metrics between each pair of models, with the diagonal set to 1.0 (or 0 for MSE).
- Each metric provides different insights into model agreement. High agreement suggests models are capturing similar patterns.
- Binary agreement often shows higher values than the more stringent Kappa or correlation metrics, as it's less sensitive to exact rating differences.
