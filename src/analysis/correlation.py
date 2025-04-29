"""
Author: Federico Torrielli
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr


def calculate_correlations(filter_by_relevant_lang=True):
    # Set paths
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate up to project root
    project_root = os.path.abspath(os.path.join(script_dir, "../.."))
    data_dir = os.path.join(project_root, "data")

    annotation_dir = os.path.join(data_dir, "human_annotations")
    features_file = os.path.join(data_dir, "concepts/features_per_ILI.csv")

    # Create output directory
    filter_suffix = "filtered" if filter_by_relevant_lang else "unfiltered"
    output_dir = os.path.join(project_root, f"results/correlation/{filter_suffix}")
    os.makedirs(output_dir, exist_ok=True)

    # Languages to process
    languages = ["en", "it", "es", "nb"]

    # Feature columns to correlate with basicness
    feature_cols = [
        "avg_word_length",
        "avg_word_frequency",
        "avg_pronounce_complexity",
        "n_hyponyms",
        "n_synonyms",
        "avg_n_senses",
        "n_syn_senses",
        "word_in_children_res",
        "word_in_second_lang_learn_res",
    ]

    # Read feature data
    features_df = pd.read_csv(features_file)

    # Dictionary to store correlation results
    all_results = {}

    # DataFrame to store all correlation results for CSV export
    pearson_results = pd.DataFrame(index=feature_cols, columns=languages)
    spearman_results = pd.DataFrame(index=feature_cols, columns=languages)
    pearson_p_values = pd.DataFrame(index=feature_cols, columns=languages)
    spearman_p_values = pd.DataFrame(index=feature_cols, columns=languages)

    # Process each language
    for lang in languages:
        print(f"Processing {lang} language...")

        # Read annotation data
        annotation_file = os.path.join(
            annotation_dir, f"culture_annotation_datasets_annotated_human_{lang}.csv"
        )

        try:
            annotation_df = pd.read_csv(annotation_file)
        except Exception as e:
            print(f"Error reading {annotation_file}: {e}")
            continue

        # Filter for entries relevant to this language if requested
        if filter_by_relevant_lang:
            lang_annotations = annotation_df[annotation_df["relevant_lang"] == lang]
            print(f"Filtered to {len(lang_annotations)} entries relevant to {lang}")
        else:
            lang_annotations = annotation_df
            print(f"Using all {len(lang_annotations)} entries for {lang}")

        if lang_annotations.empty:
            print(f"No annotations found for language {lang}")
            continue

        # Calculate mean basicness score across annotators
        lang_annotations["avg_score"] = lang_annotations[
            ["annotator_1", "annotator_2", "annotator_3"]
        ].mean(axis=1)

        # Extract relevant columns for merging
        annotations_subset = lang_annotations[["ili", "avg_score"]].dropna()

        # Filter features for this language
        lang_features = features_df[features_df["Language"] == lang]

        # Merge annotation scores with features
        merged_df = pd.merge(annotations_subset, lang_features, on="ili")

        if merged_df.empty:
            print(f"No matching data found for {lang} after merging")
            continue

        # Dictionary to store results for this language
        lang_results = {"pearson": {}, "spearman": {}}

        # Calculate correlations for each feature
        for feature in feature_cols:
            if feature not in merged_df.columns:
                print(f"Feature {feature} not found in data for {lang}")
                continue

            # Ensure data is numeric and handle missing values
            feature_data = pd.to_numeric(merged_df[feature], errors="coerce")

            # Skip if not enough valid data
            valid_indices = feature_data.notna() & merged_df["avg_score"].notna()
            if valid_indices.sum() < 3:
                print(f"Not enough valid data for {feature} in {lang}")
                continue

            x = feature_data[valid_indices].values
            y = merged_df.loc[valid_indices, "avg_score"].values

            try:
                # Calculate Pearson correlation
                pearson_corr, pearson_p = pearsonr(x, y)
                lang_results["pearson"][feature] = (pearson_corr, pearson_p)

                # Store in dataframes for CSV export
                pearson_results.loc[feature, lang] = pearson_corr
                pearson_p_values.loc[feature, lang] = pearson_p

                # Calculate Spearman correlation (better for ordinal data like basicness scores)
                spearman_corr, spearman_p = spearmanr(x, y)
                lang_results["spearman"][feature] = (spearman_corr, spearman_p)

                # Store in dataframes for CSV export
                spearman_results.loc[feature, lang] = spearman_corr
                spearman_p_values.loc[feature, lang] = spearman_p

            except Exception as e:
                print(f"Error calculating correlation for {lang} - {feature}: {e}")

        all_results[lang] = lang_results

    # Print results
    print("\nCorrelation Results:")
    print("=" * 50)

    for lang in languages:
        if lang not in all_results:
            continue

        print(f"\nCorrelations for {lang.upper()} language:")
        print("-" * 40)

        for feature in feature_cols:
            print(f"\n{feature}:")

            # Pearson correlation
            if (
                "pearson" in all_results[lang]
                and feature in all_results[lang]["pearson"]
            ):
                pearson_corr, pearson_p = all_results[lang]["pearson"][feature]
                print(f"  Pearson: r = {pearson_corr:.3f}, p = {pearson_p:.3f}")
            else:
                print("  Pearson: Could not calculate")

            # Spearman correlation
            if (
                "spearman" in all_results[lang]
                and feature in all_results[lang]["spearman"]
            ):
                spearman_corr, spearman_p = all_results[lang]["spearman"][feature]
                print(f"  Spearman: ρ = {spearman_corr:.3f}, p = {spearman_p:.3f}")
            else:
                print("  Spearman: Could not calculate")

    # Save results to CSV
    pearson_results.to_csv(os.path.join(output_dir, "pearson_correlations.csv"))
    spearman_results.to_csv(os.path.join(output_dir, "spearman_correlations.csv"))
    pearson_p_values.to_csv(os.path.join(output_dir, "pearson_p_values.csv"))
    spearman_p_values.to_csv(os.path.join(output_dir, "spearman_p_values.csv"))

    print(f"\nCSV results saved to {output_dir}")

    # Create visualization of the correlations
    if all_results:
        # --- Load p-values ---
        p_value_file = os.path.join(output_dir, "spearman_p_values.csv")
        try:
            p_values_df = pd.read_csv(p_value_file, index_col=0)
            # Ensure columns match languages (case might differ if manually edited)
            p_values_df.columns = p_values_df.columns.str.lower()
        except FileNotFoundError:
            print(
                f"Warning: P-value file not found at {p_value_file}. Cannot add significance arrows."
            )
            p_values_df = None
        except Exception as e:
            print(
                f"Warning: Error reading p-value file {p_value_file}: {e}. Cannot add significance arrows."
            )
            p_values_df = None

        # Create heatmap of Spearman correlations - adjust figure size for content
        # Dynamically calculate width based on number of features
        width = max(14, len(feature_cols) * 1.2)
        plt.figure(figsize=(width, 8))

        # Use figure's full width by adjusting subplot parameters
        plt.subplots_adjust(right=0.95)  # Extend plot area closer to right edge

        # Prepare data for heatmap and annotations
        heatmap_data = []
        annot_labels = []  # For custom annotations (corr + arrow and p-value)
        langs_with_data = []

        for lang_lower in languages:  # Use lowercase for internal consistency
            lang_upper = lang_lower.upper()
            if (
                lang_lower not in all_results
                or "spearman" not in all_results[lang_lower]
            ):
                continue

            corr_row = []
            label_row = []
            valid_lang_data = False
            for feature in feature_cols:
                if feature in all_results[lang_lower]["spearman"]:
                    corr, p_val = all_results[lang_lower]["spearman"][feature]
                    corr_row.append(corr)

                    # Determine annotation: corr coefficient, potentially with arrow and p-value
                    label = f"{corr:.2f}"  # Default label is just the correlation
                    # Check if p_values_df was loaded successfully and contains the necessary data
                    if (
                        p_values_df is not None
                        and lang_lower in p_values_df.columns
                        and feature in p_values_df.index
                    ):
                        p_val_from_file = p_values_df.loc[feature, lang_lower]
                        # Add arrow and p-value if significant and p-value is valid
                        if pd.notna(p_val_from_file) and p_val_from_file < 0.05:
                            arrow = (
                                "↑" if corr > 0 else "↓" if corr < 0 else ""
                            )  # Added check for corr == 0
                            if (
                                arrow
                            ):  # Only add arrow and p-value if there's a direction
                                label += f" {arrow} p={p_val_from_file:.3f}"

                    label_row.append(label)
                    valid_lang_data = True
                else:
                    corr_row.append(np.nan)
                    label_row.append("")  # Empty label for missing data

            if valid_lang_data:  # Only add row if it contains data
                heatmap_data.append(corr_row)
                annot_labels.append(label_row)
                langs_with_data.append(lang_upper)  # Keep original case for labels

        if not heatmap_data:
            print("No correlation data to visualize")
            return

        # Convert to numpy array
        heatmap_array = np.array(heatmap_data)
        # annot_array = np.array(annot_labels) # No longer needed

        # Create mask for NaN values in correlation data
        mask = np.isnan(heatmap_array)

        # Plot heatmap without default annotations
        ax = sns.heatmap(
            heatmap_array,
            annot=False,  # Turn off default annotations
            # annot=annot_array,  # Use custom annotations
            # fmt="",  # Use pre-formatted strings in annot_array
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            center=0,
            xticklabels=[f.replace("_", " ").title() for f in feature_cols],
            yticklabels=langs_with_data,
            mask=mask,
            # annot_kws={"size": 8},  # Adjust font size if needed
            linewidths=0.5,  # Add lines between cells for clarity
            linecolor="lightgray",
        )

        # --- Manual Annotations --- >
        for i, lang_lower in enumerate(
            [l.lower() for l in langs_with_data]
        ):  # Iterate rows (langs)
            for j, feature in enumerate(feature_cols):  # Iterate columns (features)
                if mask[i, j]:  # Skip masked (NaN) cells
                    continue

                corr = heatmap_array[i, j]
                text_color = (
                    "white" if abs(corr) > 0.6 else "black"
                )  # Contrast for text

                # Default text is just the correlation coefficient
                base_text = f"{corr:.2f}"
                pill_text = ""
                pill_color = None

                # Check significance if p-values are available
                if (
                    p_values_df is not None
                    and lang_lower in p_values_df.columns
                    and feature in p_values_df.index
                ):
                    p_val_from_file = p_values_df.loc[feature, lang_lower]
                    if pd.notna(p_val_from_file):
                        # Determine significance level and corresponding colors/text
                        if p_val_from_file < 0.001:
                            sig_text = "p<0.001"
                            pos_color = "darkgreen"
                            neg_color = "darkred"
                        elif p_val_from_file < 0.01:
                            sig_text = "p<0.01"
                            pos_color = "forestgreen"
                            neg_color = "firebrick"
                        elif p_val_from_file < 0.05:
                            sig_text = "p<0.05"
                            pos_color = "mediumseagreen"
                            neg_color = "indianred"
                        else:
                            sig_text = ""
                            pos_color = neg_color = None

                        if sig_text:  # Only create pill if significant
                            if corr > 0:
                                arrow = "↑"
                                pill_color = pos_color
                            elif corr < 0:
                                arrow = "↓"
                                pill_color = neg_color
                            else:
                                arrow = ""
                                pill_color = None

                            if arrow and pill_color:
                                pill_text = f"{arrow} {sig_text}"
                                base_text += " "  # Add space before pill

                # --- Draw the text elements --- >
                # Place base text slightly above center
                ax.text(
                    j + 0.5,  # Center horizontally
                    i + 0.65,  # Position slightly above vertical center
                    base_text,
                    ha="center",  # Center align horizontally
                    va="center",  # Vertically align relative to its position
                    color=text_color,
                    fontsize=10,
                )

                # If there's a pill, draw it below the base text
                if pill_text:
                    ax.text(
                        j + 0.5,  # Center horizontally
                        i + 0.35,  # Position slightly below vertical center
                        pill_text,
                        ha="center",  # Center align horizontally
                        va="center",  # Vertically align relative to its position
                        color="white",
                        fontsize=8,
                        bbox=dict(
                            boxstyle="round,pad=0.2",
                            facecolor=pill_color,
                            edgecolor="none",
                        ),
                    )
                # --- End Manual Annotations ---

        plt.title(
            "Spearman Correlation between Features and Basicness Scores",
            fontsize=14,
        )
        plt.tight_layout()

        # Save figure
        output_path = os.path.join(output_dir, "feature_basicness_correlations.png")
        plt.savefig(output_path)
        plt.close()

        print(f"\nHeatmap saved to {output_path}")
    else:
        print("\nNo results to visualize")


def main():
    # Run both filtered and unfiltered versions
    print("\n=== RUNNING FILTERED ANALYSIS (only relevant_lang entries) ===\n")
    calculate_correlations(filter_by_relevant_lang=True)

    print("\n=== RUNNING UNFILTERED ANALYSIS (all entries) ===\n")
    calculate_correlations(filter_by_relevant_lang=False)


if __name__ == "__main__":
    main()
