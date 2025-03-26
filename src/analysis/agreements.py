import krippendorff
import numpy as np
import pandas as pd
import xlsxwriter
import json
from pingouin import intraclass_corr
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import cohen_kappa_score, mean_squared_error

# Configuration
SILVER_PATH = "data/model_annotations/culture_annotation_datasets_silver_annotated.csv"
HUMAN_PATHS = [
    "data/human_annotations/culture_annotation_datasets_annotated_human_it.csv",
    "data/human_annotations/culture_annotation_datasets_annotated_human_nb.csv",
    "data/human_annotations/culture_annotation_datasets_annotated_human_en.csv",
    "data/human_annotations/culture_annotation_datasets_annotated_human_es.csv",
]


def load_data(path, human=False):
    df = pd.read_csv(path)
    if human:
        rater_cols = ["annotator_1", "annotator_2", "annotator_3"]
    else:
        rater_cols = [c for c in df.columns if c.startswith("basicness_score_")]
    return df, rater_cols


def calculate_agreement(ratings):
    """
    Calculate pairwise Cohen's Kappa and Spearman correlation for ordinal ratings.

    Parameters:
    -----------
    ratings : pandas.DataFrame
        DataFrame with columns representing different raters and rows representing items.

    Returns:
    --------
    tuple: (mean_kappa, mean_corr, min_kappa, min_corr, n_comparisons)
        Mean Cohen's Kappa, mean Spearman correlation, minimum kappa, minimum correlation,
        and number of comparisons made.
    """
    kappas, corrs, kendalls = [], [], []
    common_items = []

    for i in range(len(ratings.columns)):
        for j in range(i + 1, len(ratings.columns)):
            # Extract pairs with no missing values
            pair = ratings.iloc[:, [i, j]].dropna()

            common_items.append(len(pair))

            # Calculate weighted kappa (accounts for ordinal nature)
            kappas.append(
                cohen_kappa_score(pair.iloc[:, 0], pair.iloc[:, 1], weights="linear")
            )

            # Calculate Spearman rank correlation
            corrs.append(spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])[0])

            # Add Kendall's Tau-b for an additional rank-based measure
            kendalls.append(kendalltau(pair.iloc[:, 0], pair.iloc[:, 1])[0])

    if not kappas:  # No valid comparisons
        return np.nan, np.nan, np.nan, np.nan, 0

    return (np.mean(kappas), np.mean(corrs), min(kappas), min(corrs), len(kappas))


def calculate_krippendorff(ratings):
    """
    Calculate Krippendorff's alpha for ordinal data.
    Handles missing values internally.

    Parameters:
    -----------
    ratings : pandas.DataFrame
        DataFrame with columns representing different raters and rows representing items.

    Returns:
    --------
    float: Krippendorff's alpha
    """
    # Krippendorff's alpha uses the transpose (raters in rows, items in columns)
    reliability_data = ratings.T.values

    # Using ordinal level of measurement treats ratings as ordered
    return krippendorff.alpha(reliability_data, level_of_measurement="ordinal")


def calculate_icc(ratings):
    """
    Calculate Intraclass Correlation Coefficient (ICC).
    Uses ICC(2) model which treats raters as fixed.

    Parameters:
    -----------
    ratings : pandas.DataFrame
        DataFrame with columns representing different raters and rows representing items.

    Returns:
    --------
    float: ICC(2) value
    """
    # Convert to long format for pingouin
    long_df = ratings.stack().reset_index()
    long_df.columns = ["items", "raters", "scores"]

    try:
        # ICC(2) - model considers raters as fixed
        icc_result = intraclass_corr(
            data=long_df, targets="items", raters="raters", ratings="scores"
        )
        return icc_result.loc[2, "ICC"]  # ICC(2)
    except:
        # In case of failure, revert to a simpler approach
        print("Warning: ICC calculation failed, returning NaN")
        return np.nan


def calculate_mse(ratings):
    """
    Mean squared error of rater pairs.
    For ordinal data, this directly measures the average squared difference in ratings.

    Parameters:
    -----------
    ratings : pandas.DataFrame
        DataFrame with columns representing different raters and rows representing items.

    Returns:
    --------
    float: Mean Squared Error
    """
    mses = []

    for i in range(len(ratings.columns)):
        for j in range(i + 1, len(ratings.columns)):
            pair = ratings.iloc[:, [i, j]].dropna()

            if len(pair) < 5:  # Skip pairs with too few common items
                continue

            mses.append(mean_squared_error(pair.iloc[:, 0], pair.iloc[:, 1]))

    return np.mean(mses) if mses else np.nan


def calculate_gwet_ac2(ratings):
    """
    Calculate Gwet's AC2 agreement coefficient for ordinal data.
    This is a robust alternative to Kappa that handles skewed distributions better.

    Uses quadratic weights for proper handling of ordinal data, where
    disagreements between nearby categories (e.g., 3 vs 4) are less penalized
    than disagreements between distant categories (e.g., 1 vs 4).

    Parameters:
    -----------
    ratings : pandas.DataFrame
        DataFrame with columns representing different raters and rows representing items.

    Returns:
    --------
    float: Gwet's AC2 with quadratic weights
    """
    # Get unique categories and ensure they're sorted
    categories = sorted(np.unique(ratings.values[~np.isnan(ratings.values)]))
    n_categories = len(categories)
    gwets = []

    # Create category mapping for easier indexing
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}

    # Create quadratic weight matrix
    weights = np.zeros((n_categories, n_categories))
    for i in range(n_categories):
        for j in range(n_categories):
            # Quadratic weighting formula: 1 - ((i-j)/(n_categories-1))^2
            weights[i, j] = 1 - ((i - j) / (n_categories - 1)) ** 2

    for i in range(len(ratings.columns)):
        for j in range(i + 1, len(ratings.columns)):
            pair = ratings.iloc[:, [i, j]].dropna()

            if len(pair) < 5:  # Skip pairs with too few common items
                continue

            n_items = len(pair)

            # Calculate observed agreement with weights
            observed_agreement = 0
            for idx, row in pair.iterrows():
                r1, r2 = row.iloc[0], row.iloc[1]
                i_idx, j_idx = cat_to_idx[r1], cat_to_idx[r2]
                observed_agreement += weights[i_idx, j_idx]

            observed_agreement /= n_items

            # Calculate probability of each category
            p_k = np.zeros(n_categories)
            for cat_idx in range(n_categories):
                cat = categories[cat_idx]
                count_i = np.sum(pair.iloc[:, 0] == cat)
                count_j = np.sum(pair.iloc[:, 1] == cat)
                p_k[cat_idx] = (count_i + count_j) / (2 * n_items)

            # Calculate expected agreement by chance
            chance_agreement = 0
            for k in range(n_categories):
                for l in range(n_categories):
                    chance_agreement += weights[k, l] * p_k[k] * p_k[l]

            # Calculate Gwet's AC2
            gwet_ac2 = (observed_agreement - chance_agreement) / (1 - chance_agreement)
            gwets.append(gwet_ac2)

    return np.mean(gwets) if gwets else np.nan


def calculate_binary_agreement(ratings):
    """
    Calculate agreement metrics after converting to binary (1-2 → 0, 3-4 → 1)

    Parameters:
    -----------
    ratings : pandas.DataFrame
        DataFrame with columns representing different raters and rows representing items.

    Returns:
    --------
    tuple: (mean_kappa, agreement_pct)
        Mean Cohen's Kappa and percent agreement after binary conversion
    """
    # Convert to binary: 1-2 → 0, 3-4 → 1
    binary_ratings = ratings.copy()
    for col in binary_ratings.columns:
        binary_ratings[col] = binary_ratings[col].apply(
            lambda x: 0 if x <= 2 else 1 if not pd.isna(x) else np.nan
        )

    kappas = []
    agreements = []

    for i in range(len(binary_ratings.columns)):
        for j in range(i + 1, len(binary_ratings.columns)):
            # Extract pairs with no missing values
            pair = binary_ratings.iloc[:, [i, j]].dropna()

            if len(pair) < 5:  # Skip pairs with too few common items
                continue

            # Calculate kappa
            kappas.append(cohen_kappa_score(pair.iloc[:, 0], pair.iloc[:, 1]))

            # Calculate simple agreement percentage
            agreements.append(np.mean(pair.iloc[:, 0] == pair.iloc[:, 1]))

    if not kappas:  # No valid comparisons
        return np.nan, np.nan

    return np.mean(kappas), np.mean(agreements)


def analyze_dataset(df, rater_cols, dataset_name, is_human=False):
    """
    Analyze inter-rater agreement for a dataset.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing ratings and metadata
    rater_cols : list
        List of column names containing ratings
    dataset_name : str
        Name of the dataset for reporting
    is_human : bool
        Whether this is a human dataset (for reporting purposes)

    Returns:
    --------
    dict: Dictionary containing agreement metrics
    """
    ratings = df[rater_cols]

    # Basic metrics
    mean_kappa, mean_corr, min_kappa, min_corr, n_comparisons = calculate_agreement(
        ratings
    )
    kripp_alpha = calculate_krippendorff(ratings)
    icc = calculate_icc(ratings)
    mse = calculate_mse(ratings)
    gwet = calculate_gwet_ac2(ratings)

    # Binary agreement for all datasets
    binary_kappa, binary_agreement = calculate_binary_agreement(ratings)

    # Language analysis
    lang_results = {}
    for lang in df["relevant_lang"].unique():
        lang_ratings = df[df["relevant_lang"] == lang][rater_cols]
        if lang_ratings.shape[0] < 5:  # Skip languages with too few items
            lang_results[lang] = {
                "kappa": np.nan,
                "correlation": np.nan,
                "n_items": len(lang_ratings),
                "binary_agreement": np.nan,
            }
            continue

        lang_kappa, lang_corr, _, _, _ = calculate_agreement(lang_ratings)

        # Calculate binary agreement for this language
        lang_binary_kappa, lang_binary_agreement = calculate_binary_agreement(
            lang_ratings
        )

        lang_results[lang] = {
            "kappa": lang_kappa,
            "correlation": lang_corr,
            "n_items": len(lang_ratings),
            "binary_agreement": lang_binary_agreement,
        }

    # Top disagreements and agreements
    df_with_stats = df.copy()
    df_with_stats["disagreement"] = ratings.std(axis=1)
    df_with_stats["range"] = ratings.max(axis=1) - ratings.min(axis=1)

    # Add agreement score - inverse of disagreement
    # For display purposes, higher value = higher agreement
    df_with_stats["agreement"] = 1 / (
        df_with_stats["disagreement"] + 0.001
    )  # Add small value to avoid division by zero

    # Get items by complete agreement (all raters gave the same score)
    complete_agreement = df_with_stats[df_with_stats["disagreement"] == 0]
    if len(complete_agreement) > 0:
        top_agreements = complete_agreement.sort_values("representation_lemma").head(
            10
        )[
            ["representation_lemma", "relevant_lang", "disagreement", "range"]
            + rater_cols
        ]
    else:
        # If no complete agreement, get items with lowest disagreement
        top_agreements = df_with_stats.nsmallest(10, "disagreement")[
            ["representation_lemma", "relevant_lang", "disagreement", "range"]
            + rater_cols
        ]

    # Top disagreements
    top_disagreements = df_with_stats.nlargest(10, "disagreement")[
        ["representation_lemma", "relevant_lang", "disagreement", "range"] + rater_cols
    ]

    return {
        "dataset": dataset_name,
        "mean_kappa": mean_kappa,
        "mean_correlation": mean_corr,
        "min_kappa": min_kappa,
        "min_correlation": min_corr,
        "n_comparisons": n_comparisons,
        "krippendorff_alpha": kripp_alpha,
        "icc": icc,
        "mse": mse,
        "gwet_ac2": gwet,
        "binary_kappa": binary_kappa,
        "binary_agreement": binary_agreement,
        "by_language": lang_results,
        "top_disagreements": top_disagreements,
        "top_agreements": top_agreements,
        "complete_agreement_count": len(complete_agreement)
        if "complete_agreement" in locals()
        else 0,
    }


def weighted_vote(ratings, method="mean"):
    """
    Calculate an aggregate rating using various methods.
    Better than simple majority vote for ordinal data.

    Parameters:
    -----------
    ratings : pandas.DataFrame
        DataFrame with columns representing different raters and rows representing items.
    method : str
        Method to use: 'mean', 'median', 'mode', or 'weighted'

    Returns:
    --------
    pandas.Series: Aggregated ratings
    """
    if method == "mean":
        return ratings.mean(axis=1)
    elif method == "median":
        return ratings.median(axis=1)
    elif method == "mode":
        return ratings.mode(axis=1)[0]  # Original majority vote
    elif method == "weighted":
        # Calculate weights based on each rater's agreement with others
        weights = []
        for i in range(ratings.shape[1]):
            # Get column name instead of index
            col = ratings.columns[i]

            # Get rows where this rater has provided a rating
            mask = ~ratings[col].isna()
            if mask.sum() < 5:
                weights.append(0)  # Too few ratings, give zero weight
                continue

            # Get other ratings for these rows
            other_cols = [c for c in ratings.columns if c != col]
            others = ratings.loc[mask, other_cols]
            others_mean = others.mean(axis=1)

            if others_mean.isna().all() or others_mean.std() == 0:
                weights.append(
                    1 / ratings.shape[1]
                )  # Equal weight if no comparison possible
            else:
                # Use a simple alternative approach instead of spearmanr
                try:
                    # Extract values as lists
                    x = ratings.loc[mask, col][~others_mean.isna()].tolist()
                    y = others_mean[~others_mean.isna()].tolist()

                    # Calculate ranks manually
                    x_ranks = pd.Series(x).rank().tolist()
                    y_ranks = pd.Series(y).rank().tolist()

                    # Calculate correlation between ranks (simplified Spearman)
                    n = len(x)
                    if n < 2:
                        corr = 0
                    else:
                        # Use Pearson on the ranks
                        x_mean = sum(x_ranks) / n
                        y_mean = sum(y_ranks) / n

                        numerator = sum(
                            (x_ranks[i] - x_mean) * (y_ranks[i] - y_mean)
                            for i in range(n)
                        )
                        denom_x = sum((x_ranks[i] - x_mean) ** 2 for i in range(n))
                        denom_y = sum((y_ranks[i] - y_mean) ** 2 for i in range(n))

                        if denom_x > 0 and denom_y > 0:
                            corr = numerator / (denom_x * denom_y) ** 0.5
                        else:
                            corr = 0

                    corr = abs(corr)
                except:
                    corr = 0

                weights.append(corr)

        # Normalize weights
        if sum(weights) == 0:
            weights = [1 / len(weights)] * len(weights)
        else:
            weights = [w / sum(weights) for w in weights]

        # Apply weights
        weighted_ratings = pd.DataFrame(0, index=ratings.index, columns=["weighted"])
        for i, col in enumerate(ratings.columns):
            weighted_ratings["weighted"] += ratings[col].fillna(0) * weights[i]

        return weighted_ratings["weighted"]

    else:
        raise ValueError(f"Unknown method: {method}")


def inter_dataset_agreement(silver_df, human_df, method="weighted"):
    """
    Calculate agreement between silver standard and human annotations.

    Parameters:
    -----------
    silver_df : pandas.DataFrame
        DataFrame containing silver standard ratings
    human_df : pandas.DataFrame
        DataFrame containing human ratings
    method : str
        Method to use for aggregating multiple ratings

    Returns:
    --------
    dict: Dictionary containing cross-dataset agreement metrics
    """
    # Merge on ILI to match items across datasets
    merged = pd.merge(silver_df, human_df, on="ili", suffixes=("_silver", "_human"))

    if len(merged) < 5:
        print(f"Warning: Only {len(merged)} common items between datasets")
        return {
            "cohen_kappa": np.nan,
            "spearman_corr": np.nan,
            "kendall_tau": np.nan,
            "percent_agreement": np.nan,
            "mse": np.nan,
            "common_items": len(merged),
        }

    # Get silver and human ratings
    silver_ratings = merged[[c for c in merged.columns if "basicness_score_" in c]]
    human_ratings = merged[["annotator_1", "annotator_2", "annotator_3"]]

    # Aggregate ratings using the specified method
    silver_agg = weighted_vote(silver_ratings, method=method)
    human_agg = weighted_vote(human_ratings, method=method)

    # Calculate agreement metrics
    valid_mask = ~(silver_agg.isna() | human_agg.isna())
    if valid_mask.sum() < 5:
        return {
            "cohen_kappa": np.nan,
            "spearman_corr": np.nan,
            "kendall_tau": np.nan,
            "percent_agreement": np.nan,
            "mse": np.nan,
            "common_items": len(merged),
        }

    silver_valid = silver_agg[valid_mask]
    human_valid = human_agg[valid_mask]

    # Round if needed for kappa calculation
    silver_round = silver_valid.round().astype(int)
    human_round = human_valid.round().astype(int)

    # Calculate metrics
    kappa = cohen_kappa_score(silver_round, human_round, weights="linear")
    spearman = spearmanr(silver_valid, human_valid)[0]
    kendall = kendalltau(silver_valid, human_valid)[0]
    agreement = np.mean(silver_round == human_round)
    mse = mean_squared_error(silver_valid, human_valid)

    return {
        "cohen_kappa": kappa,
        "spearman_corr": spearman,
        "kendall_tau": kendall,
        "percent_agreement": agreement,
        "mse": mse,
        "common_items": len(merged),
    }


def export_to_excel(
    silver_results,
    human_results,
    cross_results,
    output_path="results/agreement_metrics/agreement_results.xlsx",
):
    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    workbook = xlsxwriter.Workbook(output_path)
    ws = workbook.add_worksheet()

    # Formats
    header = workbook.add_format(
        {"bold": True, "bg_color": "#4472C4", "font_color": "white", "border": 1}
    )
    subheader = workbook.add_format({"bold": True, "bg_color": "#8EA9DB", "border": 1})
    cell = workbook.add_format({"border": 1})
    metric_cell = workbook.add_format({"border": 1, "num_format": "0.000"})
    pct_cell = workbook.add_format({"border": 1, "num_format": "0.0%"})

    # Set column widths
    ws.set_column("A:A", 25)
    ws.set_column("B:E", 15)

    # Silver Results
    row = 0
    ws.write(row, 0, "Silver Dataset Results", header)
    row += 1

    metrics = [
        ("Mean Cohen's Kappa", silver_results["mean_kappa"], metric_cell),
        ("Min Cohen's Kappa", silver_results.get("min_kappa", np.nan), metric_cell),
        ("Mean Spearman Correlation", silver_results["mean_correlation"], metric_cell),
        (
            "Min Spearman Correlation",
            silver_results.get("min_correlation", np.nan),
            metric_cell,
        ),
        ("Krippendorff's Alpha", silver_results["krippendorff_alpha"], metric_cell),
        ("ICC(2,k)", silver_results["icc"], metric_cell),
        ("Gwet's AC2", silver_results.get("gwet_ac2", np.nan), metric_cell),
        ("Mean Squared Error", silver_results["mse"], metric_cell),
        (
            "Binary Kappa (1-2→0, 3-4→1)",
            silver_results.get("binary_kappa", np.nan),
            metric_cell,
        ),
        (
            "Binary Agreement %",
            silver_results.get("binary_agreement", np.nan),
            pct_cell,
        ),
        ("Number of Comparisons", silver_results.get("n_comparisons", np.nan), cell),
        (
            "Items with Complete Agreement",
            silver_results.get("complete_agreement_count", np.nan),
            cell,
        ),
    ]

    for metric, value, fmt in metrics:
        ws.write(row, 0, metric, cell)
        ws.write(row, 1, value, fmt)
        row += 1

    # Language Results
    row += 1
    ws.write(row, 0, "Results by Language", subheader)
    row += 1

    ws.write(row, 0, "Language", cell)
    ws.write(row, 1, "Kappa", cell)
    ws.write(row, 2, "Correlation", cell)
    ws.write(row, 3, "Binary Agreement %", cell)
    ws.write(row, 4, "n", cell)
    row += 1

    for lang, metrics in silver_results["by_language"].items():
        ws.write(row, 0, lang, cell)
        ws.write(row, 1, metrics["kappa"], metric_cell)
        ws.write(row, 2, metrics["correlation"], metric_cell)
        if "binary_agreement" in metrics and not pd.isna(metrics["binary_agreement"]):
            ws.write(row, 3, metrics["binary_agreement"], pct_cell)
            ws.write(row, 4, metrics["n_items"], cell)
        else:
            ws.write(row, 3, metrics["n_items"], cell)
        row += 1

    # Add agreement/disagreement items to a separate sheet
    agreement_sheet = workbook.add_worksheet("Agreement Items")

    # Set up header
    agreement_sheet.write(0, 0, "Silver Dataset - Top Agreement Items", header)

    # Write headers for top agreement items
    headers = ["Lemma", "Language", "Disagreement", "Range"] + silver_results[
        "top_agreements"
    ].columns.tolist()[4:]
    for i, h in enumerate(headers):
        agreement_sheet.write(1, i, h, subheader)

    # Write top agreement data
    for i, row_data in enumerate(silver_results["top_agreements"].values):
        for j, val in enumerate(row_data):
            agreement_sheet.write(i + 2, j, val)

    # Add disagreement items
    agreement_sheet.write(15, 0, "Silver Dataset - Top Disagreement Items", header)

    # Write headers for top disagreement items
    for i, h in enumerate(headers):
        agreement_sheet.write(16, i, h, subheader)

    # Write top disagreement data
    for i, row_data in enumerate(silver_results["top_disagreements"].values):
        for j, val in enumerate(row_data):
            agreement_sheet.write(i + 17, j, val)

    # Human Results
    for idx, human in enumerate(human_results):
        row += 2
        ws.write(row, 0, f"{human['dataset']} Results", header)
        row += 1

        for metric, value, fmt in [
            ("Mean Cohen's Kappa", human["mean_kappa"], metric_cell),
            ("Min Cohen's Kappa", human.get("min_kappa", np.nan), metric_cell),
            ("Mean Spearman Correlation", human["mean_correlation"], metric_cell),
            (
                "Min Spearman Correlation",
                human.get("min_correlation", np.nan),
                metric_cell,
            ),
            ("Krippendorff's Alpha", human["krippendorff_alpha"], metric_cell),
            ("ICC(2,k)", human["icc"], metric_cell),
            ("Gwet's AC2", human.get("gwet_ac2", np.nan), metric_cell),
            ("Mean Squared Error", human["mse"], metric_cell),
            (
                "Binary Kappa (1-2→0, 3-4→1)",
                human.get("binary_kappa", np.nan),
                metric_cell,
            ),
            ("Binary Agreement %", human.get("binary_agreement", np.nan), pct_cell),
            ("Number of Comparisons", human.get("n_comparisons", np.nan), cell),
            (
                "Items with Complete Agreement",
                human.get("complete_agreement_count", np.nan),
                cell,
            ),
        ]:
            ws.write(row, 0, metric, cell)
            ws.write(row, 1, value, fmt)
            row += 1

        # Add language results for human datasets
        row += 1
        ws.write(row, 0, "Results by Language", subheader)
        row += 1

        ws.write(row, 0, "Language", cell)
        ws.write(row, 1, "Kappa", cell)
        ws.write(row, 2, "Correlation", cell)
        if "binary_agreement" in next(iter(human["by_language"].values()), {}):
            ws.write(row, 3, "Binary Agreement %", cell)
            ws.write(row, 4, "n", cell)
        else:
            ws.write(row, 3, "n", cell)
        row += 1

        for lang, metrics in human["by_language"].items():
            ws.write(row, 0, lang, cell)
            ws.write(row, 1, metrics["kappa"], metric_cell)
            ws.write(row, 2, metrics["correlation"], metric_cell)
            if "binary_agreement" in metrics:
                ws.write(row, 3, metrics["binary_agreement"], pct_cell)
                ws.write(row, 4, metrics["n_items"], cell)
            else:
                ws.write(row, 3, metrics["n_items"], cell)
            row += 1

        # Add top agreements/disagreements to the agreement sheet
        agreement_offset = 35 + (idx * 35)  # Space between datasets

        agreement_sheet.write(
            agreement_offset, 0, f"{human['dataset']} - Top Agreement Items", header
        )

        # Write headers for top agreement items
        headers = ["Lemma", "Language", "Disagreement", "Range"] + human[
            "top_agreements"
        ].columns.tolist()[4:]
        for i, h in enumerate(headers):
            agreement_sheet.write(agreement_offset + 1, i, h, subheader)

        # Write top agreement data
        for i, row_data in enumerate(human["top_agreements"].values):
            for j, val in enumerate(row_data):
                agreement_sheet.write(agreement_offset + 2 + i, j, val)

        # Add disagreement items
        agreement_sheet.write(
            agreement_offset + 15,
            0,
            f"{human['dataset']} - Top Disagreement Items",
            header,
        )

        # Write headers for top disagreement items
        for i, h in enumerate(headers):
            agreement_sheet.write(agreement_offset + 16, i, h, subheader)

        # Write top disagreement data
        for i, row_data in enumerate(human["top_disagreements"].values):
            for j, val in enumerate(row_data):
                agreement_sheet.write(agreement_offset + 17 + i, j, val)

    # Cross-Dataset Results (Median Method Only)
    row += 2
    ws.write(row, 0, "Cross-Dataset Agreements (Median Method)", header)
    row += 1

    for cross in cross_results:
        ws.write(row, 0, f"Silver vs {cross['dataset']}", subheader)
        row += 1

        metrics = [
            ("Cohen's Kappa", cross["cohen_kappa"], metric_cell),
            ("Spearman Correlation", cross["spearman_corr"], metric_cell),
            ("Kendall's Tau", cross.get("kendall_tau", np.nan), metric_cell),
            ("Percent Agreement", cross["percent_agreement"], pct_cell),
            ("Mean Squared Error", cross.get("mse", np.nan), metric_cell),
            ("Common Items", cross["common_items"], cell),
        ]

        for metric, value, fmt in metrics:
            ws.write(row, 0, metric, cell)
            ws.write(row, 1, value, fmt)
            row += 1

    workbook.close()


def export_to_json(
    silver_results,
    human_results,
    cross_results,
    output_path="results/agreement_metrics/agreement_results.json",
):
    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Create a structured JSON object
    result_obj = {
        "silver_dataset": {
            "name": silver_results["dataset"],
            "metrics": {
                "mean_kappa": silver_results["mean_kappa"],
                "min_kappa": silver_results.get("min_kappa", None),
                "mean_correlation": silver_results["mean_correlation"],
                "min_correlation": silver_results.get("min_correlation", None),
                "krippendorff_alpha": silver_results["krippendorff_alpha"],
                "icc": silver_results["icc"],
                "gwet_ac2": silver_results.get("gwet_ac2", None),
                "mse": silver_results["mse"],
                "binary_kappa": silver_results.get("binary_kappa", None),
                "binary_agreement": silver_results.get("binary_agreement", None),
                "n_comparisons": silver_results.get("n_comparisons", None),
                "complete_agreement_count": silver_results.get(
                    "complete_agreement_count", None
                ),
            },
            "by_language": {},
            "top_agreements": silver_results["top_agreements"].to_dict(
                orient="records"
            ),
            "top_disagreements": silver_results["top_disagreements"].to_dict(
                orient="records"
            ),
        },
        "human_datasets": [],
        "cross_dataset_comparisons": [],
    }

    # Handle silver dataset language breakdown
    for lang, metrics in silver_results["by_language"].items():
        result_obj["silver_dataset"]["by_language"][lang] = {
            "kappa": metrics["kappa"],
            "correlation": metrics["correlation"],
            "n_items": metrics["n_items"],
        }
        if "binary_agreement" in metrics:
            result_obj["silver_dataset"]["by_language"][lang]["binary_agreement"] = (
                metrics["binary_agreement"]
            )

    # Handle human datasets
    for human in human_results:
        human_obj = {
            "name": human["dataset"],
            "metrics": {
                "mean_kappa": human["mean_kappa"],
                "min_kappa": human.get("min_kappa", None),
                "mean_correlation": human["mean_correlation"],
                "min_correlation": human.get("min_correlation", None),
                "krippendorff_alpha": human["krippendorff_alpha"],
                "icc": human["icc"],
                "gwet_ac2": human.get("gwet_ac2", None),
                "mse": human["mse"],
                "binary_kappa": human.get("binary_kappa", None),
                "binary_agreement": human.get("binary_agreement", None),
                "n_comparisons": human.get("n_comparisons", None),
                "complete_agreement_count": human.get("complete_agreement_count", None),
            },
            "by_language": {},
            "top_agreements": human["top_agreements"].to_dict(orient="records"),
            "top_disagreements": human["top_disagreements"].to_dict(orient="records"),
        }

        # Handle language breakdown for each human dataset
        for lang, metrics in human["by_language"].items():
            human_obj["by_language"][lang] = {
                "kappa": metrics["kappa"],
                "correlation": metrics["correlation"],
                "n_items": metrics["n_items"],
            }
            if "binary_agreement" in metrics:
                human_obj["by_language"][lang]["binary_agreement"] = metrics[
                    "binary_agreement"
                ]

        result_obj["human_datasets"].append(human_obj)

    # Handle cross-dataset comparisons
    for cross in cross_results:
        cross_obj = {
            "silver_vs": cross["dataset"],
            "method": cross.get("method", "median"),
            "cohen_kappa": cross["cohen_kappa"],
            "spearman_corr": cross["spearman_corr"],
            "kendall_tau": cross.get("kendall_tau", None),
            "percent_agreement": cross["percent_agreement"],
            "mse": cross.get("mse", None),
            "common_items": cross["common_items"],
        }
        result_obj["cross_dataset_comparisons"].append(cross_obj)

    # Convert numpy values to native Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return convert_numpy(obj.tolist())
        elif obj is np.nan:
            return None
        else:
            return obj

    result_obj = convert_numpy(result_obj)

    # Write to file
    with open(output_path, "w") as f:
        json.dump(result_obj, f, indent=2)

    print(f"JSON results exported to {output_path}")


def main():
    """
    Main function to run the agreement analysis.
    Loads data, analyzes agreement, and exports results.
    """
    print("Loading data...")

    # Analyze silver dataset
    silver_df, silver_raters = load_data(SILVER_PATH)
    silver_results = analyze_dataset(silver_df, silver_raters, "Silver")

    # Analyze human datasets
    human_results = []
    cross_results = []

    for human_path in HUMAN_PATHS:
        human_df, human_raters = load_data(human_path, human=True)
        human_name = f"Human ({human_path.split('_')[-1].split('.')[0].upper()})"
        human_res = analyze_dataset(human_df, human_raters, human_name, is_human=True)
        human_results.append(human_res)

        # Cross-dataset analysis with only median aggregation method
        print(f"\nRunning cross-dataset analysis for {human_name}...")
        method = "median"  # Only use median method
        cross = inter_dataset_agreement(silver_df, human_df, method=method)
        cross["dataset"] = f"{human_name} ({method})"
        cross["method"] = method
        cross_results.append(cross)

    # Print results
    print(f"\n=== {silver_results['dataset']} Dataset Results ===")
    print(f"Mean Cohen's Kappa: {silver_results['mean_kappa']:.3f}")
    print(f"Mean Spearman Correlation: {silver_results['mean_correlation']:.3f}")
    print(f"Krippendorff's Alpha: {silver_results['krippendorff_alpha']:.3f}")
    print(f"ICC(2,k): {silver_results['icc']:.3f}")
    print(f"Gwet's AC2: {silver_results.get('gwet_ac2', np.nan):.3f}")
    print(f"Mean Squared Error: {silver_results['mse']:.3f}")
    print(
        f"Binary Kappa (1-2→0, 3-4→1): {silver_results.get('binary_kappa', np.nan):.3f}"
    )
    print(f"Binary Agreement %: {silver_results.get('binary_agreement', np.nan):.2%}")
    print(
        f"Number of pairwise comparisons: {silver_results.get('n_comparisons', 'N/A')}"
    )
    print(
        f"Items with complete agreement: {silver_results.get('complete_agreement_count', 'N/A')}"
    )

    print("\nBy Language:")
    for lang, metrics in silver_results["by_language"].items():
        output = (
            f"  {lang}: Kappa={metrics['kappa']:.2f}, Corr={metrics['correlation']:.2f}"
        )
        if "binary_agreement" in metrics and not pd.isna(metrics["binary_agreement"]):
            output += f", BinAgree={metrics['binary_agreement']:.2%}"
        output += f" (n={metrics['n_items']})"
        print(output)

    print("\nTop 10 Agreement Items:")
    print(silver_results["top_agreements"].to_string(index=False))

    print("\nTop 10 Disagreements:")
    print(silver_results["top_disagreements"].to_string(index=False))

    for result in human_results:
        print(f"\n=== {result['dataset']} Results ===")
        print(f"Mean Cohen's Kappa: {result['mean_kappa']:.3f}")
        print(f"Mean Spearman Correlation: {result['mean_correlation']:.3f}")
        print(f"Krippendorff's Alpha: {result['krippendorff_alpha']:.3f}")
        print(f"ICC(2,k): {result['icc']:.3f}")
        print(f"Gwet's AC2: {result.get('gwet_ac2', np.nan):.3f}")
        print(f"Mean Squared Error: {result['mse']:.3f}")
        print(f"Binary Kappa (1-2→0, 3-4→1): {result.get('binary_kappa', np.nan):.3f}")
        print(f"Binary Agreement %: {result.get('binary_agreement', np.nan):.2%}")
        print(f"Number of pairwise comparisons: {result.get('n_comparisons', 'N/A')}")
        print(
            f"Items with complete agreement: {result.get('complete_agreement_count', 'N/A')}"
        )

        print("\nBy Language:")
        for lang, metrics in result["by_language"].items():
            output = f"  {lang}: Kappa={metrics['kappa']:.2f}, Corr={metrics['correlation']:.2f}"
            if "binary_agreement" in metrics and not pd.isna(
                metrics["binary_agreement"]
            ):
                output += f", BinAgree={metrics['binary_agreement']:.2%}"
            output += f" (n={metrics['n_items']})"
            print(output)

        print("\nTop 10 Agreement Items:")
        print(result["top_agreements"].to_string(index=False))

        print("\nTop 10 Disagreements:")
        print(result["top_disagreements"].to_string(index=False))

    print("\n=== Cross-Dataset Agreements (Silver vs. Humans, Median Method) ===")
    for cross in cross_results:
        print(f"\nWith {cross['dataset']}:")
        print(f"Cohen's Kappa: {cross['cohen_kappa']:.3f}")
        print(f"Spearman Correlation: {cross['spearman_corr']:.3f}")
        if "kendall_tau" in cross:
            print(f"Kendall's Tau: {cross['kendall_tau']:.3f}")
        print(f"Percent Agreement: {cross['percent_agreement']:.2%}")
        if "mse" in cross:
            print(f"Mean Squared Error: {cross['mse']:.3f}")
        print(f"Common Items Analyzed: {cross['common_items']}")

    # Export to Excel
    export_to_excel(silver_results, human_results, cross_results)
    print("\nResults exported to results/agreement_metrics/agreement_results.xlsx")

    # Export to JSON
    export_to_json(silver_results, human_results, cross_results)
    print("Results exported to results/agreement_metrics/agreement_results.json")


if __name__ == "__main__":
    main()
