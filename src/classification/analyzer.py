"""
Author: Gianluca Barmina
"""

import os
from typing import Dict, List

import numpy as np
import pandas as pd  # type: ignore
import ujson as json  # type: ignore
from utils import get_path_from_root


class OMWBasicnessAnalyzer:
    def __init__(
        self,
        input_df: pd.DataFrame,
        basicness_weights: dict | None = None,
        opt_set_path: str | None = None,
        language_specific_weights: Dict[str, dict] | None = None,
    ):
        self.__input_df = input_df.copy()
        self.__result_df = input_df.copy()
        self.__result_df_ranked = None
        self.__language_specific_weights = language_specific_weights

        # Load cultural ilis
        # TODO add check on file existence
        with open(
            get_path_from_root("resources/culture/culture_dict_lang_ilis.json"), "r"
        ) as f:
            self.__lang_ilis_dict = json.load(f)

        self.__opt_set_df: pd.DataFrame | None = pd.DataFrame()

        if opt_set_path is not None:
            self.__opt_set_df = pd.read_excel(opt_set_path)
            # Shuffle and split train and test
            shuffled = self.__opt_set_df.sample(frac=1, random_state=42).reset_index(
                drop=True
            )
            train_size = int(0.8 * len(shuffled))
            self.__opt_set_train_df = shuffled[:train_size]
            self.__opt_set_test_df = shuffled[train_size:]
        else:
            self.__opt_set_df = None

        if basicness_weights is None:
            self.__basicness_weights = {
                "word_frequency_weight": 3,
                "word_length_weight": 1,
                "pronounce_complexity_weight": 1,
                "n_hyponyms_weight": 0.1,
                "n_synonyms_weight": 0.1,
                "n_senses_weight": 0.1,
                "word_in_children_res_weight": 0.1,
                "word_in_second_lang_learn_res_weight": 0.1,
                "n_syn_senses_weight": 0.1,
            }
        else:
            self.__basicness_weights = basicness_weights

    def set_weights(self, basicness_weights: dict):
        self.__basicness_weights = basicness_weights

    def get_weights(self) -> dict:
        """
        Get the current basicness weights.

        :return: Dictionary of basicness weights
        """
        return self.__basicness_weights

    def set_language_specific_weights(self, language_specific_weights: Dict[str, dict]):
        """
        Set language-specific weights for basicness calculation.

        :param language_specific_weights: Dictionary mapping language codes to weight dictionaries
        """
        self.__language_specific_weights = language_specific_weights

    def get_language_specific_weights(self) -> Dict[str, dict] | None:
        """
        Get the current language-specific weights.

        :return: Dictionary mapping language codes to weight dictionaries
        """
        return self.__language_specific_weights

    @staticmethod
    def load_correlation_weights(correlation_csv_path: str) -> Dict[str, dict]:
        """
        Load correlation weights from a CSV file.

        :param correlation_csv_path: Path to the CSV file with correlation values
        :return: Dictionary mapping language codes to weight dictionaries
        """
        corr_df = pd.read_csv(correlation_csv_path, index_col=0)
        language_weights = {}

        for lang in corr_df.columns:
            weights = {
                "word_frequency_weight": corr_df.loc["avg_word_frequency", lang],
                "word_length_weight": corr_df.loc["avg_word_length", lang],
                "pronounce_complexity_weight": corr_df.loc[
                    "avg_pronounce_complexity", lang
                ],
                "n_hyponyms_weight": corr_df.loc["n_hyponyms", lang],
                "n_synonyms_weight": corr_df.loc["n_synonyms", lang],
                "n_senses_weight": corr_df.loc["avg_n_senses", lang],
                "word_in_children_res_weight": corr_df.loc[
                    "word_in_children_res", lang
                ],
                "word_in_second_lang_learn_res_weight": corr_df.loc[
                    "word_in_second_lang_learn_res", lang
                ],
                "n_syn_senses_weight": corr_df.loc["n_syn_senses", lang],
            }
            language_weights[lang] = weights

        return language_weights

    def __calculate_basicness_score(self, row: pd.DataFrame) -> float:
        # Get language-specific weights if available, otherwise use default weights
        # TODO check: it doesn't make sense to use this formula with language-specific weights
        #   since the weights of different feature have different ranges
        #   but language-specific weights have all the same range
        # if (
        #     self.__language_specific_weights
        #     and row["Language"] in self.__language_specific_weights
        # ):
        #     weights = self.__language_specific_weights[row["Language"]]
        # else:
        #     weights = self.__basicness_weights

        weights = self.__basicness_weights

        # Between 0 and 3
        word_frequency_weight = weights["word_frequency_weight"]
        word_length_weight = weights["word_length_weight"]
        pronounce_complexity_weight = weights["pronounce_complexity_weight"]

        # Between 0 and 1
        n_hyponyms_weight = weights["n_hyponyms_weight"]
        n_synonyms_weight = weights["n_synonyms_weight"]
        n_senses_weight = weights["n_senses_weight"]
        word_in_children_res_weight = weights["word_in_children_res_weight"]
        word_in_second_lang_learn_res_weight = weights[
            "word_in_second_lang_learn_res_weight"
        ]
        n_syn_senses_weight = weights["n_syn_senses_weight"]

        normalized_frequency = row["normalized_frequency"]
        normalized_length = row["normalized_length"]
        normalized_pronounce_complexity = row["normalized_pronounce_complexity"]
        normalized_n_hyponyms = row["normalized_n_hyponyms"]
        normalized_n_synonyms = row["normalized_n_synonyms"]
        normalized_n_senses = row["normalized_n_senses"]
        word_in_children_res = row["word_in_children_res"]
        word_in_second_lang_learn_res = row["word_in_second_lang_learn_res"]
        normalized_n_syn_senses = row["normalized_n_syn_senses"]

        # Handle negative correlations by inverting the normalized values
        # TODO check: since language-specific weights should not be used with this formula the following is not needed
        # length_factor = (
        #     (1 - normalized_length)
        #     if weights["word_length_weight"] < 0
        #     else normalized_length
        # )
        # pronounce_complexity_factor = (
        #     (1 - normalized_pronounce_complexity)
        #     if weights["pronounce_complexity_weight"] < 0
        #     else normalized_pronounce_complexity
        # )
        # n_synonyms_factor = (
        #     (1 - normalized_n_synonyms)
        #     if weights["n_synonyms_weight"] < 0
        #     else normalized_n_synonyms
        # )
        # n_senses_factor = (
        #     (1 - normalized_n_senses)
        #     if weights["n_senses_weight"] < 0
        #     else normalized_n_senses
        # )
        # n_syn_senses_factor = (
        #     (1 - n_syn_senses) if weights["n_syn_senses_weight"] < 0 else n_syn_senses
        # )
        # word_in_children_res_factor = (
        #     (1 - word_in_children_res)
        #     if weights["word_in_children_res_weight"] < 0
        #     else word_in_children_res
        # )
        # word_in_second_lang_learn_res_factor = (
        #     (1 - word_in_second_lang_learn_res)
        #     if weights["word_in_second_lang_learn_res_weight"] < 0
        #     else word_in_second_lang_learn_res
        # )

        dependent_terms = (
            np.pow(normalized_frequency, abs(word_frequency_weight))
            * np.pow(
                normalized_length, abs(word_length_weight)
            )  # changed from factor to normalized_length
            * np.pow(
                normalized_pronounce_complexity, abs(pronounce_complexity_weight)
            )  # changed from factor to normalized_pronounce_complexity
        )

        # Also here changed from factors to result data directly, restoring original formula
        independent_terms = (
            abs(n_senses_weight) * normalized_n_senses
            + abs(n_synonyms_weight) * normalized_n_synonyms
            + abs(n_hyponyms_weight) * normalized_n_hyponyms
            + abs(word_in_children_res_weight) * word_in_children_res
            + abs(word_in_second_lang_learn_res_weight) * word_in_second_lang_learn_res
            + abs(n_syn_senses_weight) * normalized_n_syn_senses
        )

        basicness_score = dependent_terms + independent_terms

        return basicness_score

    def __calculate_basicness_score_experimental(self, row: pd.Series) -> float:
        """
        Calculates a basicness score using a weighted linear combination
        of normalized features, followed by a sigmoid transformation.
        This method is designed to be mathematically sound for signed weights
        (e.g., Spearman correlations).

        Args:
            row: A pandas Series containing the normalized feature values for a word/language.
                 Requires columns like 'normalized_frequency', 'normalized_length', etc.,
                 and assumes 'n_syn_senses' is also normalized [0, 1].

        Returns:
            The calculated basicness score, scaled between 0 and 1.
        """

        if (
            self.__language_specific_weights
            and row["Language"] in self.__language_specific_weights
        ):
            weights = self.__language_specific_weights[row["Language"]]
        else:
            weights = self.__basicness_weights

        # --- 1. Feature Extraction (Ensure all are scaled 0-1) ---
        # Note: We directly use the normalized features. No (1-feature) inversion needed here.
        # The sign of the weight handles the direction of influence.
        norm_freq = row["normalized_frequency"]
        norm_len = row["normalized_length"]
        norm_pron_comp = row["normalized_pronounce_complexity"]
        norm_hypo = row["normalized_n_hyponyms"]
        norm_syn = row["normalized_n_synonyms"]
        norm_senses = row["normalized_n_senses"]
        in_child = row["word_in_children_res"]  # Already 0/1
        in_learn = row["word_in_second_lang_learn_res"]  # Already 0/1

        # CRITICAL ASSUMPTION: 'n_syn_senses' must be pre-normalized to [0, 1]
        # If it's a raw count, it needs scaling before this step.
        norm_syn_senses = row[
            "normalized_n_syn_senses"
        ]  # TODO check: now should be normalized, need to recalculate correlations for this feature
        # if not (0 <= norm_syn_senses <= 1):
        #     # Add a warning or raise an error if the assumption is violated
        #     # For now, let's clip it as a safeguard, but pre-normalization is better.
        #     # print(f"Warning: 'n_syn_senses' ({norm_syn_senses}) not in [0,1]. Clipping.")
        #     norm_syn_senses = np.clip(norm_syn_senses, 0, 1)

        # --- 2. Weighted Linear Combination (Raw Score) ---
        raw_score = (
            weights["word_frequency_weight"] * norm_freq
            + weights["word_length_weight"] * norm_len  # Weight sign determines effect
            + weights["pronounce_complexity_weight"]
            * norm_pron_comp  # Weight sign determines effect
            + weights["n_hyponyms_weight"] * norm_hypo
            + weights["n_synonyms_weight"] * norm_syn
            + weights["n_senses_weight"] * norm_senses
            + weights["word_in_children_res_weight"] * in_child
            + weights["word_in_second_lang_learn_res_weight"] * in_learn
            + weights["n_syn_senses_weight"] * norm_syn_senses
        )

        # --- 3. Output Transformation (Sigmoid) ---
        # Maps the unbounded raw_score to the range (0, 1)
        final_score = 1 / (1 + np.exp(-raw_score))

        return final_score

    def analyze_lang_syn_group(
        self, word: str | None = None, thresholds: List[float] | None = None
    ) -> pd.DataFrame:
        df = self.__input_df.copy()

        # Add word_count column
        df["word_count"] = df.groupby(["Language", "Synset"])["word_length"].transform(
            "count"
        )

        avoid_zero_division = 1e-12

        n_hyponyms_denom = avoid_zero_division
        n_synonyms_denom = avoid_zero_division
        norm_length_denom = avoid_zero_division
        norm_freq_denom = avoid_zero_division
        norm_basicness_denom = avoid_zero_division
        n_senses_denom = avoid_zero_division
        n_syn_senses_denom = avoid_zero_division

        # Step 1: Plain metrics
        plain_metrics = (
            df.groupby(["Language", "ili"])
            .agg(
                avg_word_length=("word_length", "mean"),
                avg_word_frequency=("word_frequency", "mean"),
                avg_pronounce_complexity=("pronounce_complexity", "mean"),
                n_hyponyms=("n_hyponyms", "max"),
                n_synonyms=("n_synonyms", "max"),
                avg_n_senses=("n_senses", "mean"),
                word_in_children_res=("word_in_children_res", "max"),
                word_in_second_lang_learn_res=("word_in_second_lang_learn_res", "max"),
                n_syn_senses=("n_syn_senses", "max"),
            )
            .reset_index()
        )

        # Step 2: Calculate min-max metrics by language instead of by ILI
        min_max_metrics_by_lang = (
            plain_metrics.groupby("Language")
            .agg(
                min_word_length=("avg_word_length", "min"),
                max_word_length=("avg_word_length", "max"),
                min_word_frequency=("avg_word_frequency", "min"),
                max_word_frequency=("avg_word_frequency", "max"),
                min_pronounce_complexity=("avg_pronounce_complexity", "min"),
                max_pronounce_complexity=("avg_pronounce_complexity", "max"),
                min_n_hyponyms=("n_hyponyms", "min"),
                max_n_hyponyms=("n_hyponyms", "max"),
                min_n_synonyms=("n_synonyms", "min"),
                max_n_synonyms=("n_synonyms", "max"),
                min_n_senses=("avg_n_senses", "min"),
                max_n_senses=("avg_n_senses", "max"),
                # Not present before
                min_n_syn_senses=("n_syn_senses", "min"),
                max_n_syn_senses=("n_syn_senses", "max"),
            )
            .reset_index()
        )

        # Step 3: Normalize metrics within each language
        normalized = pd.merge(plain_metrics, min_max_metrics_by_lang, on="Language")
        normalized["normalized_length"] = (
            normalized["avg_word_length"] - normalized["min_word_length"]
        ) / (
            normalized["max_word_length"]
            - normalized["min_word_length"]
            + norm_length_denom
        )

        normalized["normalized_frequency"] = (
            normalized["avg_word_frequency"] - normalized["min_word_frequency"]
        ) / (
            normalized["max_word_frequency"]
            - normalized["min_word_frequency"]
            + norm_freq_denom
        )

        normalized["normalized_pronounce_complexity"] = normalized[
            "avg_pronounce_complexity"
        ]

        normalized["normalized_n_hyponyms"] = (
            normalized["n_hyponyms"] - normalized["min_n_hyponyms"]
        ) / (
            normalized["max_n_hyponyms"]
            - normalized["min_n_hyponyms"]
            + n_hyponyms_denom
        )

        normalized["normalized_n_synonyms"] = (
            normalized["n_synonyms"] - normalized["min_n_synonyms"]
        ) / (
            normalized["max_n_synonyms"]
            - normalized["min_n_synonyms"]
            + n_synonyms_denom
        )

        normalized["normalized_n_senses"] = (
            normalized["avg_n_senses"] - normalized["min_n_senses"]
        ) / (normalized["max_n_senses"] - normalized["min_n_senses"] + n_senses_denom)

        # Not present before
        normalized["normalized_n_syn_senses"] = (
            normalized["n_syn_senses"] - normalized["min_n_syn_senses"]
        ) / (
            normalized["max_n_syn_senses"]
            - normalized["min_n_syn_senses"]
            + n_syn_senses_denom
        )

        # Step 4: Combined metrics
        # Calculate original score (if needed for comparison or if it's still the primary target)
        normalized["combined_metric_normalized"] = normalized.apply(
            self.__calculate_basicness_score, axis=1
        )
        # Calculate the NEW experimental score
        normalized["basicness_score_experimental"] = normalized.apply(
            self.__calculate_basicness_score_experimental, axis=1
        )

        # Step 5: Min-max of combined metrics
        # min_max_values = (
        #     normalized.groupby("Language") # TODO check: here changed groupby from ili to language
        #     .agg(
        #         min_combined_normalized=("combined_metric_normalized", "min"),
        #         max_combined_normalized=("combined_metric_normalized", "max"),
        #     )
        #     .reset_index()
        # )

        # Calculate min and max values of metric for each language
        min_combined_normalized = normalized.groupby("Language")[
            "combined_metric_normalized"
        ].transform("min")
        max_combined_normalized = normalized.groupby("Language")[
            "combined_metric_normalized"
        ].transform("max")

        # Step 6: Lemmas by language
        # Filter the DataFrame for each language and join the lemmas for each group
        en_lemmas = (
            df[df["Language"] == "en"]
            .groupby("ili")["Lemma"]
            .apply(", ".join)
            .reset_index(name="en_lemmas")
        )
        it_lemmas = (
            df[df["Language"] == "it"]
            .groupby("ili")["Lemma"]
            .apply(", ".join)
            .reset_index(name="it_lemmas")
        )
        nb_lemmas = (
            df[df["Language"] == "nb"]
            .groupby("ili")["Lemma"]
            .apply(", ".join)
            .reset_index(name="nb_lemmas")
        )
        es_lemmas = (
            df[df["Language"] == "es"]
            .groupby("ili")["Lemma"]
            .apply(", ".join)
            .reset_index(name="es_lemmas")
        )

        # Merge the dataframes on 'ili' to get all language lemmas in a single dataframe
        lemmas_by_language = (
            en_lemmas.merge(it_lemmas, on="ili", how="left")
            .merge(nb_lemmas, on="ili", how="left")
            .merge(es_lemmas, on="ili", how="left")
        )

        lemmas_by_language = lemmas_by_language.reset_index(drop=True)

        # Step 7: Select an english example lemma and gloss for each ILI
        example_en_lemma = (
            df[df["Language"] == "en"]
            .groupby("ili")["Lemma"]
            .apply(lambda x: x[x != "None"].head(1))
            .reset_index(name="example_en_lemma")
        )
        example_en_gloss = (
            df[df["Language"] == "en"]
            .groupby("ili")["Gloss"]
            .apply(lambda x: x[x != "None"].head(1))
            .reset_index(name="example_en_gloss")
        )

        # Step 8: Merge sub-dataframes to get the final dataframe
        # result = pd.merge(normalized, min_max_values, on="ili") # not needed anymore because we add min and max values to "normalize" below
        # Add mix and max values (by language) for the metric as new columns
        normalized["min_combined_normalized"] = min_combined_normalized
        normalized["max_combined_normalized"] = max_combined_normalized
        result = pd.merge(normalized, lemmas_by_language, on="ili")
        result = pd.merge(result, example_en_lemma, on="ili")
        result = pd.merge(result, example_en_gloss, on="ili")

        # Step 9: Calculate final scores
        # Original basicness score (derived from the potentially flawed __calculate_basicness_score)
        result["basicness_score"] = (
            result["combined_metric_normalized"] - result["min_combined_normalized"]
        ) / (
            result["max_combined_normalized"]
            - result["min_combined_normalized"]
            + norm_basicness_denom
        )  # Uses norm_basicness_denom

        # The experimental score is already scaled 0-1 by sigmoid, no further ILI-group scaling needed here.
        # We keep the direct output of __calculate_basicness_score_experimental
        result["basicness_score_experimental"] = result[
            "basicness_score_experimental"
        ]  # Already 0-1

        # Final selection of columns and sorting
        final_result = result[
            [
                "Language",
                "basicness_score",
                "basicness_score_experimental",
                "word_in_children_res",
                "word_in_second_lang_learn_res",
                "en_lemmas",
                "n_hyponyms",
                "n_synonyms",
                "n_syn_senses",
                "avg_word_length",
                "avg_pronounce_complexity",
                "avg_word_frequency",
                "avg_n_senses",
                "normalized_length",
                "normalized_pronounce_complexity",
                "normalized_n_senses",
                "normalized_frequency",
                "normalized_n_hyponyms",
                "normalized_n_synonyms",
                "normalized_n_syn_senses",
                "example_en_lemma",
                "example_en_gloss",
                "ili",
            ]
        ]

        final_result = final_result.sort_values(
            by=["ili", "basicness_score_experimental"], ascending=[True, False]
        )

        self.__result_df = final_result

        # Add basicness rank
        ranked_df = self.add_basicness_rank(
            score_column="basicness_score_experimental", thresholds=thresholds
        )

        self.__result_df = ranked_df

        # Save the dataframe to CSV
        if word is not None:
            save_path = get_path_from_root(f"analysis/{word}_lang_syn.csv")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            ranked_df.to_csv(save_path, index=False)

        return ranked_df

    def evaluate_basicness_score(self) -> dict:
        """
        Evaluate basicness score recall. For each tuple ili, L(language), basicness_score
        if ili is in the list of relevant ilis for L checks that it has a sufficiently high basicness_score.
        :param input_df: DataFrame with columns 'Language', 'ili', 'basicness_score'
        :return: The accuracy of the basicness score on the input data
        """
        basicness_relevance_threshold = 0.65
        basicness_irrelevance_threshold = 0.5
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0

        # Check for each row if the ili is in the relevant ilis for the language
        for index, row in self.__result_df.iterrows():
            ili = row["ili"]
            lang = row["Language"]
            basicness_score = row["basicness_score"]
            if ili in self.__lang_ilis_dict[lang]:
                if basicness_score >= basicness_relevance_threshold:
                    true_positives += 1
                else:
                    false_negatives += 1
            else:
                if basicness_score < basicness_irrelevance_threshold:
                    true_negatives += 1
                elif basicness_score >= basicness_relevance_threshold:
                    false_positives += 1

        # Calculate evaluation metrics
        accuracy = (true_positives + true_negatives) / (
            true_positives + true_negatives + false_positives + false_negatives
        )
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1 = 2 * (precision * recall) / (precision + recall)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def evaluate_basicness_score_max(self) -> int:
        """
        Evaluate basicness score recall. For each tuple ili, L(language), basicness_score
        if ili is in the list of relevant ilis for L checks that it has a sufficiently high basicness_score.
        :param input_df: DataFrame with columns 'Language', 'ili', 'basicness_score'
        :return: The accuracy of the basicness score on the input data
        """
        basicness_relevance_threshold = 0.65
        true_positive = 0
        n_present_ilis = 0

        # Check for each row if the ili is in the relevant ilis for the language
        for index, row in self.__result_df.iterrows():
            ili = row["ili"]
            lang = row["Language"]
            basicness_score = row["basicness_score"]
            if ili in self.__lang_ilis_dict[lang]:
                n_present_ilis += 1
                if basicness_score >= basicness_relevance_threshold:
                    true_positive += 1

        return true_positive // n_present_ilis

    def evaluate_basicness_score_mse(self, use_test_set: bool = False) -> float:
        if self.__opt_set_df is None:
            raise ValueError(
                "Ordinal regression loss requires an optimization dataset."
            )

        function_outputs = []
        human_outputs = []

        if use_test_set:
            function_result_df = self.__opt_set_test_df
        else:
            function_result_df = self.__opt_set_train_df

        for index, row in self.__result_df.iterrows():
            ili = row["ili"]
            lang = row["Language"]
            basicness_score = row["basicness_score"]
            for i, row_annotation in function_result_df.iterrows():
                if row_annotation["ili"] == ili and row_annotation["Language"] == lang:
                    human_outputs.append(row_annotation["basicness_score"])
                    function_outputs.append(basicness_score)

        human_outputs_transformed = (np.array(human_outputs) - 1) / 3.0

        mse = np.mean(
            (np.array(function_outputs) - np.array(human_outputs_transformed)) ** 2
        )

        return mse

    def evaluate_basicness_score_binary(self, threshold: float) -> dict:
        if self.__opt_set_df is None:
            raise ValueError("Ranking evaluation requires an optimization dataset.")

        n_total = 0
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        for index, row in self.__result_df.iterrows():
            ili = row["ili"]
            lang = row["Language"]
            score = row["basicness_score"]
            if score >= threshold:
                predicted_binary_score = 1
            else:
                predicted_binary_score = 0
            for i, row_annotation in self.__opt_set_df.iterrows():
                if row_annotation["ili"] == ili and row_annotation["Language"] == lang:
                    n_total += 1
                    actual_binary_score = row_annotation["basicness_score"]
                    if (
                        actual_binary_score == predicted_binary_score
                        and predicted_binary_score == 1
                    ):
                        true_positives += 1
                    elif (
                        actual_binary_score == predicted_binary_score
                        and predicted_binary_score == 0
                    ):
                        true_negatives += 1
                    elif (
                        actual_binary_score != predicted_binary_score
                        and predicted_binary_score == 1
                    ):
                        false_positives += 1
                    elif (
                        actual_binary_score != predicted_binary_score
                        and predicted_binary_score == 0
                    ):
                        false_negatives += 1

        accuracy = (true_positives + true_negatives) / n_total
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1 = 2 * (precision * recall) / (precision + recall)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def evaluate_accordance(
        self,
        opt_split: str = "full",
        allow_margin_error: bool = True,
        thresholds: list = None,
    ) -> float:
        if self.__opt_set_df is None:
            raise ValueError("Ranking evaluation requires an optimization dataset.")

        if opt_split == "full":
            function_result_df = self.__opt_set_df
        elif opt_split == "train":
            function_result_df = self.__opt_set_train_df
        else:
            function_result_df = self.__opt_set_test_df

        n_correct = 0
        n_total = 0
        for index, row in self.__result_df.iterrows():
            ili = row["ili"]
            lang = row["Language"]
            if thresholds is not None:
                rank = self.map_score_to_rank(
                    row["basicness_score_experimental"], thresholds
                )
            else:
                rank = row["basicness_rank"]
            for i, row_annotation in function_result_df.iterrows():
                if row_annotation["ili"] == ili and row_annotation["Language"] == lang:
                    n_total += 1
                    if row_annotation["basicness_score"] == rank:
                        n_correct += 1
                    # Else if the rank is distant by 1
                    elif (
                        allow_margin_error
                        and abs(row_annotation["basicness_score"] - rank) == 1
                    ):
                        n_correct += 0.5

        return n_correct / n_total

    def evaluate_ordinal_regression_loss(
        self, opt_split: str = "full", thresholds: list = [0.25, 0.5, 0.75]
    ) -> float:
        """
        Evaluate ordinal regression loss for predicted basicness scores against human-annotated ranks.
        Penalizes based on squared differences in rank.

        :param opt_split: Split of the optimization dataset to use for evaluation. Can be "train", "test" or "full".
        :return: The average ordinal regression loss.
        """
        if self.__opt_set_df is None:
            raise ValueError(
                "Ordinal regression loss requires an optimization dataset."
            )

        if opt_split == "full":
            function_result_df = self.__opt_set_df
        elif opt_split == "train":
            function_result_df = self.__opt_set_train_df
        else:
            function_result_df = self.__opt_set_test_df

        predicted_ranks = []
        true_ranks = []

        for index, row in self.__result_df.iterrows():
            ili = row["ili"]
            lang = row["Language"]
            basicness_score = row["basicness_score"]

            # Match row with ground-truth
            for i, row_annotation in function_result_df.iterrows():
                if row_annotation["ili"] == ili and row_annotation["Language"] == lang:
                    true_rank = row_annotation["basicness_score"]

                    # Convert predicted score to ranks (1-4)
                    predicted_rank = map_score_to_rank(basicness_score, thresholds)

                    predicted_ranks.append(predicted_rank)
                    true_ranks.append(true_rank)

        # Convert lists to numpy arrays
        predicted_ranks = np.array(predicted_ranks)
        true_ranks = np.array(true_ranks)

        # Compute squared differences (ordinal loss)
        ordinal_loss = np.mean((predicted_ranks - true_ranks) ** 2)

        return ordinal_loss

    def evaluate_cumulative_ordinal_loss(
        self, opt_split: str = "full", thresholds: list = None, gamma: float = None
    ) -> float:
        """
        Evaluate cumulative ordinal regression loss using cumulative probabilities.

        :param opt_split: Split of the optimization dataset to use for evaluation. Can be "train", "test" or "full".
        :param thresholds: List of thresholds for cumulative probabilities.
        :param gamma: Scaling factor for smoothness of sigmoid.
        :return: The average cumulative ordinal regression loss.
        """
        if thresholds is None:
            thresholds = [0.25, 0.5, 0.75]

        if gamma is None:
            gamma = 0.1

        if self.__opt_set_df is None:
            raise ValueError(
                "Ordinal regression loss requires an optimization dataset."
            )

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        if opt_split == "full":
            function_result_df = self.__opt_set_df
        elif opt_split == "train":
            function_result_df = self.__opt_set_train_df
        else:
            function_result_df = self.__opt_set_test_df

        cumulative_loss = 0
        n_samples = 0

        for index, row in self.__result_df.iterrows():
            ili = row["ili"]
            lang = row["Language"]
            score = row["basicness_score"]

            for _, row_annotation in function_result_df.iterrows():
                if row_annotation["ili"] == ili and row_annotation["Language"] == lang:
                    true_rank = row_annotation["basicness_score"]  # Ground truth rank

                    # Compute cumulative probabilities
                    cumulative_probs = [
                        1 - sigmoid((score - t) / gamma) for t in thresholds
                    ]
                    cumulative_probs = (
                        [0] + cumulative_probs + [1]
                    )  # Add bounds for P(y <= 0) = 0 and P(y <= 4) = 1

                    # Calculate probabilities for each rank
                    rank_probs = [
                        cumulative_probs[i + 1] - cumulative_probs[i]
                        for i in range(len(thresholds) + 1)
                    ]

                    # True rank probability
                    true_rank_prob = rank_probs[int(true_rank) - 1]

                    # Add negative log likelihood loss
                    # The lower the loss, the better the model as it means that the probability of the true rank is higher
                    cumulative_loss += -np.log(true_rank_prob + 1e-8)  # Avoid log(0)

                    n_samples += 1

        return cumulative_loss / n_samples

    def add_basicness_rank(
        self, score_column: str = "basicness_score", thresholds: List[float] = None
    ) -> pd.DataFrame:
        """
        Convert basicness scores to ranks using adaptive k-means clustering or fixed thresholds.

        Args:
            score_column: Column containing the score to rank.
            thresholds: Fixed thresholds [t1, t2, t3] for rank boundaries or None for adaptive clustering.

        Returns:
            DataFrame with added 'basicness_rank' column (4 = most basic, 1 = least basic)
        """
        from sklearn.cluster import KMeans

        ranked_data = self.__result_df.copy()

        if score_column not in ranked_data.columns:
            raise ValueError(f"Score column '{score_column}' not found in DataFrame.")

        if thresholds is not None:
            # Fixed threshold approach
            ranked_data["basicness_rank"] = ranked_data[score_column].apply(
                lambda x: self.map_score_to_rank(x, thresholds)
            )
        else:
            # Group by ILI and apply adaptive ranking
            for ili in ranked_data["ili"].unique():
                group = ranked_data[ranked_data["ili"] == ili]

                if len(group) >= 8:  # Enough data for k-means (2+ per cluster)
                    # Use k-means to find natural clusters in the data
                    scores = group[score_column].values.reshape(-1, 1)
                    kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
                    clusters = kmeans.fit_predict(scores)

                    # Map clusters to ranks (higher centroid = higher rank)
                    centers = kmeans.cluster_centers_.flatten()
                    cluster_rank_map = {
                        i: r for r, i in enumerate(np.argsort(centers), 1)
                    }
                    ranks = np.array([cluster_rank_map[c] for c in clusters])

                    ranked_data.loc[group.index, "basicness_rank"] = ranks
                elif len(group) >= 4:
                    # Enough for quartile ranking
                    ranked_data.loc[group.index, "basicness_rank"] = (
                        group[score_column]
                        .rank(ascending=False, method="dense")
                        .astype(int)
                    )
                else:
                    # Not enough for quartiles, use simple ranking
                    ranks = (
                        group[score_column]
                        .rank(ascending=False, method="dense")
                        .astype(int)
                    )
                    ranked_data.loc[group.index, "basicness_rank"] = ranks

        # Reorder columns
        cols_to_front = ["Language", score_column, "basicness_rank"]
        other_cols = [col for col in ranked_data.columns if col not in cols_to_front]
        ranked_data = ranked_data[cols_to_front + other_cols]

        self.__result_df_ranked = ranked_data
        return ranked_data

    def map_score_to_rank(self, basicness_score: float, thresholds: list = None) -> int:
        """Maps basicness score to rank (4=most basic, 1=least basic)"""
        if thresholds is None:
            thresholds = [
                0.25,
                0.5,
                0.75,
            ]  # Equal intervals (fixed from [0.1, 0.65, 0.75])

        if basicness_score is None or np.isnan(basicness_score):
            return 1

        thresholds = sorted(thresholds)
        for rank, threshold in enumerate(thresholds, start=1):
            if basicness_score <= threshold:
                return rank
        return len(thresholds) + 1
