"""
Author: Gianluca Barmina
"""

import os
from typing import Iterable

from wordfreq import word_frequency, zipf_frequency

import pandas as pd
from pandasql import sqldf

def get_path_from_root(path: str) -> str:
    # split the path into parts
    parts = path.split(os.sep)

    # Get the root directory of the project
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Construct the path relative to the root directory
    file_path = os.path.join(project_root, *parts)

    return file_path

def save_dataframe_to_csv(dataframe: pd.DataFrame, file_name):
    file_path = get_path_from_root(f"/analysis/{file_name}")
    dataframe.to_csv(file_path, index=False)

def get_resource_word_match(resource_path: str, words_path: str) -> dict:
    """
    This function takes in two csv files, one containing a resource and the other containing words.
    It matches every row in the resource file with the corresponding row in the words file.
    For example the resource file could contain a series of synsets ids or ILIs ids
    and the words file contains their corresponding words.
    :param resource_path: path to the resource file
    :param words_path: path to the words file
    :return: a dictionary containing the resource as the key and the word as the value
    """
    resource_df = pd.read_csv(resource_path)
    words_df = pd.read_csv(words_path)

    # Create dictionaries for each column pairing
    columns = words_df.columns
    resource_word_dict = {}

    # Loop through each column to pair synsets with words
    for col in columns:
        resource_word_dict[col] = dict(zip(resource_df[col], words_df[col]))

    return resource_word_dict

def analyze_lang_syn_group_babel(word: str, df: pd.DataFrame) -> pd.DataFrame:
    df['word_count'] = df.groupby(['Language', 'Synset'])['word_length'].transform('count')

    all_metrics_lang_syn_query = """
        WITH plain_metrics AS (
            SELECT Language, Synset, 
                   AVG(word_length) AS avg_word_length,
                   AVG(word_frequency) AS avg_word_frequency
            FROM df
            GROUP BY Language, Synset
        ),
        normalized AS (
            SELECT p.Language, p.Synset,
                   p.avg_word_length,
                   p.avg_word_frequency,
                   (p.avg_word_length - (SELECT MIN(avg_word_length) FROM plain_metrics)) / 
                   ((SELECT MAX(avg_word_length) FROM plain_metrics) - (SELECT MIN(avg_word_length) FROM plain_metrics)) 
                   AS normalized_length,
    
                   (p.avg_word_frequency - (SELECT MIN(avg_word_frequency) FROM plain_metrics)) / 
                   ((SELECT MAX(avg_word_frequency) FROM plain_metrics) - (SELECT MIN(avg_word_frequency) FROM plain_metrics)) 
                   AS normalized_frequency
            FROM plain_metrics p
        ),
        combined_metrics AS (
            SELECT Language, Synset, 
                   avg_word_length, 
                   avg_word_frequency, 
                   normalized_length, 
                   normalized_frequency,
                   (0.7 * avg_word_frequency - 0.3 * avg_word_length) AS combined_metric_weighted_sum,
                   (0.7 * normalized_frequency - 0.3 * normalized_length) AS combined_metric_normalized
            FROM normalized
        ),
        min_max_values AS (
            SELECT 
                MIN(combined_metric_weighted_sum) AS min_weighted_sum,
                MAX(combined_metric_weighted_sum) AS max_weighted_sum,
                MIN(combined_metric_normalized) AS min_combined_normalized,
                MAX(combined_metric_normalized) AS max_combined_normalized
            FROM combined_metrics
        )
    SELECT 
        c.Language, 
        c.Synset, 
        (c.combined_metric_normalized - m.min_combined_normalized) / (m.max_combined_normalized - m.min_combined_normalized) AS combined_normalized,
        c.avg_word_length, 
        c.avg_word_frequency, 
        c.normalized_length, 
        c.normalized_frequency
    FROM 
        combined_metrics c
    CROSS JOIN 
        min_max_values m
    ORDER BY 
        c.Synset, combined_normalized DESC;
    """
    # Sorted by synset and combined_metric_weighted_sum, descending, higher values first means the synset is more "basic" for the corresponding language

    all_metrics_lang_syn_df = sqldf(all_metrics_lang_syn_query, locals())
    # all_metrics_lang_syn_df['combined_metric_weighted_sum'] = (
    #     all_metrics_lang_syn_df['combined_metric_weighted_sum'].abs())
    save_dataframe_to_csv(all_metrics_lang_syn_df, f"{word}_lang_syn.csv")
    return all_metrics_lang_syn_df

def analyze_lang_group_babel(word: str, df: pd.DataFrame) -> pd.DataFrame:
    df['word_count'] = df.groupby(['Language'])['word_length'].transform('count')
    df['word_count_synsets'] = df.groupby(['Language', 'Synset'])['word_length'].transform('count')

    all_metrics_lang_query = """
        WITH plain_metrics AS (
        SELECT Language, 
               AVG(word_length) AS avg_word_length,
               AVG(word_frequency) AS avg_word_frequency
        FROM df
        GROUP BY Language
    ),
    normalized AS (
        SELECT p.Language,
               p.avg_word_length,
               p.avg_word_frequency,
               (p.avg_word_length - (SELECT MIN(avg_word_length) FROM plain_metrics)) / 
               ((SELECT MAX(avg_word_length) FROM plain_metrics) - (SELECT MIN(avg_word_length) FROM plain_metrics)) 
               AS normalized_length,

               (p.avg_word_frequency - (SELECT MIN(avg_word_frequency) FROM plain_metrics)) / 
               ((SELECT MAX(avg_word_frequency) FROM plain_metrics) - (SELECT MIN(avg_word_frequency) FROM plain_metrics)) 
               AS normalized_frequency
        FROM plain_metrics p
    ),
    combined_metrics AS (
        SELECT Language, 
               avg_word_length, 
               avg_word_frequency, 
               normalized_length, 
               normalized_frequency,
               (0.5 * avg_word_frequency - 0.5 * avg_word_length) AS combined_metric_weighted_sum,
               (0.5 * normalized_frequency - 0.5 * normalized_length) AS combined_metric_normalized
        FROM normalized
    ),
    min_max_values AS (
        SELECT 
            MIN(combined_metric_weighted_sum) AS min_weighted_sum,
            MAX(combined_metric_weighted_sum) AS max_weighted_sum,
            MIN(combined_metric_normalized) AS min_combined_normalized,
            MAX(combined_metric_normalized) AS max_combined_normalized
        FROM combined_metrics
    )
SELECT 
    c.Language, 
    (c.combined_metric_normalized - m.min_combined_normalized) / (m.max_combined_normalized - m.min_combined_normalized) AS combined_normalized,
    c.avg_word_length, 
    c.avg_word_frequency, 
    c.normalized_length, 
    c.normalized_frequency
FROM 
    combined_metrics c
CROSS JOIN 
    min_max_values m
ORDER BY 
    combined_normalized DESC;
    """

    all_metrics_lang_df = sqldf(all_metrics_lang_query, locals())
    save_dataframe_to_csv(all_metrics_lang_df, f"{word}_lang.csv")
    return all_metrics_lang_df

def phonemizer_lang_string(lang: str):
    """
    Convert the language string to the language string requested by phonemizer library
    """
    if lang == "no":
        lang = "nb"
    elif lang == "en":
        lang = "en-us"

    return lang

def wordfreq_lang_string(lang: str):
    """
    Convert the language string to the language string requested by wordfreq library
    """
    if lang == "no":
        lang = "nb"
    return lang

def get_lemma_freq(lemma: str, lang: str) -> float:
    whitespaced_word = lemma.replace("_", " ")
    wordfreq_lang = wordfreq_lang_string(lang)
    word_freq = word_frequency(whitespaced_word, wordfreq_lang)
    formatted_freq = float(format(word_freq, '.10f'))
    return formatted_freq

def get_lemma_zipf_freq(lemma: str, lang: str) -> float:
    whitespaced_word = lemma.replace("_", " ")
    wordfreq_lang = wordfreq_lang_string(lang)
    zipf_freq = zipf_frequency(whitespaced_word, wordfreq_lang)
    formatted_zipf_freq = float(format(zipf_freq, '.10f'))
    return formatted_zipf_freq

def is_iterable_of_strings(obj) -> bool:
    return isinstance(obj, Iterable) and all(isinstance(item, str) for item in obj)

def get_iso_639_lang(lang: str) -> str:
    """
    Convert the language string to the ISO 639 language code
    """
    lang_dict = {
        "English": "en",
        "Spanish": "es",
        "Italian": "it",
        "Norwegian": "nb"
    }
    return lang_dict[lang]