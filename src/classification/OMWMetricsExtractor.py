"""
Author: Gianluca Barmina
"""

import os
from collections import defaultdict
from datetime import datetime
from typing import Iterable, Union, List
from pron_difficulty import PronDifficulty

import pandas as pd
import ujson as json
import wn

from wn import Synset

from utils_omw import get_omw_synsets, synsets_empty
from utils import get_path_from_root, get_lemma_freq

class OMWMetricsExtractor:
    def __init__(self):
        self.__result_dict = {}
        self.__n_lemmas_lang_dict = {}
        self.__num_synsets = 0
        self.extraction_settings = {}
        self.pronounce_complexity_evaluator = PronDifficulty()

        with open(get_path_from_root("resources/preprocessed_stories_dict.json"), "r") as f:
            self.__children_stories_dict = json.load(f)

        with open(get_path_from_root("resources/second_language_learning_dict.json"), "r") as f:
            self.__second_language_learn_dict = json.load(f)

        # create the folder results/omw if it does not exist
        os.makedirs(get_path_from_root("results/omw"), exist_ok=True)


    def __calculate_pronounce_complexity(self, word: str, lang: str):
        # Evaluate a word's pronunciation difficulty (returns a score from 0 to 1)
        difficulty = self.pronounce_complexity_evaluator.evaluate(word, language=lang)

        return difficulty


    def __extract_lemma_info(self, lemma: str, lang: str):
        """
        Extract the info for a word in a given language
        :param lemma: Lemma for which to extract info
        :param lang: Language of the lemma
        :return: Dictionary with the info: word_length, word_frequency, pronounce_complexity
        """
        metrics = {}
        metrics["word_length"] = len(lemma)

        metrics["word_frequency"] = get_lemma_freq(lemma, lang)
        # Uncomment the following line to calculate the pronounce complexity, excluded because of cost-benefit imbalance
        metrics["pronounce_complexity"] = self.__calculate_pronounce_complexity(lemma, lang)

        metrics["word_in_children_res"] = 1 if lemma in self.__children_stories_dict[lang] else 0
        metrics["word_in_second_lang_learn_res"] = 1 if lemma in self.__second_language_learn_dict[lang] else 0
        metrics["n_senses"] = len(wn.senses(lemma, pos='n', lang=lang))

        return metrics

    def __extract_synset_info(self,
                              synset: Synset,
                              lemmas: List[str],
                              lang: str,
                              max_lemmas: int = 10):
        """
        Extract info for a language subset (lemmas of a given language) of a given synset
        and add them to the result dictionary.
        Current info extracted: definition, number of hyponyms, number of synonyms, ili, word info for each lemma
        :param synset: Synset of which to compute the metrics
        :param lemmas: List of lemmas to consider
        :param lang: Language of the lemmas to consider
        :param max_lemmas: Maximum number of lemmas (of a language) to consider in the synset
        """
        lang_lower = str(lang).lower()
        synset_str = str(synset)

        self.__n_lemmas_lang_dict.setdefault(str(lang).lower(), 0)

        # Consider only the first max_lemmas
        if len(lemmas) > max_lemmas:
            # Sort the lemmas by frequency
            lemmas = sorted(lemmas, key=lambda x: get_lemma_freq(x, lang), reverse=True) # TODO is this ok?
            lemmas = lemmas[:max_lemmas]

        # Get the definition of the synset and add it to the result dictionary
        definition = synset.definition()
        if definition is None:
            definition = "nd"
        self.__result_dict.setdefault(lang_lower, {}).setdefault(synset_str, {}).setdefault("definition", definition)

        # Get the number of hyponyms of the synset, filtering out the inferred ones as no other information is available
        filtered_hyponyms = [h for h in synset.hyponyms() if "*INFERRED*" not in h.id]
        n_hyponyms = len(filtered_hyponyms)

        # Get the number of synonyms of the synset
        n_synonyms = len(synset.lemmas()) # in this case only valid lemmas are considered. Use len(synset.lemmas()) to consider all lemmas

        # Get the total number of senses for all the synset lemmas
        n_syn_senses = 0
        for lemma in synset.lemmas():
            n_syn_senses += len(wn.senses(lemma, pos='n', lang=lang))

        self.__result_dict.setdefault(lang_lower, {}).setdefault(synset_str, {}).setdefault("n_hyponyms", n_hyponyms)
        self.__result_dict.setdefault(lang_lower, {}).setdefault(synset_str, {}).setdefault("n_synonyms", n_synonyms)
        self.__result_dict.setdefault(lang_lower, {}).setdefault(synset_str, {}).setdefault("n_syn_senses", n_syn_senses)

        # TODO (1) eventually finish integrating all lemmas in the dataframe for the "lang_lemmas" column (synonyms).
        #  Now using the pandaSQL query it outputs only a part of them
        # self.__result_dict.setdefault(lang_lower, {}).setdefault(synset_str, {}).setdefault(f"{lang}_words", synset.lemmas())

        for lemma in lemmas:
            lemma_str = str(lemma)

            # TODO review this for a better way of excluding tuples (lang, ili, lemma)
            #  already present in the result dictionary instead of deduplicating the dataframe afterwards
            # Check if the lemma is already present in the result dictionary, if so skip it
            # temp = self.__result_dict[lang_lower]
            #
            # lemma_present = False
            # for syn, item in temp.items():
            #     for lemma_2, item_2 in temp[syn].items():
            #         if lemma_2 != "definition":
            #             if lemma_2 == lemma_str:
            #                 lemma_present = True
            # if lemma_present:
            #     continue
            # End of check

            metrics = self.__extract_lemma_info(lemma_str, lang_lower)

            self.__n_lemmas_lang_dict[lang_lower] += 1
            self.__result_dict.setdefault(lang_lower, {}).setdefault(synset_str, {}).setdefault("ili", synset.ili.id)
            self.__result_dict.setdefault(lang_lower, {}).setdefault(synset_str, {}).setdefault(lemma_str, {})
            for key, value in metrics.items():
                self.__result_dict[lang_lower][synset_str][lemma_str][key] = value

        if len(self.__result_dict[lang_lower]) == 0:
            print(f"Warning: no lemmas matching the set conditions for synset {synset_str} in language {lang_lower}")

    def extract(self,
                input: Union[Iterable[Synset], Synset, str, Iterable[str], pd.DataFrame],
                languages: Iterable[str] = ("en", "it", "en", "nb"),
                max_lemmas: int = 10,
                filter_zero_freq: bool = False,
                freq_threshold: float = 0.0,
                verbose: bool = False,
                reset_data: bool = True,
                json_path: str = None,
                csv_path: str = None,
                use_ili_list: bool = False) -> dict:
        """
        For each synset retrieved, if it has lemmas in all the specified languages
        and all the eventual filtering condition are met, extract info for it and its lemmas.
        For each language each synset is specified with the following info: n_hyponyms, n_synonyms.
        For each synset each lemma is specified with the following info: word_length, word_frequency, pronounce_complexity
        :param input: If str or Iterable[str], these are used as word or list of words to retrieve related synsets
        from Open Multilingual Wordnet (OMW) resources.
        If Synset or Iterable[Synset], these are used as starting synsets from which corresponding synsets
        in the languages specified are retrieved and their information extracted.
        :param languages: Iterable of str to consider following the ISO 639 standard
        :param max_lemmas: Maximum number of lemmas to consider for each synset
        :param filter_zero_freq: If True, lemmas with zero frequency are skipped
        :param freq_threshold: Minimum frequency threshold for a lemma to be considered
        :param verbose: If True, print some information about the data retrieved
        :param reset_data: If True, reset the eventual data previously extracted before starting a new extraction
        :param json_path: If specified, export the data to a json file
        :param csv_path: If specified, export the data to a csv file
        :return: Dictionary with the extracted info. The structure is the following:
        Language -> Synsets for each language -> Info for each synset + Lemmas for each synset -> Info for each lemma
        """

        if reset_data:
            self.__reset_data()

        # Handle the input according to its type getting synsets
        lang_syn_dict, syn_lemmas_dict = get_omw_synsets(input, langs=list(languages), poses=['n'],
                                                         use_ili=True, match_exact_ili=True,
                                                         filter_zero_freq=filter_zero_freq,
                                                         freq_threshold=freq_threshold,
                                                         use_ili_list=use_ili_list)

        if synsets_empty(lang_syn_dict):
            print("No synsets found for the given input")
            return self.__result_dict

        # Populate the extraction settings
        self.populate_extraction_settings(input, languages,
                                          max_lemmas, filter_zero_freq, freq_threshold, verbose,
                                          reset_data)

        for lang, synsets in lang_syn_dict.items():
            for synset in synsets:
                lemmas = syn_lemmas_dict[synset.id]
                self.__num_synsets += 1
                self.__extract_synset_info(synset, lemmas, lang, max_lemmas)

        if verbose:
            self.__print_verbose()
        self.__output_settings()

        # Remove duplicates and reconstruct the dictionary
        deduplicated_df = self.get_deduplicated_df()
        self.__reconstruct_dict(deduplicated_df)

        if json_path:
            self.export_json(json_path)
        if csv_path:
            self.export_csv(csv_path)

        return self.__result_dict

    def export_json(self, file_path: str):
        """
        Export the extracted metrics dictionary to a json file
        :param file_path: path of the destination json file
        """
        json_str = json.dumps(self.__result_dict, indent=4)

        with open(file_path, "w") as f:
            f.write(json_str)

    def export_csv(self, file_path: str):
        """
        Export the extracted metrics dictionary to a csv file
        :param file_path: Path of the destination csv file
        """
        df = self.__convert_to_df()
        df.to_csv(file_path, index=False)

    def __convert_to_df(self) -> pd.DataFrame:
        data = []
        # langs = self.__result_dict.keys()
        for language, synsets in self.__result_dict.items():
            for synset, synset_info in synsets.items():
                gloss = synset_info["definition"]
                ili = synset_info["ili"]
                n_hyponyms = synset_info["n_hyponyms"]
                n_synonyms = synset_info["n_synonyms"]
                n_syn_senses = synset_info["n_syn_senses"]
                # TODO (1)
                # synonyms_dict = defaultdict(list)
                # for lang in langs:
                #     synonyms_dict[lang] = synset_info[lang + "_words"]
                # Skip first elements of synset_info as they are the definition, ili, n_hyponyms and n_synonyms
                n_to_skip = 5
                for lemma, metrics in list(synset_info.items())[n_to_skip:]:
                    row = {"Language": language, "Synset": synset, "ili": ili, "Gloss": gloss,
                           "Lemma": lemma, "n_hyponyms": n_hyponyms, "n_synonyms": n_synonyms,
                           "n_syn_senses": n_syn_senses
                           }
                    # row.update(synonyms_dict)
                    row.update(metrics)
                    data.append(row)

        return pd.DataFrame(data)

    def __reset_data(self):
        self.__result_dict = {}
        self.__n_lemmas_lang_dict = {}
        self.__num_synsets = 0
        self.extraction_settings = {}

    def __print_verbose(self):
        print(f"|----| Extraction results |----|")
        print("Number of valid synsets: ", self.__num_synsets)
        print("Number of valid lemmas per language: ", self.__n_lemmas_lang_dict)

    def populate_extraction_settings(self, input, languages, max_lemmas,
                                     filter_zero_freq, freq_threshold, verbose, reset_data):
        """
        Populate the extraction settings dictionary with the provided parameters
        """
        self.extraction_settings["input"] = str(input)
        self.extraction_settings["languages"] = [str(lang) for lang in languages]
        self.extraction_settings["max_lemmas"] = max_lemmas
        self.extraction_settings["filter_zero_freq"] = filter_zero_freq
        self.extraction_settings["freq_threshold"] = freq_threshold
        self.extraction_settings["verbose"] = verbose
        self.extraction_settings["reset_data"] = reset_data

    def __output_settings(self):
        """
        Append the extraction settings to a file in json format specifying date and time of the execution
        """
        settings = {
            "datetime": str(datetime.now()),
            "num_synsets": self.__num_synsets,
            "num_lemmas_for_lang": self.__n_lemmas_lang_dict
        }
        settings.update(self.extraction_settings)

        os.makedirs(get_path_from_root("results"), exist_ok=True)
        file_path = get_path_from_root("results/extraction_settings.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                data = json.load(f)
        else:
            data = []

        data.append(settings)

        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

    def get_deduplicated_df(self) -> pd.DataFrame:
        """
        Remove from the result dictionary the duplicates which have the same 'Lemma', 'Language' and 'ili' values
        :return: DataFrame with the deduplicated data
        """
        df = self.__convert_to_df()

        # Convert "nd" strings to None for proper None handling in 'Gloss' column
        df['Gloss'].replace({'nd'}, None, inplace=True)

        # Sort by 'Gloss' so that non-None Gloss rows come before None ones, then drop duplicates by 'Lemma' and 'Language'
        deduplicated_data_with_headers = df.sort_values(by='Gloss', na_position='last').drop_duplicates(
            subset=['Lemma', 'Language', 'ili'], keep='first')

        return deduplicated_data_with_headers

    def __reconstruct_dict(self, deduplicated_df):
        """
        Reconstruct the result dictionary, according to its original structure, from the deduplicated DataFrame.
        :param deduplicated_df: DataFrame with the deduplicated data
        """
        # Initialize the nested dictionary structure
        result_dict = defaultdict(lambda: defaultdict(dict))

        for _, row in deduplicated_df.iterrows():
            language = row['Language']
            synset = row['Synset']
            gloss = row['Gloss']
            ili = row['ili']
            n_hyponyms = row['n_hyponyms']
            n_synonyms = row['n_synonyms']
            n_syn_senses = row['n_syn_senses']
            lemma = row['Lemma']
            # TODO (1)

            # Add the main structure for the synset
            if 'definition' not in result_dict[language][synset]:
                result_dict[language][synset]['definition'] = gloss
                result_dict[language][synset]['ili'] = ili
                result_dict[language][synset]['n_hyponyms'] = n_hyponyms
                result_dict[language][synset]['n_synonyms'] = n_synonyms
                result_dict[language][synset]['n_syn_senses'] = n_syn_senses

            # Add lemma metrics
            result_dict[language][synset][lemma] = {
                'word_length': row['word_length'],
                'word_frequency': row['word_frequency'],
                'pronounce_complexity': row['pronounce_complexity'],
                'word_in_children_res': row['word_in_children_res'],
                'word_in_second_lang_learn_res': row['word_in_second_lang_learn_res'],
                'n_senses': row['n_senses']
                # Add other metrics as needed
            }

        self.__result_dict = result_dict


