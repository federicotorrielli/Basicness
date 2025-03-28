import ast
from collections import defaultdict
from typing import List, Dict, Iterable, Union, Tuple

import wn
from wn import Synset

import pandas as pd

from utils import is_iterable_of_strings, get_lemma_freq, get_path_from_root


# TODO: update documentation
def get_omw_synsets(input: Union[Iterable[Synset], Synset, str, Iterable[str], pd.DataFrame],
                    langs: List[str] = ["en", "it", "nb", "es"],
                    poses: List[str] = ["n"],
                    use_ili: bool = True,
                    match_exact_ili: bool = True,
                    filter_zero_freq: bool = True,
                    freq_threshold: float = 0.0,
                    ignore_language_match: bool = False,
                    use_ili_list: bool = False) -> Tuple[Dict[str, List[Synset]], Dict[str, List[str]]]:
    """
    Get the Open Multilingual Wordnet (OMW) valid synsets from the input (according to its type)
    and get the valid lemmas for each synset.
    :param input: If str, this is used as query word to retrieve synsets from OMW resources.
        If Synset, this is used as reference to get the ili and retrieve synsets from OMW resources.
        If Iterable of str, each word is used as query word to retrieve synsets from OMW resources.
        If Iterable of Synset, each synset is used as reference to get the ili and retrieve synsets from OMW resources.
    :param langs: List of languages to consider written as str following ISO 639 standard
    :param poses: List of part of speech tags to consider written as str following WN standard
    :param use_ili: If True, the Interlingual Index (ili) is used to retrieve synsets from OMW resources
    :param match_exact_ili: If True, only synsets exactly maching ili of input are considered
    :param filter_zero_freq: If True, lemmas with zero frequency are filtered out
    :param freq_threshold: If greater than 0.0, lemmas with frequency below this threshold are filtered out
    :param ignore_language_match: If True, all languages synsets are considered even if they do not have a synset for the same ili
    :return: Tuple of two dictionaries.
    The first dictionary has the languages as keys and the corresponding synsets as values.
    The second dictionary has the synset ids as keys and the corresponding valid lemmas as values.
    """
    en_synsets = []
    syn_lemmas_dict = defaultdict(list)

    all_synsets: Dict[str, List[Synset]] = {lang: [] for lang in langs} # init with empty lists

    if isinstance(input, str):
        for pos in poses:
            en_synsets.extend(wn.synsets(input, lang="en", pos=pos))
    elif use_ili_list:
        return get_omw_synsets_from_ilis(input, langs, filter_zero_freq, freq_threshold)
    elif isinstance(input, pd.DataFrame):
        return get_omw_synsets_from_ili_df(input, langs, filter_zero_freq, freq_threshold)
    elif isinstance(input, Synset):
        en_synsets.append(input)
    elif is_iterable_of_strings(input):
        for word in input:
            for pos in poses:
                en_synsets.extend(wn.synsets(word, lang="en", pos=pos))
    else:
        en_synsets = input

    # Filter out synsets with no valid lemmas in english
    for en_syn in en_synsets:
        valid_lemmas = get_valid_lemmas(en_syn, "en", filter_zero_freq, freq_threshold)
        if len(valid_lemmas) > 0:
            syn_lemmas_dict[en_syn.id] = valid_lemmas
        else:
            en_synsets.remove(en_syn)

    # For each synset in english, get the corresponding valid synsets in the other languages
    for en_syn in en_synsets:
        ili = en_syn.ili.id
        temp_synsets = {lang: [] for lang in langs}
        temp_synsets["en"] = [en_syn]
        lang_count = 1
        for lang in langs:
            if lang != "en": # skip english synsets as we already put them in the temp_synsets dict
                if use_ili:
                    synsets = wn.synsets(ili=ili, lang=lang) # get synsets explicitly by ili
                else:
                    synsets = en_syn.translate(lang=lang) # get synsets by translation provided by OMW
                if match_exact_ili:
                    synsets = filter_ili_synsets(synsets, ili) # select only synsets with the exact ili
                if len(synsets) > 0: # if there are synsets for the word in the language
                    exists_valid_synset = False # flag to check if there is at least one valid synset which satisfies the conditions

                    # For each synset, get the lemmas matching the conditions and signal if there is at least one valid synset
                    for s in synsets:
                        valid_lemmas = get_valid_lemmas(s, lang, filter_zero_freq, freq_threshold)
                        if len(valid_lemmas) > 0:
                            exists_valid_synset = True
                            temp_synsets[lang].append(s)
                            syn_lemmas_dict[s.id] = valid_lemmas

                    if exists_valid_synset:
                        lang_count += 1

        # if all languages have synsets with valid lemmas for the same ili merge temp_synsets into all_synsets
        if lang_count == len(langs) or ignore_language_match:
            for lang in langs:
                all_synsets[lang].extend(temp_synsets[lang])

    return all_synsets, syn_lemmas_dict

def get_omw_synsets_from_ili_df(input: pd.DataFrame,
                                langs: List[str] = ["en", "it", "nb", "es"],
                                filter_zero_freq: bool = True,
                                freq_threshold: float = 0.0) -> Tuple[Dict[str, List[Synset]], Dict[str, List[str]]]:
    all_synsets: Dict[str, List[Synset]] = {lang: [] for lang in langs} # init with empty lists
    syn_lemmas_dict = defaultdict(list)

    for index, row in input.iterrows():
        ili_list = row["ilis"]
        for ili in ili_list:
            for lang in langs:
                synsets = wn.synsets(ili=ili, lang=lang, pos="n")
                if len(synsets) > 0:
                    valid_synsets = []
                    for syn in synsets:
                        valid_lemmas = get_valid_lemmas(syn, lang, filter_zero_freq, freq_threshold)
                        if len(valid_lemmas) > 0:
                            valid_synsets.append(syn)
                    if len(valid_synsets) > 0:
                        all_synsets[lang].extend(valid_synsets)
                        for syn in valid_synsets:
                            syn_lemmas_dict[syn.id] = get_synset_lemmas(syn)

    return all_synsets, syn_lemmas_dict

def get_omw_synsets_from_ilis(ili_list: Iterable[str],
                                langs: List[str] = ["en", "it", "nb", "es"],
                                filter_zero_freq: bool = True,
                                freq_threshold: float = 0.0) -> Tuple[Dict[str, List[Synset]], Dict[str, List[str]]]:
    all_synsets: Dict[str, List[Synset]] = {lang: [] for lang in langs} # init with empty lists
    syn_lemmas_dict = defaultdict(list)

    # Remove duplicates
    ili_list = list(set(ili_list))

    for ili in ili_list:
        for lang in langs:
            synsets = wn.synsets(ili=ili, lang=lang, pos="n")
            if len(synsets) > 0:
                valid_synsets = []
                for syn in synsets:
                    valid_lemmas = get_valid_lemmas(syn, lang, filter_zero_freq, freq_threshold)
                    if len(valid_lemmas) > 0:
                        valid_synsets.append(syn)
                if len(valid_synsets) > 0:
                    all_synsets[lang].extend(valid_synsets)
                    for syn in valid_synsets:
                        syn_lemmas_dict[syn.id] = get_synset_lemmas(syn)

    return all_synsets, syn_lemmas_dict

def filter_ili_synsets(synsets: List[Synset], ili: str) -> List[Synset]:
    """
    Filter synsets by ili. Only synsets with the exact ili are kept.
    :param synsets: The list of synsets to filter
    :param ili: The ili to filter by
    :return: List of synsets with the exactly matching the ili passed
    """
    filtered_synsets = []
    for syn in synsets:
        if syn.ili.id == ili:
            filtered_synsets.append(syn)
    return filtered_synsets

def get_valid_lemmas(synset: Synset, lang: str, filter_zero_freq: bool = True, freq_threshold: float = 0.0):
    """
    Get the valid lemmas for a synset in a given language.
    :param synset: The synset to get the lemmas from
    :param lang: The language of the synset
    :param filter_zero_freq: If True, lemmas with zero frequency are filtered out
    :param freq_threshold: If greater than 0.0, lemmas with frequency below this threshold are filtered out
    :return: List of valid lemmas according to previous conditions
    """
    lemmas = get_synset_lemmas(synset)

    if freq_threshold > 0.0:
        for lemma in lemmas:
            if get_lemma_freq(lemma, lang) < freq_threshold:
                lemmas.remove(lemma)
    elif filter_zero_freq:
        for lemma in lemmas:
            if get_lemma_freq(lemma, lang) == 0.0:
                lemmas.remove(lemma)

    return lemmas

def get_synset_lemmas(synset: Synset) -> List[str]:
    lemmas = synset.lemmas()
    # To lower case and remove duplicates
    lemmas = list(set([lemma.lower() for lemma in lemmas]))
    return lemmas

def synsets_empty(lang_syn_dict: Dict[str, List[Synset]]) -> bool:
    for lang, synsets in lang_syn_dict.items():
        if len(synsets) == 0:
            return True
    return False

def print_omw_synsets(synsets: Union[List[Synset], Dict[str, List[Synset]]], add_tab: bool = False):
    if isinstance(synsets, dict):
        for lang in synsets:
            print(lang)
            print(f"\tlen: {len(synsets[lang])}")
            print_synset_list(synsets[lang], add_tab)
            print()
    else:
        print(f"\tlen: {len(synsets)}")
        print_synset_list(synsets, add_tab)

def print_synset_list(synsets: List[Synset], add_tab: bool = False):
    for s in synsets:
        if add_tab:
            print(f"\t{s}")
            print(f"\t\tdef: {s.definition()}")
            print(f"\t\tili: {s.ili} status: {s.ili.status}")
            print(f"\t\tlemmas: {s.lemmas()}")
        else:
            print(s)
            print(f"\tdef: {s.definition()}")
            print(f"\tili: {s.ili} status: {s.ili.status}")
            print(f"\tlemmas: {s.lemmas()}")

def merge_culture_dfs(langs: List[str]) -> defaultdict:
    """
    Merge the best cultural ilis for each language into a single DataFrame.
    :param langs: List of languages to consider
    :return: DataFrame with columns 'Language', 'ili', 'basicness_score'
    """
    best_culture_ilis = defaultdict(list)
    for lang in langs:
        culture_df = pd.read_csv(get_path_from_root(f"resources/culture/ili_{lang}_culture.csv"))
        culture_df["ilis"] = culture_df["ilis"].apply(ast.literal_eval)
        for index, row in culture_df.iterrows():
            ilis = row["ilis"]
            best_culture_ilis[lang].extend(ilis)

    return best_culture_ilis

def get_ilis_from_lemmas(lemmas_df: pd.DataFrame, lang: str, lemmas_in_lang: bool = False, use_translate: bool = True) -> pd.DataFrame:
    """
    Get the Interlingual Indices (ili) corresponding to the input lemmas in the given language.
    :param lemmas_df: DataFrame with a column 'lemma' containing the lemmas to get the ili from
    :param lang: The language of the lemmas
    :return: DataFrame with a column lemma a column ilis. Where each row contains a lemma and its corresponding list of ilis.
    """
    lemmas_ilis = []
    for lemma in lemmas_df["lemma"]:
        ilis = []
        if lemmas_in_lang:
            synsets = wn.synsets(lemma, lang=lang, pos="n")
            ilis = [syn.ili.id for syn in synsets]
        else:
            synsets = wn.synsets(lemma, lang="en", pos="n")
            if lang == "en":
                ilis = [syn.ili.id for syn in synsets]
            else:
                if use_translate:
                    translated_synsets_list = [syn.translate(lang=lang) for syn in synsets]
                    ilis = [syn.ili.id for syn_list in translated_synsets_list for syn in syn_list]
                else:
                    ilis_en = [syn.ili.id for syn in synsets]
                    for ili in ilis_en:
                        synsets_lang = wn.synsets(ili=ili, lang=lang)
                        ilis.extend([syn.ili.id for syn in synsets_lang])
        # remove duplicates
        ilis = list(set(ilis))
        lemmas_ilis.append({"lemma": lemma, "ilis": ilis})
    return pd.DataFrame(lemmas_ilis)

### Test functions

def get_en_ili_synsets(ili: str, pos: str = "n") -> List[Synset]:
    synsets = wn.synsets(lang="en", pos=pos, ili=ili)

    return synsets