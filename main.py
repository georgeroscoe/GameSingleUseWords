import urllib
from collections import defaultdict
from functools import reduce
from typing import Any, Dict, List, Tuple
import logging
import requests
from nltk.corpus import stopwords
from nltk.corpus import words as wd
from nltk.stem import WordNetLemmatizer

logging.basicConfig(filename='debug.log', level=logging.DEBUG)

wordnet_lemmatizer = WordNetLemmatizer()


def get_response_list(word: str, position: str, num_words: int) -> List[Dict[str, Any]]:
    """
        Given a word, a position, and the number of words to include, this function returns a list of words
        that appear in the same context as the given word.

        Args:
            word (str): The input word.
            position (str): A string that indicates whether to look for words that appear before or after the input word.
            num_words (int): The number of words to include in the result.

        Returns:
            List[str]: A list of words that appear in the same context as the input word.
    """
    if position == 'before':
        bracket_words = ' '.join([num_words * '?'])
        encoded_query = urllib.parse.quote(f"{bracket_words} {word}")
    else:
        bracket_words = ' '.join([num_words * '?'])
        encoded_query = urllib.parse.quote(f"{word} {bracket_words}")

    params = {'corpus': 'eng-gb', 'query': encoded_query}
    params = '&'.join(f"{name}={value}" for name, value in params.items())
    try:
        response = requests.get('https://api.phrasefinder.io/search?' + params)
        response_json = response.json()
    except requests.exceptions.RequestException as e:
        logging.error(str(e))
        return []
    else:
        response_list = response_json['phrases']
        return response_list


def is_word(word: str) -> bool:
    """
    Checks whether a given string is a word.

    Args:
        word (str): The word to check.

    Returns:
        bool: True if the string is a word, False otherwise.
    """
    if word is None or word not in wd.words():
        return False
    else:
        return True


def is_not_stopword(word: str) -> bool:
    """
    Checks whether a given word is a stop word.

    Args:
        word (str): The word to check.

    Returns:
        bool: True if the word is not a stop word, False otherwise.
    """
    stop_words = set(stopwords.words('english'))
    return word not in stop_words


def collect_responses(response_list: List[Dict[str, Any]], position: str, num_words: int) -> List[Tuple[str, float]]:
    """
    Collects the cleaned responses from the given list of responses.

    Args:
        response_list (List[Dict[str, Any]]): A list of phrases containing the given word or phrase.
        position (str): Indicates whether the word or phrase appears before or after the unknown words.
        num_words (int): The number of unknown words to search for.

    Returns:
        List[Tuple[str, float]]: A list of cleaned responses.
    """
    cleaned_responses = []

    for response in response_list:
        answer_words = response['tks'][1:] if position == 'after' else response['tks'][-1 - num_words:-1]
        answer = ' '.join([i['tt'] for i in answer_words])
        if response['sc'] > 0.001 and all(is_word(a) for a in [i['tt'] for i in answer_words]):
            cleaned_responses.append((answer, response['mc']))
    return cleaned_responses


def lemmatize_and_sum(words: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """
    Lemmatizes the words in the given list and returns a new list of tuples
    containing the lemmatized words and their corresponding scores.

    Args:
        words (List[Tuple[str, float]]): A list of words and their corresponding scores.

    Returns:
        List[Tuple[str, float]]: A new list of tuples containing the lemmatized words and their scores.
    """
    lemmatized_lst = [(wordnet_lemmatizer.lemmatize(word[0]), word[1]) for word in words]
    d = defaultdict(float)
    for k, v in lemmatized_lst:
        d[k] += v
    return sorted(d.items(), key=lambda x: x[1], reverse=True)


def score_generator(words: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """
    Generates a new list of tuples containing the words from the input list,
    along with their corresponding scores as a percentage of the total score.

    Args:
        words (List[Tuple[str, float]]): A list of words and their corresponding scores.

    Returns:
        List[Tuple[str, float]]: A new list of tuples containing the words and their scores as a percentage.

    """
    total_score = sum(w[1] for w in words)
    return [(w[0], (w[1] / total_score) * 100) for w in words]


def before_or_after() -> str:
    """
    Asks the user to input whether to look for words that appear before or after the input word.

    Returns:
    str: Either "before" or "after", depending on the user's input.
    """
    while True:
        word_type = input("Type B for before word, or A for after: ").lower()
        if word_type == "b":
            return 'before'
        elif word_type == "a":
            return 'after'
        print("Invalid character, please try again: ")


def add_stopword_result(scores: List[Tuple[str, float]]) -> List[Tuple[str, float, bool]]:
    """
    Adds a third element to each tuple in the input list, indicating whether
    the corresponding word is a stop word or not.
    Args:
    scores (List[Tuple[str, float]]): A list of words and their corresponding scores.

    Returns:
        List[Tuple[str, float, bool]]: A new list of tuples with a third element indicating whether
                                       the corresponding word is a stop word or not.
    """
    return [(w[0], w[1], is_not_stopword(w[0])) for w in scores]


def how_many_more_words() -> int:
    """
    Asks the user how many unknown words to search for.
    Returns:
        int: The number of unknown words to search for.
    """
    return int(input("How many words in brackets: "))


def run_with_input(phrase: str) -> List[Tuple[str, float]]:
    params = {'position': 'before', 'num_words': 1}
    response_list = get_response_list(phrase, **params)
    responses = collect_responses(response_list, **params)
    answer = lemmatize_and_sum(responses)
    score = score_generator(answer)
    return score


def run_module() -> List[Tuple[str, float, bool]]:
    input_word = input("Please input word: ")
    list_position = before_or_after()
    num_words = how_many_more_words()
    response_list = get_response_list(input_word, list_position, num_words)
    responses = collect_responses(response_list, list_position, num_words)
    answer = lemmatize_and_sum(responses)
    score = score_generator(answer)
    scores_with_result = add_stopword_result(score)
    results_str = [f"  {w[0]} {w[1]:.2f}%{' STOPWORD' if not w[2] else ''}" for w in scores_with_result]
    print('\n'.join(results_str))
    return scores_with_result


if __name__ == "__main__":
    run_module()
