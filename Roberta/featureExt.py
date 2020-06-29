import nltk
import string
from nltk.corpus import stopwords

def unique_word_fraction(text):
    """function to calculate the fraction of unique words on total words of the text"""
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    unique_count = list(set(text_splited)).__len__()
    if word_count == 0:
        return 0
    return (unique_count/word_count)


eng_stopwords = set(stopwords.words("english"))
def stopwords_count(text):
    """ Number of stopwords fraction in a text"""
    text = text.lower()
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    stopwords_count = len([w for w in text_splited if w in eng_stopwords])
    if word_count == 0:
        return 0
    return (stopwords_count/word_count)


def punctuations_fraction(text):
    """functiopn to claculate the fraction of punctuations over total number of characters for a given text """
    char_count = len(text)
    punctuation_count = len([c for c in text if c in string.punctuation])
    if char_count == 0:
        return 0
    return (punctuation_count/char_count)

def char_count(text):
    """function to return number of chracters """
    return len(text)

def fraction_noun(text):
    """function to give us fraction of noun over total words """
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    pos_list = nltk.pos_tag(text_splited)
    noun_count = len([w for w in pos_list if w[1] in ('NN','NNP','NNPS','NNS')])
    if word_count == 0:
        return 0
    return (noun_count/word_count)

def fraction_adj(text):
    """function to give us fraction of adjectives over total words in given text"""
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    pos_list = nltk.pos_tag(text_splited)
    adj_count = len([w for w in pos_list if w[1] in ('JJ','JJR','JJS')])
    if word_count == 0:
        return 0
    return (adj_count/word_count)

def fraction_verbs(text):
    """function to give us fraction of verbs over total words in given text"""
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    pos_list = nltk.pos_tag(text_splited)
    verbs_count = len([w for w in pos_list if w[1] in ('VB','VBD','VBG','VBN','VBP','VBZ')])
    if word_count == 0:
        return 0
    return (verbs_count/word_count)

def process_data(text, selected_text):
    idx = text.find(selected_text)
    idx = len(text[:idx].split())
    targets_start = max(0, idx)
    targets_end = targets_start + len(selected_text.split()) + 1
    return targets_start, targets_end