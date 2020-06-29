import numpy as np
import torch

def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    if len(a) == 0 and len(b) == 0:
        return 1
    return float(len(c)) / (len(a) + len(b) - len(c))

def calculate_jaccard_score(
    original_tweet,
    target_string,
    sentiment_val,
    idx_start,
    idx_end,
    offsets,
    tokenizer=None,
    verbose=False):

    if idx_end < idx_start:
        idx_end = idx_start

    filtered_output  = ""
    #for ix in range(idx_start, idx_end + 1):
    #    filtered_output += original_tweet[offsets[ix][0]: offsets[ix][1]]
    #    if (ix+1) < len(offsets) and offsets[ix][1] < offsets[ix+1][0]:
    #        filtered_output += " "
    text1 = " " + " ".join(original_tweet.split())
    enc = tokenizer.encode(text1)
    filtered_output = ''.join(tokenizer.convert_ids_to_tokens(enc[idx_start-1:idx_end]))
    filtered_output = filtered_output.replace("▁"," ")
    #print('enc, filtered_output',enc, filtered_output)
    #filtered_output = original_tweet[idx_start:idx_end+1]
    #if sentiment_val == "neutral" or len(original_tweet.split()) < 3 or idx_end < idx_start:
    #    filtered_output = original_tweet

    if sentiment_val != "neutral" and verbose == True:
        if filtered_output.strip().lower() != target_string.strip().lower():
            print("********************************")
            print(f"Output= {filtered_output.strip()}")
            print(f"Target= {target_string.strip()}")
            print(f"Tweet= {original_tweet.strip()}")
            print("********************************")

    jac = jaccard(target_string.strip(), filtered_output.strip())
    return jac, filtered_output

class AverageMeter(object):
    """Computes and stores the average and current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), "checkpoint.pt")
        self.val_loss_min = val_loss
        
        
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
    return (stopwords_count/word_count)


def punctuations_fraction(text):
    """functiopn to claculate the fraction of punctuations over total number of characters for a given text """
    char_count = len(text)
    punctuation_count = len([c for c in text if c in string.punctuation])
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
    return (noun_count/word_count)

def fraction_adj(text):
    """function to give us fraction of adjectives over total words in given text"""
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    pos_list = nltk.pos_tag(text_splited)
    adj_count = len([w for w in pos_list if w[1] in ('JJ','JJR','JJS')])
    return (adj_count/word_count)

def fraction_verbs(text):
    """function to give us fraction of verbs over total words in given text"""
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    pos_list = nltk.pos_tag(text_splited)
    verbs_count = len([w for w in pos_list if w[1] in ('VB','VBD','VBG','VBN','VBP','VBZ')])
    return (verbs_count/word_count)

import random

class Dictogram(dict):
    def __init__(self, iterable=None):
        """Initialize this histogram as a new dict; update with given items"""
        super(Dictogram, self).__init__()
        self.types = 0  # the number of distinct item types in this histogram
        self.tokens = 0  # the total count of all item tokens in this histogram
        if iterable:
            self.update(iterable)

    def update(self, iterable):
        """Update this histogram with the items in the given iterable"""
        for item in iterable:
            if item in self:
                self[item] += 1
                self.tokens += 1
            else:
                self[item] = 1
                self.types += 1
                self.tokens += 1

    def count(self, item):
        """Return the count of the given item in this histogram, or 0"""
        if item in self:
            return self[item]
        return 0

    def return_random_word(self):
        # Another way:  Should test: random.choice(histogram.keys())
        random_key = random.sample(self, 1)
        return random_key[0]

    def return_weighted_random_word(self):
        # Step 1: Generate random number between 0 and total count - 1
        random_int = random.randint(0, self.tokens-1)
        index = 0
        list_of_keys = self.keys()
        # print 'the random index is:', random_int
        for i in range(0, self.types):
            index += self[list_of_keys[i]]
            # print index
            if(index > random_int):
                # print list_of_keys[i]
                return list_of_keys[i]

# markov chain based features, 3 words memory 
def make_higher_order_markov_model(order, data):
    markov_model = dict()

    for i in range(0, len(data)-order):
        # Create the window
        window = tuple(data[i: i+order])
        # Add to the dictionary
        if window in markov_model:
            # We have to just append to the existing Dictogram
            markov_model[window].update([data[i+order]])
        else:
            markov_model[window] = Dictogram([data[i+order]])
    return markov_model

#train_df.loc[train_df['author']=='EAP'].shape[0]
def tokenixed_list(text):
    """function to calculate the fraction of unique words on total words of the text"""
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    return (text_splited)
    
def check(train):
    jacs =[]
    for x in train.index:
        tweet=train.loc[x, 'text']
        selected_text = train.loc[x, 'selected_text']
        sentiment=train.loc[x, 'sentiment']
        tokenizer=BERTweettokenizer
        max_len=MAX_LEN
        d=process_data(tweet, selected_text, sentiment, tokenizer, max_len, training=True)
        decoded=utils.calculate_jaccard_score(
                        original_tweet=d['orig_tweet'],
                        target_string=d['orig_selected'],
                        sentiment_val=d['sentiment'],
                        idx_start=d['targets_start'],
                        idx_end=d['targets_end'],
                        offsets=d['offsets'],
                        tokenizer=BERTweettokenizer
                    )
        jacs.append(decoded[0])
    return np.mean(jacs)