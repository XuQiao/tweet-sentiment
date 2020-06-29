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

    #if idx_end < idx_start:
    #    idx_end = idx_start

    filtered_output  = ""
    #for ix in range(idx_start, idx_end + 1):
    #    filtered_output += original_tweet[offsets[ix][0]: offsets[ix][1]]
    #    if (ix+1) < len(offsets) and offsets[ix][1] < offsets[ix+1][0]:
    #        filtered_output += " "
    text1 = " ".join(original_tweet.split())
    enc = tokenizer.encode(text1)
    filtered_output = tokenizer.decode(enc[idx_start-1:idx_end])
    #print('enc, filtered_output',enc, filtered_output)
    #filtered_output = original_tweet[idx_start:idx_end+1]
    if sentiment_val == "neutral" or len(original_tweet.split()) < 3 or idx_end < idx_start:
        filtered_output = original_tweet

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

def tokenixed_list(text):
    """function to calculate the fraction of unique words on total words of the text"""
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    return (text_splited)