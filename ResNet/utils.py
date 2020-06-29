import numpy as np
from collections import Counter
import torch

def data_aug(train_sentiment, multiple = 1):
    def aug(row):
        tweet = row['text']
        selected_text = row['selected_text']
        idx = tweet.find(selected_text) 
        if idx>=0:
            new_tweets = []
            prev = tweet[:idx].split()
            after = tweet[idx+len(selected_text):].split()
            pool = [(i,j) for i in range(len(prev)+1) for j in range(len(after)+1)]
            pool.remove((len(prev),len(after)))
            if len(pool) == 0:
                return None
            for r in np.random.choice(len(pool), multiple):
                r1, r2 = pool[r]
                start = ''
                end = ''
                if r1 > 0:
                    start =' '.join(prev[r1:]) + ' '
                if r2 > 0:
                    end = ' '+' '.join(after[:r2])
                
                new_tweets.append(start+selected_text+end)
            if len(new_tweets) > 0:
                return new_tweets
            return None
        else:
            return None
        
    train_aug = {'text':[],'selected_text':[], 'sentiment':[], 'textID':[]}
    for row in train_sentiment.iterrows():
        row = row[1]
        new_tweets = aug(row)
        if new_tweets:
            for new_tweet in new_tweets:
                train_aug['text'].append(new_tweet)
                train_aug['selected_text'].append(row['selected_text'])
                train_aug['sentiment'].append(row['sentiment'])
                train_aug['textID'].append(row['textID'])
    train_aug = pd.DataFrame(train_aug).dropna()
    return train_aug

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
    verbose=False):

    if idx_end < idx_start:
        idx_end = idx_start

    filtered_output  = ""
    for ix in range(idx_start, idx_end + 1):
        filtered_output += original_tweet[offsets[ix][0]: offsets[ix][1]]
        if (ix+1) < len(offsets) and offsets[ix][1] < offsets[ix+1][0]:
            filtered_output += " "

    if sentiment_val == "neutral" or len(original_tweet.split()) < 2:
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
