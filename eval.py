import numpy as np
import functools

from sklearn import svm
from abc import ABC, abstractmethod
from sklearn.metrics import f1_score, accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression   # used for Computers and Photo
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder
import torch
from tqdm import tqdm
from torch import nn
from torch.optim import Adam
import random
import os
from time import perf_counter as t

seed = random.randint(1,999999)
# seed = 847609
print("downstream seed: ", seed)
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

def repeat(n_times):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            results = [f(*args, **kwargs) for _ in range(n_times)]
            statistics = {}
            for key in results[0].keys():
                values = [r[key] for r in results]
                statistics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values)}
            print_statistics(statistics, f.__name__)
            return statistics
        return wrapper
    return decorator


def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, np.bool)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret


def print_statistics(statistics, function_name):
    print(f'(E) | {function_name}:', end=' ')
    for i, key in enumerate(statistics.keys()):
        mean = statistics[key]['mean']
        std = statistics[key]['std']
        print(f'{key}={mean:.4f}+-{std:.4f}', end='')
        if i != len(statistics.keys()) - 1:
            print(',', end=' ')
        else:
            print()


@repeat(3)
def label_classification(embeddings, y, ratio):
    start = t()

    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(np.bool)
    X = normalize(X, norm='l2')

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1 - ratio)
    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)
    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)
    y_pred = prob_to_one_hot(y_pred)
    micro = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)

    now = t()
    return {
        'F1Mi': micro,
        'F1Ma': macro,
        'ACC': acc,
        'time': now-start
    }

class LogisticRegression_bgrl(nn.Module):
    def __init__(self, num_features, num_classes):
        super(LogisticRegression_bgrl, self).__init__()
        self.fc = nn.Linear(num_features, num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x):
        z = self.fc(x)
        return z

class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict, config: dict) -> dict:
        pass

    def __call__(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict, config: dict) -> dict:
        for key in ['train', 'test', 'valid']:
            assert key in split

        result = self.evaluate(x, y, split, config)
        return result

class LREvaluator(BaseEvaluator):
    def __init__(self, num_epochs: int = 1000, learning_rate: float = 0.1,
                 weight_decay: float = 0.0, test_interval: int = 20):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.test_interval = test_interval

    @repeat(3)
    def evaluate(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict, seed: int):
        device = x.device
        x = x.detach().to(device)
        input_dim = x.size()[1]
        y = y.to(device)
        num_classes = y.max().item() + 1
        classifier = LogisticRegression_bgrl(input_dim, num_classes).to(device)
        optimizer = Adam(classifier.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        output_fn = nn.LogSoftmax(dim=-1)
        criterion = nn.NLLLoss()

        best_val_micro = 0
        best_val_acc = 0
        best_test_micro = 0
        best_test_macro = 1
        best_test_acc = 0
        best_epoch = 0

        start = t()

        with tqdm(total=self.num_epochs, desc='(LR)', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]') as pbar:
        
            for epoch in range(self.num_epochs):
                classifier.train()
                optimizer.zero_grad()

                output = classifier(x[split['train']])
                loss = criterion(output_fn(output), torch.squeeze(y[split['train']]))

                loss.backward()
                optimizer.step()

                if (epoch + 1) % self.test_interval == 0:
                    classifier.eval()
                    y_test = y[split['test']].detach().cpu().numpy()
                    y_pred = classifier(x[split['test']]).argmax(-1).detach().cpu().numpy()
                    test_micro = f1_score(y_test, y_pred, average='micro')
                    test_macro = f1_score(y_test, y_pred, average='macro')
                    test_acc = accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)

                    y_val = y[split['valid']].detach().cpu().numpy()
                    y_pred = classifier(x[split['valid']]).argmax(-1).detach().cpu().numpy()

                    val_acc = accuracy_score(y_val, y_pred, normalize=True, sample_weight=None)
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_test_micro = test_micro
                        best_test_macro = test_macro
                        best_test_acc = test_acc
                        best_epoch = epoch

                    now = t()
                    pbar.set_postfix({'best test F1Mi': best_test_micro, 'F1Ma': best_test_macro, 'ACC': best_test_acc, 'time': now - start})
                    pbar.update(self.test_interval)
            
        return {
            'micro_f1': best_test_micro,
            'macro_f1': best_val_micro,
            'ACC': best_test_acc,
            'time': now - start,
        }
