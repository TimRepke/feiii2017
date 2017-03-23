from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.svm import SVC
import IPython.display as dp
from pandas import Index
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
import numpy as np
import feiii_transformers as ft

import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%I:%M:%S')

_params = {
    'cv': {
        'ngram_range': (1, 3),
        'min_df': 0.4,
        'max_df': 0.6,
        'stop_words': 'english'
    },
    'tt': {
        'use_idf': True,
        'sublinear_tf': True,
    },
    'emb': {
        'num_files': 30,
        'num_epoch': 20
    },
    'logit': {
        'loss': 'log',  # ['hinge', 'log', 'perceptron','huber'] # for pred_proba: log or modified_huber
        'penalty': 'l2',
        'shuffle': True,
        'alpha': 1e-4,
        'n_iter': 15,
        'random_state': 42,
        'class_weight': 'balanced'
    },
    'rf': {
        'n_estimators': 20,
        'criterion': 'gini',  # gini or entropy
        'max_features': 'auto',  # int, float, auto, sqrt, log2, None
        'random_state': 42,
        'class_weight': 'balanced'
    },
    'svm': {
        'C': 1.0,
        'kernel': 'sigmoid',  # linear’, ‘poly’, ‘rbf’, ‘sigmoid’
        'probability': True,
        'class_weight': 'balanced',
        'decision_function_shape': 'ovr',  # ovo, ovr
        'random_state': 42
    }
}


class FeiiiPipeline:
    def __init__(self, pipln, embedding=None, params=None):
        self.ratingmap = {'irrelevant': 0, 'neutral': 1, 'relevant': 2, 'highly': 3}
        self.params = _params if params is None else params

        line = []
        for classifier, features in pipln:
            if type(features) == list:
                features = [self._pick_features(f, embedding) for f in features]
            else:
                features = self._pick_features(features, embedding)
            line.append(self._union_skeleton(features, self._pick_classifier(classifier)))

        self.pipeline = Pipeline(self._voting_skeleton(line))

    def get(self):
        return self.pipeline

    def _pick_classifier(self, clf_name):
        if clf_name == 'logit':
            return 'clf', SGDClassifier(**self.params['logit'])
        if clf_name == 'svm':
            return 'clf', SVC(**self.params['svm'])
        return 'clf', RandomForestClassifier(**self.params['rf'])

    def _voting_skeleton(self, pipln):
        return [
            ('clf', VotingClassifier(voting='soft',
                                     # weights=[1,2,3],
                                     estimators=pipln))
        ]

    def _union_skeleton(self, features, classifier):
        return [
            ('union', FeatureUnion(
                transformer_list=features,
                # transformer_weights={'syn': 1, 'bow':1, 'emb':1}
            )),
            classifier
        ]

    def _pick_features(self, feat_name, embedding=None):
        if feat_name == 'emb':
            return [('emb', ft.Embedder(embedding))]
        if feat_name == 'bow':
            self.tfidf = TfidfTransformer(**self.params['tt'])
            return [('lem', ft.Lemmatiser()),
                    ('vect', CountVectorizer(**self.params['cv'])),
                    ('tfidf', self.tfidf)]
        return [('feats', ft.SyntaxFeatures())]

    def get_vocabulary(self):
        if self.tfidf:
            return self.tfidf.vocabulary_

    def get_inv_vocabulary(self):
        return {i: w for w, i in self.get_vocabulary().items()}

    def fit(self, frm, target):
        self.targets = list(set(frm['rating'].map(self.ratingmap)))
        self.pipeline.fit(frm, target)

    def predict(self, frm):
        num_targets = 4
        # make prediction (some options don't allow predict_proba, so simulate it!)
        try:
            pred = self.pipeline.predict_proba(frm)
        except AttributeError as e:
            print('predict_proba failed:', e)
            pred = np.zeros((len(frm), num_targets))
            pred[np.arange(len(frm)), self.pipeline.predict(frm)] = 1

        # in case a category is missing, add it
        if pred.shape[1] < num_targets:
            tmp = np.zeros((len(pred), num_targets))
            col = 0
            for mi in range(num_targets):
                if mi in self.targets:
                    tmp[:, mi:mi + 1] = pred[:, col:col + 1]
                    col += 1
            pred = tmp

        return pred.argmax(axis=1), pred

    def transform(self, frm, target):
        pred = self.pipeline.transform(frm, target)
        return pred.argmax(axis=1), pred

    def fit_transform(self, frm, target):
        pred = self.pipeline.fit_transform(frm, target)

        try:
            return pred.argmax(axis=1), pred
        except AttributeError:
            return pred

