from sklearn.base import BaseEstimator, TransformerMixin
from multiprocessing import Pool
import numpy as np
import pandas as pd
import re
from gensim.models import Doc2Vec
from bs4 import BeautifulSoup
from gensim.models.doc2vec import TaggedDocument
import os
from collections import Counter

from code.feiii_nlp import get_nlp_model

nlp = get_nlp_model()


class Lemmatiser(BaseEstimator, TransformerMixin):
    def __init__(self, textcol='THREE_SENTENCES'):
        self.textcol = textcol

    def fit(self, x, y=None):
        return self

    def transform(self, frm: pd.DataFrame):
        if 'clean' in frm.columns:
            return list(frm['clean'])

        texts = []
        for doc, row in frm.iterrows():
            tmp = []
            text = row[self.textcol]
            for token in nlp(text):
                if not (token.like_num or
                            token.is_stop or
                            token.is_space or
                            token.is_digit or
                            token.is_punct or
                                len(token.orth_) < 3):
                    tmp.append(token.lemma_)
            texts.append(' '.join(tmp))
        return texts


class _EmbeddingHolder:
    def __init__(self, directory='/home/tim/Uni/HPI/workspace/FEII/full_reports/'):
        self.directory = directory
        self.embedding = None
        self._is_trained = False
        self.only_wordchars = True

    def save(self, filename):
        self.embedding.save(filename)

    def load(self, filename):
        self.embedding = Doc2Vec.load(filename)
        self._is_trained = True

    @property
    def is_trained(self):
        return self._is_trained

    def _read_files(self, num_files):
        """
        Internal function to read all reports inside the directory into a single string.
        Files are assumed to be provided as HTML. This function strips all that away
        and returns plain text.

        :param num_files: number of files to read, None if all available
        :return: string containing text from all files
        """
        cleantext = ''
        for cnt, file in enumerate(os.listdir(self.directory)):
            if file.endswith('.html'):
                print('reading: ' + file)
                with open(self.directory + file, "r", encoding='utf-8', errors='ignore') as f:
                    cleantext += BeautifulSoup(f.read(), "html5lib").text

                if cnt > num_files:
                    break
        return cleantext

    def _clean_string(self, s):
        if self.only_wordchars:
            return re.sub(r'[^a-z ]', '', str(s).lower(), flags=re.IGNORECASE)
        return str(s).lower()

    def _prepare_string(self, text):
        print('extracting sentences...')
        sents = [self._clean_string(s) for s in nlp(text).sents]

        print('words:', len(text.split()))
        print('sentences:', len(sents))

        return [TaggedDocument(words=s.split(), tags=['SENT_' + str(i)]) for i, s in enumerate(sents)]

    def train(self, num_files=3, num_epochs=10, only_wordchars=True):
        self.only_wordchars = only_wordchars
        labsents = self._prepare_string(self._read_files(num_files))

        self.embedding = Doc2Vec(size=40, window=10, min_count=5,
                                 workers=6, alpha=0.025, min_alpha=0.025,
                                 batch_words=100, dm=0)
        self.embedding.build_vocab(labsents)
        self.train_cont(labsents, num_epochs)

        self._is_trained = True

    def train_cont(self, labsents, num_epochs):
        for epoch in range(num_epochs):
            if self.embedding.alpha < 0:
                print('Sub-zero learning rate. stopping!')
                break
            print('Training-Epoch:', epoch, '| lr:', self.embedding.alpha)
            self.embedding.train(labsents)
            self.embedding.alpha -= 0.002  # decrease the learning rate
            self.embedding.min_alpha = self.embedding.alpha  # fix the learning rate, no decay

    def infer(self, raw_sentence):
        if not self.is_trained:
            raise AttributeError('Embedding not trained yet.')
        return self.embedding.infer_vector(self._clean_string(raw_sentence).split())


class Embedder(BaseEstimator, TransformerMixin):
    def __init__(self, embedding, sentence_col='THREE_SENTENCES'):
        if embedding is None:
            raise ValueError('Expected embedding, but got None!')
        self.embedding = embedding
        self.col = sentence_col
        self.esize = self.embedding.embedding.vector_size
        self.num_sents = 3

    def fit(self, x, y=None):
        return self

    def transform(self, frm):
        X = []
        for i, row in frm.iterrows():
            vecs = [self.embedding.infer(se) for s in nlp(row['THREE_SENTENCES']).sents for se in s][
                   :self.num_sents]
            while len(vecs) < self.num_sents:
                vecs.append(list(np.zeros((self.esize,))))
            X.append(np.array(vecs).reshape((self.num_sents * self.esize,)))

        return X


class SyntaxFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, asfrm=False, n_workers=6, features=None):
        self.asfrm = asfrm
        self.n_workers = n_workers
        self.features_ = None
        self.feats = features

    def fit(self, X, y=None):
        return self

    def transform(self, frm):
        p = Pool(self.n_workers)
        Xd = pd.DataFrame(p.map(self._features, frm.iterrows()), index=frm.index)
        p.close()

        self.features_ = list(Xd.columns)

        if self.feats is not None:
            Xd = Xd[self.feats]

        if self.asfrm:
            return Xd

        return Xd.as_matrix()

    def _features(self, row):
        _, row = row
        raw = row['THREE_SENTENCES']
        n = nlp(raw)
        tagc = Counter([nn.tag_ for nn in n])
        posc = Counter([nn.pos_ for nn in n])
        clean = row['clean']
        counts_r = Counter(raw.split())
        counts_c = Counter(clean.split())
        try:
            dist_role_ent = raw.lower().index(row['MENTIONED_FINANCIAL_ENTITY'].lower()) - raw.lower().index(row['ROLE'].lower())
        except:
            dist_role_ent = 0
        ret = {
            'num_chars': len(raw),
            'num_words': len(raw.split()),
            'num_upper_chars': sum(1 for c in raw if c.isupper()),
            'num_upper_words': sum(1 for w in raw.split() if w[0].isupper()),
            'ratio_upper_chars': sum(1 for c in raw if c.isupper()) / len(raw),
            'ratio_upper_words': sum(1 for w in raw.split() if w[0].isupper()) / max(1, len(raw.split())),
            'mean_word_len': np.mean([len(w) for w in raw.split()]),
            'num_word_repetitions_raw': len({k: v for k, v in counts_r.items() if v > 1}),
            'num_word_repetitions_clean': len({k: v for k, v in counts_c.items() if v > 1}),
            'ratio_word_repetitions_raw': len({k: v for k, v in counts_r.items() if v > 1}) / max(1, len(raw.split())),
            'ratio_word_repetitions_clean': len({k: v for k, v in counts_c.items() if v > 1}) / max(1, len(clean.split())),
            'num_dollarsigns': len(raw) - len(raw.replace('$', '')),
            'num_numbers': len(re.findall(r'\d+', raw)),
            'num_digits': len(re.findall(r'\d', raw)),
            'ratio_numbers': len(re.findall(r'\d+', raw)) / max(1, len(raw.split())),
            'ratio_digits': len(re.findall(r'\d', raw)) / len(raw),
            'num_ents_org': len([ent for ent in n.ents if ent.root.ent_type_ == 'ORG']),
            'num_ents_person': len([ent for ent in n.ents if ent.root.ent_type_ == 'PERSON']),
            'num_verbs': tagc['VB'] + tagc['VBD'] + tagc['VBG'] + tagc['VBN'] + tagc['VBP'] + tagc['VBZ'],
            'num_adverbs': tagc['RB'] + tagc['RBR'] + tagc['RBS'],
            'num_adj': tagc['JJ'] + tagc['JJR'] + tagc['JJS'],
            'num_punct': posc['PUNCT']+posc['SYM'],
            'num_1letter': len([tok for tok in n if len(tok) == 1 and tok.pos_ != 'PUNCT' and tok.pos_ != 'SYM']),
            'num_role_mentions': raw.lower().count(row['ROLE']),
            'role+ent': len([sent for sent in n.sents if row['MENTIONED_FINANCIAL_ENTITY'] in str(sent) and row['ROLE'] in str(sent)]),
            'dist_role_ent': dist_role_ent + (len(row['MENTIONED_FINANCIAL_ENTITY']) if dist_role_ent < 0 else len(row['ROLE'])),
            'dist_role_ent_abs': abs(dist_role_ent + (len(row['MENTIONED_FINANCIAL_ENTITY']) if dist_role_ent < 0 else len(row['ROLE'])))
        }
        return ret
