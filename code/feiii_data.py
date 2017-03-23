import matplotlib.pyplot as plt
import os
import pandas as pd
import re
import numpy as np
import spacy

from feiii_experiment import ndcg2

nlp = spacy.load('en')


def avg2rating(rating_avg):
    try:
        if rating_avg < 1.0:
            return 'irrelevant'
        if rating_avg <= 1.7:
            return 'neutral'
        if rating_avg <= 2.5:
            return 'relevant'
        return 'highly'
    except:
        return rating_avg


class DataHolder:
    def __init__(self,
                 traindir='/home/tim/Uni/HPI/workspace/FEII/Training/',
                 testfile='/home/tim/Uni/HPI/workspace/FEII/FEIII2_Testing.csv',
                 eval_docs=2):

        self.ratingmap = {'irrelevant': 0, 'neutral': 1, 'relevant': 2, 'highly': 3}

        self.train_full = self._load_trainframe(traindir)
        self.train_full = self._prepare_frame(self.train_full)
        self.files = list(set(self.train_full['SOURCE']))

        eval_samples = np.random.choice(self.files, eval_docs)
        self.set_train_eval(eval_samples)

        self.test = self._prepare_frame(pd.read_csv(testfile, index_col=None))

    def get_roles(self):
        return list(set(self.train['grp']))

    def set_train_eval(self, eval_samples):
        self.train = self.train_full[~self.train_full['SOURCE'].isin(eval_samples)]
        self.eval = self.train_full[self.train_full['SOURCE'].isin(eval_samples)]

    def _load_trainframe(self, path):
        list_ = []
        for file in os.listdir(path):
            if file.endswith('.csv'):
                df = pd.read_csv(path + file, index_col=None)
                df['FILE'] = [file] * len(df)
                print('reading file', file, 'with', len(df), 'entries.')
                list_.append(df)

        frm = pd.concat(list_)
        frm.reset_index(drop=True, inplace=True)

        return frm

    def _prepare_frame(self, frm):
        frm['grp'] = frm['ROLE'].map(lambda x: re.sub("(ies|y|s)$", "", x)).str.lower()

        frm['SOURCE'] = [r['FILER_CIK'] + '-' + r['FILING_INTERVAL'] for r in frm.iterrows()]

        frm = self._lemmatise(frm)
        frm = self._adjust_ratings(frm)

        return frm

    def _lemmatise(self, frm):
        texts = []
        for doc, row in frm.iterrows():
            tmp = []
            text = row['THREE_SENTENCES']
            for token in nlp(text):
                if not (token.like_num or
                            token.is_stop or
                            token.is_space or
                            token.is_digit or
                            token.is_punct or
                                len(token.orth_) < 3):
                    tmp.append(token.lemma_)
            texts.append(' '.join(tmp))
        frm['clean'] = texts
        return frm

    def _adjust_ratings(self, frm):
        cnts = []
        avgs = []
        rats = []
        mins = []
        maxs = []
        for n, row in frm.replace(to_replace=self.ratingmap).filter(regex=("RATING")).iterrows():
            cnt = 0
            s = -1
            for k, v in row.to_dict().items():
                if v >= 0:
                    cnt += 1
                    s = v if s < 0 else s + v
            cnts.append(cnt)
            avgs.append(s / cnt if cnt > 0 else -1)
            rats.append(avg2rating(s / cnt if cnt > 0 else -1))
            mins.append(row.min())
            maxs.append(row.max())

        frm['num_experts'] = cnts
        frm['rating_avg'] = avgs
        frm['rating'] = rats
        frm['rating_min'] = mins
        frm['rating_max'] = maxs
        return frm

    def _get_rating_agg(self, frm):
        # create pivot table
        tmp = frm.pivot_table(values='rating_avg',
                              columns=['rating'],
                              index=['FILE'],
                              aggfunc=lambda x: len(x)).fillna(0)
        cols = ['irrelevant', 'neutral', 'relevant', 'highly']

        # fill missing columns
        for c in set(cols) - set(tmp.columns):
            tmp[c] = [0.0] * len(tmp)

        return tmp[cols].as_matrix().T

    def get_target(self, frm='train', group=None):
        frm = self.train if frm == 'train' else self.test if frm == 'test' else self.eval
        if group is not None:
            frm = frm[frm['grp'] == group]

        return np.array(frm['rating'].map(self.ratingmap))

    def get_frame(self, frm='train', group=None):
        frm = self.train if frm == 'train' else self.test if frm == 'test' else self.eval
        if group is not None:
            frm = frm[frm['grp'] == group]

        return frm

    def draw_rating_distribution(self, relative=True, include_total=True, figsize=(15, 8)):
        files = self.files
        rating_agg = self._get_rating_agg(self.train)
        if include_total:
            files = ['TOTAL'] + files
            rating_agg = np.column_stack((rating_agg.sum(axis=1), rating_agg))

        if relative:
            rating_agg = rating_agg / rating_agg.sum(axis=0)

        plt.figure(figsize=figsize)

        p0 = plt.bar(np.arange(len(files)), rating_agg[0])
        p1 = plt.bar(np.arange(len(files)), rating_agg[1], bottom=rating_agg[0])
        p2 = plt.bar(np.arange(len(files)), rating_agg[2], bottom=rating_agg[0] + rating_agg[1])
        p3 = plt.bar(np.arange(len(files)), rating_agg[3], bottom=rating_agg[0] + rating_agg[1] + rating_agg[2])

        plt.ylabel('Count')
        plt.xticks(np.arange(len(files)), files, rotation=90)
        plt.legend((p0[0], p1[0], p2[0], p3[0]), ('irrelevant', 'neutral', 'relevant', 'highly'))
        plt.show()

    def short_setinfo(self):
        print('Items in training set:', len(self.train),
              '({:.2f}%)'.format(len(self.train) / (len(self.train) + len(self.eval)) * 100))
        print('Items in eval set:', len(self.eval))
        print('Items in test set:', len(self.test))
        print(' =', len(self.train) + len(self.eval) + len(self.test))

        a = len(set(self.train['SOURCE']))
        b = len(set(self.eval['SOURCE']))
        c = len(set(self.test['SOURCE']))
        print('Number of source documents:', a + b + c, 'total,', a, 'train,', b, 'eval', c, 'test')

        rating_agg = self._get_rating_agg(self.train)
        print('Absolute (training): IR {:.2f}, N {:.2f}, R {:.2f}, HR {:.2f}'.format(*rating_agg.sum(axis=1)))
        print('Relative (training): IR {:.2f}, N {:.2f}, R {:.2f}, HR {:.2f}'.format(*rating_agg.sum(axis=1) /
                                                                                      rating_agg.sum()))

        for grp in set(self.train['grp']):
            print('Role samples for {} in train: {}, eval: {}, test: {}'.format(
                grp.upper(),
                len(self.train[self.train['grp'] == grp]),
                len(self.eval[self.eval['grp'] == grp]),
                len(self.test[self.test['grp'] == grp])))

            # for e in ['RATING_EXPERT_1', 'RATING_EXPERT_2', 'RATING_EXPERT_3', 'RATING_EXPERT_4', 'RATING_EXPERT_5',
            #           'RATING_EXPERT_6', 'RATING_EXPERT_7', 'RATING_EXPERT_1.1', 'RATING_EXPERT_9', 'RATING_EXPERT_10']:
            #     tr = self.trainfrm[self.trainfrm[e].notnull()]
            #     te = self.testfrm[self.testfrm[e].notnull()]
            #     if (len(tr) + len(te)) > 0:
            #         print("{} gave {} ratings, in train: {}, in test: {}.".format(e,
            #                                                                       len(tr) + len(te),
            #                                                                       len(tr),
            #                                                                       len(te)))

    def establish_baseline(self, grp=None, include_train=False, include_eval=True, include_test=False, num_random=100):
        frames = []
        if include_train:
            frames.append(self.train)
        if include_eval:
            frames.append(self.eval)
        if include_test:
            frames.append(self.test)
        tmp = pd.concat(frames)

        if grp is not None:
            tmp = tmp[tmp['grp'] == grp]

        ndcgtest = []
        for k in range(num_random):
            ndcgtest.append(ndcg2(tmp, [(tmp.index[k], v) for k, v in enumerate(np.random.rand(len(tmp)))]))

        print('NDCG after ' + str(num_random) + 'x random order:')
        print(' > mean ndcg =', np.mean(ndcgtest), '| std =', np.std(ndcgtest))

        print('NDCG for worst case (inverted best) order:')
        worst = ndcg2(tmp, [(i, abs(self.ratingmap[r['rating']] - 3))
                            for i, r in tmp.iterrows()])
        print(' > ndcg =', worst)

        return worst, ndcgtest
