from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import IPython.display as dp
from pandas import Index
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
import numpy as np
import pandas as pd
import logging

# forward mapping
# rating_map = {'Irrelevant':0.0, 'Neutral': 0.4, 'Relevant':0.75, 'Highly relevant': 1.0}
rating_map = {'Irrelevant': 0, 'Neutral': 0, 'Relevant': 1, 'Highly relevant': 2}
rating_map2 = {'irrelevant': 0, 'neutral': 0, 'relevant': 1, 'highly': 2}

# normally (wikipedia) scale 0,1,2,3
rating_map = {'Irrelevant': 0, 'Neutral': 1, 'Relevant': 2, 'Highly relevant': 3}
rating_map2 = {'irrelevant': 0, 'neutral': 1, 'relevant': 2, 'highly': 3}


# text says "The pRL gets one point of gain for each relevant triple and two points for each highly relevant triple"

# backward mapping
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


def dcg_np(ol, p):
    ol = ol[:p]
    try:
        return ol[0] + np.sum(ol[1:] / np.log2(np.arange(2, ol.size + 1)))
    except:
        return np.nan


def ndcg_np(ol, ideal, p):
    try:
        return dcg_np(ol, p) / dcg_np(ideal, p)
    except:
        return np.nan


def ndcg(frame, p=None):
    if not p:
        p = len(frame)

    scored = np.array(frame.sort_values('score', ascending=False)['rating_avg'])
    ideal = np.array(frame.sort_values('rating_avg', ascending=False)['rating_avg'])

    return ndcg_np(scored, ideal, p)


def ndcg2(frame, scoring, p=None):
    if not p:
        p = len(frame)

    frmap = frame['rating'].map(rating_map2)
    s = np.array([sc[1] for sc in scoring])

    scored = np.array([frmap.loc[scoring[i][0]] for i in (-s).argsort()])
    ideal = np.array(frmap.sort_values(ascending=False))

    return ndcg_np(scored, ideal, p)


def evaluate(n_folds, data, pipeline_generator):
    conf_matrix_role = np.zeros((4, 4))
    conf_matrix_full = np.zeros((4, 4))

    # holer for stats
    res = {
        'baseline_rand': [],
        'baseline_worst': [],
        'ndcg_role': [],
        'ndcg_full': [],
        'ndcg_role_proba': [],
        'ndcg_full_proba': [],
        'acc_role': [],
        'acc_full': [],
        'f1_role': [],
        'f1_full': []
    }
    macro_res = {
        'ndcg_role': [],
        'ndcg_full': [],
        'ndcg_role_proba': [],
        'ndcg_full_proba': [],
    }

    n_leaveout_docs = min(1, int(data.files / n_folds))
    print('Leaving {} docs out per fold'.format(n_leaveout_docs))

    for crosseval in range(n_folds):

        print("\n\n==========================================================================")
        print("===                      CROSSEVAL ITERATION " + str(crosseval + 1) + "/" + str(
            n_folds) + "                     =====")
        print("==========================================================================\n\n")

        scores_role = []
        scores_full = []
        scores_role_proba = []
        scores_full_proba = []

        data.set_train_eval(data.files[(n_leaveout_docs * crosseval):(n_leaveout_docs * crosseval) + n_leaveout_docs])

        models = {role: pipeline_generator() for role in data.get_roles()}

        fullmodel = pipeline_generator()
        fullmodel.fit(data.train, data.get_target(frm='train', group=None))

        # run evaluation for each role
        for role, model in models.items():
            print('=== ' + role.upper() + ' ======')
            grp = None if role == 'full' else role
            bl1, bl2 = data.establish_baseline(grp=grp, include_eval=True,
                                               include_test=False, include_train=False)
            res['baseline_rand'].append(bl2)
            res['baseline_worst'].append(bl1)

            model.fit(data.get_frame(frm='train', group=grp), data.get_target(frm='train', group=grp))

            eval_target = data.get_target(frm='eval', group=grp)
            eval_data = data.get_frame(frm='eval', group=grp)

            # get predictions from model trained on role
            pred_role, pred_proba_role = model.predict(eval_data)

            # get predictions from model trained on all
            pred_full, pred_proba_full = fullmodel.predict(eval_data)

            score_role = np.sum(pred_proba_role * np.array([1, 2, 4, 5]), axis=1)
            score_full = np.sum(pred_proba_full * np.array([1, 2, 4, 5]), axis=1)

            # calculate accuracy
            res['acc_role'].append(np.mean(pred_role == eval_target))
            print('Accuracy | role :', np.mean(pred_role == eval_target))
            res['acc_full'].append(np.mean(pred_full == eval_target))
            print('Accuracy | full :', np.mean(pred_full == eval_target))

            _, _, f1, supp = precision_recall_fscore_support(eval_target, pred_full, labels=[0, 1, 2, 3])
            res['f1_full'].append((f1 * supp).sum() / supp.sum())
            _, _, f1, supp = precision_recall_fscore_support(eval_target, pred_role, labels=[0, 1, 2, 3])
            res['f1_role'].append((f1 * supp).sum() / supp.sum())

            # print classification report
            print(metrics.classification_report(eval_target, pred_full,
                                                labels=[0, 1, 2, 3],
                                                target_names=list(rating_map2.keys())))

            # print confusion matrix
            conf_matrix_role += np.array(metrics.confusion_matrix(eval_target, pred_role, labels=[0, 1, 2, 3]))
            cm = metrics.confusion_matrix(eval_target, pred_full, labels=[0, 1, 2, 3])
            conf_matrix_full += np.array(cm)
            print(cm)
            # print(metrics.confusion_matrix(testtarget, pred_full, labels=[0,1,2,3]))

            # add scores to full list
            score_role_tmp = list(zip(list(eval_data.index), pred_role))
            scores_role += score_role_tmp
            score_full_tmp = list(zip(list(eval_data.index), pred_full))
            scores_full += score_full_tmp

            # add scores based on probability to full list
            score_role_proba_tmp = list(zip(list(eval_data.index), score_role))
            scores_role_proba += score_role_proba_tmp
            score_full_proba_tmp = list(zip(list(eval_data.index), score_full))
            scores_full_proba += score_full_proba_tmp

            # add NDCG to results
            res['ndcg_role'].append(ndcg2(eval_data, score_role_tmp))
            res['ndcg_role_proba'].append(ndcg2(eval_data, score_role_proba_tmp))
            res['ndcg_full'].append(ndcg2(eval_data, score_full_tmp))
            res['ndcg_full_proba'].append(ndcg2(eval_data, score_full_proba_tmp))

            # echo results
            print('> NDCG Score | role | categ  | {:.5f}'.format(ndcg2(eval_data, score_role_tmp)))
            print('> NDCG Score | role | proba* | {:.5f}'.format(ndcg2(eval_data, score_role_proba_tmp)))
            print('> NDCG Score | full | categ  | {:.5f}'.format(ndcg2(eval_data, score_full_tmp)))
            print('> NDCG Score | full | proba* | {:.5f}'.format(ndcg2(eval_data, score_full_proba_tmp)))

        # echo results of NDCG for entire set
        tmp = ndcg2(data.get_frame(frm='train', group=None).loc[[k for k, v in scores_role]], scores_role)
        macro_res['ndcg_role'].append(tmp)
        print('TOTAL NDCG | role | categ  | {:.5f}'.format(tmp))

        tmp = ndcg2(data.get_frame(frm='train', group=None).loc[[k for k, v in scores_role_proba]], scores_role_proba)
        macro_res['ndcg_role_proba'].append(tmp)
        print('TOTAL NDCG | role | proba* | {:.5f}'.format(tmp))

        tmp = ndcg2(data.get_frame(frm='train', group=None).loc[[k for k, v in scores_full]], scores_full)
        macro_res['ndcg_full'].append(tmp)
        print('TOTAL NDCG | full | categ  | {:.5f}'.format(tmp))

        tmp = ndcg2(data.get_frame(frm='train', group=None).loc[[k for k, v in scores_full_proba]], scores_full_proba)
        macro_res['ndcg_full_proba'].append(tmp)
        print('TOTAL NDCG | full | proba* | {:.5f}'.format(tmp))

        res['baseline_rand'] = np.mean(res['baseline_rand'], axis=1)
