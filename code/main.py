from feiii_experiment import evaluate, kfold
from feiii_transformers import _EmbeddingHolder
from feiii_data import DataHolder
from feiii_pipeline import FeiiiPipeline
import spacy
import pandas as pd

nlp = spacy.load('en')

embedding = _EmbeddingHolder(nlp)
embedding.load('embedding')
data = DataHolder(nlp, eval_docs=2)


def pipeline():
    return FeiiiPipeline(nlp, pipln='syn', embedding=embedding)


res, macro_res, conf_matrix_role, conf_matrix_full = kfold(3, data, pipeline)
print(pd.DataFrame(res).describe())
print(pd.DataFrame(macro_res).describe())


# %matplotlib inline
# pd.DataFrame(res)[['baseline_worst','baseline_rand','ndcg_full','ndcg_role','ndcg_full_proba','ndcg_role_proba']]\
#    .boxplot(figsize=(5,8), rot=45)

# ##############
# % matplotlib inline
#
# from matplotlib import cm as colm
#
# plt.figure(figsize=(8, 5))
# sub = plt.subplot(121)
# normed = conf_matrix_full / conf_matrix_full.sum()
# plt.imshow(normed, cmap=colm.Blues, vmax=0.5)
# plt.title('conf matrix FULL')
# sub.set_yticks([0, 1, 2, 3])
# sub.set_yticklabels(['irrelevant', 'neutral', 'relevant', 'highly'])
# sub.set_xticks([0, 1, 2, 3])
# sub.set_xticklabels(['irrelevant', 'neutral', 'relevant', 'highly'])
#
# for i in range(normed.shape[0]):
#     for j in range(normed.shape[1]):
#         v = normed[i][j]
#         c = '%.2f' % v if v > 0.005 else ''
#         sub.text(i, j, c, va='center', ha='center')
#
# sub = plt.subplot(122)
# normed = conf_matrix_role / conf_matrix_role.sum()
# plt.imshow(normed, cmap=colm.Blues, vmax=0.5)
# sub.yaxis.tick_right()
# sub.set_yticks([0, 1, 2, 3])
# sub.set_yticklabels(['irrelevant', 'neutral', 'relevant', 'highly'])
# sub.set_xticks([0, 1, 2, 3])
# sub.set_xticklabels(['irrelevant', 'neutral', 'relevant', 'highly'])
#
# for i in range(normed.shape[0]):
#     for j in range(normed.shape[1]):
#         v = normed[i][j]
#         c = '%.2f' % v if v > 0.01 else ''
#         sub.text(i, j, c, va='center', ha='center')
#
# plt.title('conf matrix ROLE')
#
# plt.tight_layout()
# plt.show()