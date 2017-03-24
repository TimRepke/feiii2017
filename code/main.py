from code.feiii_experiment import kfold
from code.feiii_transformers import _EmbeddingHolder
from code.feiii_data import DataHolder
from code.feiii_pipeline import FeiiiPipeline

import pandas as pd

embedding = _EmbeddingHolder()
embedding.load('embedding')
data = DataHolder(eval_docs=2)


def pipeline():
    return FeiiiPipeline(pipln='syn', embedding=embedding)


res, macro_res, conf_matrix_role, conf_matrix_full = kfold(3, data, pipeline)
print(pd.DataFrame(res).describe())
print(pd.DataFrame(macro_res).describe())
