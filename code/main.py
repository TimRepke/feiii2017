from feiii_experiment import evaluate
from feiii_transformers import _EmbeddingHolder
from feiii_data import DataHolder
from feiii_pipeline import FeiiiPipeline


embedding = _EmbeddingHolder()
embedding.load('embedding')
data = DataHolder()


def pipeline():
    return FeiiiPipeline([('logit', 'bow'), ('logit', 'emb'), ('rf', 'syn')], embedding=embedding)


evaluate(3, data, pipeline)
