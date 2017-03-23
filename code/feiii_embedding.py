from gensim.models import Doc2Vec
from bs4 import BeautifulSoup
from gensim.models.doc2vec import LabeledSentence, TaggedDocument
import os
import re
import logging
import spacy
nlp = spacy.load('en')

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%I:%M:%S')


class EmbeddingHolder:


    instance = None

    def __init__(self, **kwargs):
        if not EmbeddingHolder.instance:
            EmbeddingHolder.instance = EmbeddingHolder.__EmbeddingHolder(**kwargs)
        #else:
        #    EmbeddingHolder.instance.val = arg

    def __getattr__(self, name):
        return getattr(self.instance, name)



