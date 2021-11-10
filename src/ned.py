import pandas as pd
import torch.nn.functional as F

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from flair.data import Sentence

from src.utils import GoogleSearch, WikiDataSearch


class NED:
    def __init__(self, cand_source='wikidata', num_cands=5):
        self.stop_words = set(stopwords.words('english'))
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.cand_source = cand_source
        self.num_cands = num_cands

    def filter(self, s):
        word_tokens = self.tokenizer.tokenize(s)
        return [w for w in word_tokens if not w.lower() in self.stop_words]

    def generateCandidates(self, mention):
        if self.cand_source == 'wikidata':
            return WikiDataSearch(mention, self.num_cands)
        elif self.cand_source == 'google':
            return GoogleSearch(mention, self.num_cands)
        else:
            raise ValueError("Not valid candidate generation source")


class BasicFlairNED(NED):
    def __init__(self, emb, cand_source='wikidata', num_cands=5):
        super().__init__(cand_source, num_cands)
        self.emb = emb

    def encodeSentence(self, s, filter=True):
        s = Sentence(self.filter(s)) if filter else Sentence(s)
        self.emb.embed(s)
        return s.get_embedding()

    def link(self, mention, context, candidates=None):
        # 1. Generate candidate entities
        if candidates is None:
            candidates = [(i['label'], i['description'])
                          for i in self.generateCandidates(mention)]

        # 2. Encode context-mention and candidates
        context = self.encodeSentence(context)
        candidateEnc = {i: self.encodeSentence(j) for i, j in candidates}

        # 3. Candidate Ranking
        ranking = []
        for candidate in candidates:
            score = F.cosine_similarity(
                context, candidateEnc[candidate[0]], dim=0)
            ranking.append([candidate[0], candidate[1], score.item()])

        ranking.sort(key=lambda x: x[2], reverse=True)
        return pd.DataFrame(ranking, columns=['Entity', 'Description', 'Confidence'])
