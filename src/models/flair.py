import torch.nn.functional as F
from flair.data import Sentence
from src.models.base import Base


class BaseFlair(Base):
    def __init__(self, emb, cased=True, filter_stopwords=True, cand_source='wiki', num_cands=5):
        super().__init__(emb, cased, filter_stopwords, cand_source, num_cands)

    def encodeSentence(self, s):
        s = s if self.cased else s.lower()
        s = Sentence(self.filter(s) if self.filter_stopwords else s)
        self.emb.embed(s)
        return s.get_embedding()

    def rank(self, candidates, context):
        ranking = []
        for candidate in candidates.keys():
            score = F.cosine_similarity(context, candidates[candidate], dim=0)
            ranking.append([candidate, score.item()])
        return ranking
