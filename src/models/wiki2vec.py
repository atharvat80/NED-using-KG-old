import numpy as np
from src.models.base import Base


class BaseWiki2Vec(Base):
	def __init__(self, emb, cased=True, filter_stopwords=True, cand_source='wiki', num_cands=5):
		super().__init__(emb, cased, filter_stopwords, cand_source, num_cands)
		self.vector_size = 100

	def encodeSentence(self, s):
		s = s if self.cased else s.lower()
		words = self.filter(s) if self.filter_stopwords else self.tokenizer.tokenize(s)
		enc = []
		for w in words:
			if self.emb.get_word(w) is not None:
				enc.append(self.emb.get_word_vector(w))
			elif self.emb.get_word(w.lower()) is not None:
				enc.append(self.emb.get_word_vector(w.lower()))
		try:
			return sum(enc)/len(enc)
		except:
			return np.zeros((self.vector_size,))

	def encodeEntity(self, entity):
		if self.emb.get_entity(entity) is not None:
			return self.emb.get_entity_vector(entity)
		else:
			return None

	def link(self, mention, context, candidates=None, top_only=True):
		# 1. Candidate Generation
		candidates = self.generateCandidates(mention) if candidates is None else candidates
		candidates = [i.replace('_', ' ') for i, _ in candidates]

		# 2. Encode context-mention and candidates
		context = self.encodeSentence(context)
		candidateEnc = {i: self.encodeEntity(i) for i in candidates}

		# 3. Candidate Ranking
		ranking = self.rank(candidateEnc, context)
		ranking.sort(key=lambda x: x[1], reverse=True)
		if ranking != []:
			ranking = [(i[0].replace(' ', '_'), i[1]) for i in ranking]
		if top_only:
			return ranking[0] if len(ranking) > 0 else ['NULL',  0]
		else:
			return ranking
