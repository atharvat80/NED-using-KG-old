import numpy as np
import torch.nn.functional as F

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from flair.data import Sentence

from src.utils import GoogleSearch, WikiSearch, getEntityInfo


class Base():
	def __init__(self, emb, cased=True, filter_stopwords=True, cand_source='wiki', num_cands=5):
		self.stop_words = set(stopwords.words('english'))
		self.tokenizer = RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')  # 'r'\w+'
		self.cand_source = cand_source
		self.num_cands = num_cands
		self.emb = emb
		self.cased = cased
		self.filter_stopwords = filter_stopwords

	def filter(self, s):
		word_tokens = self.tokenizer.tokenize(s)
		return [w for w in word_tokens if not w.lower() in self.stop_words]

	def generateCandidates(self, mention):
		res = []
		if self.cand_source == 'wiki':
			res = WikiSearch(mention, self.num_cands)
		elif self.cand_source == 'google':
			res = GoogleSearch(mention, self.num_cands)
		return [[i, getEntityInfo(i, inc_extract=True)[1]] for i in res]

	def cosSim(self, v1, v2):
		return np.dot(v2, v1)/(np.linalg.norm(v2) * np.linalg.norm(v1))

	def encodeSentence(self, s):
		s = s if self.cased else s.lower()
		words = self.filter(s) if self.filter_stopwords else self.tokenizer.tokenize(s)
		enc = []
		for w in words:
			if w in self.emb.vocab:
				enc.append(self.emb.get_vector(w))
			elif w.lower() in self.emb.vocab:
				enc.append(self.emb.get_vector(w.lower()))
		try:
			return sum(enc)/len(enc)
		except:
			return np.zeros((self.emb.vector_size,))

	def rank(self, candidates, context):
		ranking = []
		for candidate in candidates:
			if candidates[candidate] is not None:
				score = self.cosSim(context, candidates[candidate])
				ranking.append([candidate, score])
		return ranking

	def link(self, mention, context, candidates=None, top_only=True):
		# 1. Candidate Generation
		candidates = self.generateCandidates(mention) if candidates is None else candidates
		candidates = [i for i in candidates if i[1] != '']

		# 2. Encode context-mention and candidates
		context = self.encodeSentence(context)
		candidateEnc = {i: self.encodeSentence(j) for i, j in candidates}

		# 3. Candidate Ranking
		ranking = self.rank(candidateEnc, context)
		ranking.sort(key=lambda x: x[1], reverse=True)
		if top_only:
			return ranking[0] if len(ranking) > 0 else ['NULL',  0]
		else:
			return ranking
