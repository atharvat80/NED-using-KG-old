import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

class GBRT():
	def __init__(self, emb):
		self.emb = emb
		self.tokenizer = RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')
		self.stop_words = set(stopwords.words('english'))

	def get_nouns(self, s):
		s = self.tokenizer.tokenize(s)
		nouns = []
		for word, pos in nltk.pos_tag(s):
			if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
				nouns.extend(word.split(' '))
		filtered = [w for w in nouns if not w in self.stop_words]
		return list(set(filtered))

	def encode_entity(self, entity):
		if self.emb.get_entity(entity) is not None:
			return self.emb.get_entity_vector(entity)
		else:
			return np.zeros((100,))

	def encode_context(self, s, mention):
		nouns = [i for i in self.get_nouns(s) if i != mention]
		all = []
		for i in nouns:
			try:
				all.append(self.emb.get_word_vector(i.lower()))
			except:
				pass
		return sum(all)/len(all) if len(all) > 0 else np.zeros((100,))

	def cosSim(self, v1, v2):
		return np.dot(v2, v1)/(np.linalg.norm(v2) * np.linalg.norm(v1))
		
	def rank(self, candidates, context):
		ranking = []
		for candidate in candidates:
			if candidates[candidate] is not None:
				score = self.cosSim(context, candidates[candidate])
				if not(np.isnan(score)):
					ranking.append([candidate, score])
		return ranking

	def link(self, mention, doc, candidates):
		# Convert everything to lowercase
		mention = mention.lower()
		doc = doc.lower()

		# Encode context and candidates
		contextEnc = self.encode_context(doc, mention)
		candidateEnc = {i: self.encode_entity(i.replace('_', ' ')) for i in candidates}

		ranking = self.rank(candidateEnc, contextEnc)
		if ranking:
			ranking.sort(key=lambda x: x[1], reverse=True)
			return ranking
		else:
			return [['NIL', 0]]