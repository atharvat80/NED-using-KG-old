import nltk
import numpy as np
from src.models.wiki2vec import BaseWiki2Vec 

class GBRT(BaseWiki2Vec):
	def __init__(self, emb, cased=True, filter_stopwords=True, cand_source='wiki', num_cands=5):
		super().__init__(emb, cased, filter_stopwords, cand_source, num_cands)
		self.vector_size = 100

	def filter(self, s):
		word_tokens = self.tokenizer.tokenize(s)
		nouns = []
		# Only keep nouns from the sentense
		for word, pos in nltk.pos_tag(word_tokens):
			if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
				nouns.append(word)
		return [w for w in nouns if not w.lower() in self.stop_words]