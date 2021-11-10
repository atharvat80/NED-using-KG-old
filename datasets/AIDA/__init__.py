import os
import pandas as pd

class AIDA:
	def __init__(self):
		self.cwd = os.path.dirname(__file__) 

	def get_candidates(self, doc, mention=None):
		"""
		Get candidates for tagged mentions in specified documents
		"""
		df =  pd.read_csv(os.path.join(self.cwd, 'candidates', f'{doc}.csv'))
		if mention is None:
			return df
		else:
			return df[df['forMention'] == mention]

	def get_mentions(self, doc):
		"""
		Get candidates for tagged mentions in specified documents
		"""
		return pd.read_csv(os.path.join(self.cwd, 'mentions', f'{doc}.csv'))