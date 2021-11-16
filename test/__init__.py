import os
import pandas as pd

cwd = os.path.dirname(__file__)

def get_candidates(doc, mention=None):
		"""
		Get candidates for tagged mentions in specified documents
		"""
		df =  pd.read_csv(os.path.join(cwd, 'candidates', f'{doc}.csv'))
		if mention is None:
			return df
		else:
			return df[df['forMention'] == mention]

def get_mentions(doc):
	"""
	Get candidates for tagged mentions in specified documents
	"""
	return pd.read_csv(os.path.join(cwd, 'mentions', f'{doc}.csv'))