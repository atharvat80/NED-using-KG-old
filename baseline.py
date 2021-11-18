# %% [markdown]
# # Setup

# %%
import pandas as pd
import torch, flair
import flair.datasets as datasets

from tqdm import tqdm
from gensim.models import KeyedVectors
from flair.embeddings import BytePairEmbeddings
from flair.embeddings import DocumentPoolEmbeddings, TransformerDocumentEmbeddings #,ELMoEmbeddings
from wikipedia2vec import Wikipedia2Vec
from src.models.base import Base, BaseFlair, BaseWiki2Vec
from test import get_candidates

# %%
aida = datasets.NEL_ENGLISH_AIDA()
mentions_tags = []
doc = 1162
for i in aida.test:
	context = i.to_plain_string()
	if context != '-DOCSTART-':
		mentions_tags += [[j.text, j.tag, context, doc] for j in i.get_spans()]
	else:
		doc += 1

# %%
entity_desc = pd.read_csv('./data/entity_desc.csv')

def get_entity_desc(entity):
	try:
		return entity_desc[entity_desc['entity'] == entity]['description'].values[0]
	except:
		return ''

def candidates(mention, doc):
	df = get_candidates(doc, mention=mention)
	res = [i.split('/')[-1] for i in df['url'].values]
	cands = []
	for i in res:
		desc = get_entity_desc(i)
		if desc != '':
			cands.append([i, desc])
	return cands

# %% [markdown]
# # Testing base NED model using various types of embeddings

# %%
def test_model(model):
	preds = []
	skipped = 0
	try:
		for mention, tag, context, doc in tqdm(mentions_tags):
			cands = candidates(mention, doc)
			# Check the tag is a valid entity and is present in the candidate set
			if tag in [i[0] for i in cands]:
				pred_tag, conf = model.link(mention, context, candidates=cands)
				preds.append([mention, tag, pred_tag, conf])
			else:
				skipped += 1
	except Exception as e:
		print(e)

	print(skipped, 'mentions skipped due to invalid tag or lack of candidates.')
	res = pd.DataFrame(preds, columns=['mention', 'tag', 'predicted', 'confidence'])
	acc = (res[res['tag'] == res['predicted']].shape[0]/res.shape[0])*100
	return res, acc

# %% [markdown]
# ### Word2Vec Google News 300d

# %%
w2v = KeyedVectors.load('./embeddings/word2vec-google-news-300')
w2v_res, w2v_acc = test_model(Base(w2v))
w2v_res.to_csv('./results/base_word2vec.csv')
print('Word2Vec', w2v_acc) 
# Cased   : 52.35 %
# Uncased : 47.76 %

# %% [markdown]
# ### Glove Wiki-Gigaword 300d

# %%
glove = KeyedVectors.load('./embeddings/glove-wiki-gigaword-300')
glove_res, glove_acc = test_model(Base(glove))
glove_res.to_csv('./results/base_glove.csv')
print('Glove', glove_acc, end='') 
# Cased   : 51.95 %
# Uncased : 51.95 %

# %% [markdown]
# ### Byte-Pair, 300d

# %%
byte_pair = BytePairEmbeddings('en', dim=300, syllables=200000)
bp_doc_emb = DocumentPoolEmbeddings([byte_pair], fine_tune_mode='nonlinear')
bp_res, bp_acc = test_model(BaseFlair(bp_doc_emb))
bp_res.to_csv('./results/base_byte_pair.csv')
print('Byte Pair', bp_acc, end='')
# Cased   : 50.54 %
# Uncased : 49.51 %

# %% [markdown]
# ### FastText Wiki-News Subword 300d

# %%
ftext = KeyedVectors.load('./embeddings/fasttext-wiki-news-subwords-300')
ftext_res, ftext_acc = test_model(Base(ftext))
ftext_res.to_csv('./results/base_fasttext.csv')
print('Fasttext', ftext_acc, end='') 
# Cased   : 39.43 %
# Uncased : 38.75 %

# %% [markdown]
# ### Wikipedia2vec

# %% [markdown]
# #### 1. 100d (Window=10, NegSample=15)

# %%
wiki2vec = Wikipedia2Vec.load('./embeddings/wiki2vec_w10_100d.pkl')
wiki2vec_res, wiki2vec_acc = test_model(BaseWiki2Vec(wiki2vec))
wiki2vec_res.to_csv('./results/base_wiki2vec.csv')
print('Word2Vec', wiki2vec_acc, end='')
# Cased   : 62.05%
# Uncased : 62.05% 

# %% [markdown]
# #### 2. 300d (Window=10, NegSample=15)

# %%
wiki2vec = Wikipedia2Vec.load('./embeddings/wiki2vec_w10_300d.pkl')
wiki2vec_res, wiki2vec_acc = test_model(BaseWiki2Vec(wiki2vec))
wiki2vec_res.to_csv('./results/base_wiki2vec.csv')
print('Word2Vec', wiki2vec_acc, end='')
# Cased   : 58.78 %
# Uncased : 58.78 % 

# %% [markdown]
# ### RoBERTa

# %%
flair.device = torch.device('cuda')
roberta_doc_emb = TransformerDocumentEmbeddings('roberta-base')
with torch.no_grad():
	roberta_res, roberta_acc = test_model(BaseFlair(roberta_doc_emb))
	roberta_res.to_csv('./results/base_roberta.csv')
	print('Roberta Base', roberta_acc)
# Cased   : - %
# Uncased : - % 

# %% [markdown]
# ### ELMo

# %%
# elmo = ELMoEmbeddings('original')


