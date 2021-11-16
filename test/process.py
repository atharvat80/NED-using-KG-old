import os
import pandas as pd
# import wikipedia as wiki

from tqdm import tqdm

ENT_COLS = ['text', 'normalName', 'predictedType', 'q', 'qid', 'docId', 'origText', 'url']
CAND_COLS = ['forMention', 'id', 'inCount', 'outCount', 'links', 'url', 'name', 'normalName', 'normalWikiTitle', 'predictedType']


def parse(x):
	return x[x.index(':') + 1:]


def process_doc(docId):
	ent_cand, entities, candidates = [], [], []
	with open(f'./raw/{docId}', 'r', encoding='utf-8') as f:
		ent_cand = f.read().split("ENTITY")[1:]

	for i in ent_cand:
		cands = i.split('\n')[:-1]
		entity = cands.pop(0).split('\t')[1:]
		entities.append(entity)
		cands = [j.replace('CANDIDATE', entity[6]) for j in cands]
		candidates += [j.split('\t')[:-1] for j in cands]
	
	ent_df = pd.DataFrame(entities, columns=ENT_COLS).applymap(parse).drop_duplicates()
	cand_df = pd.DataFrame(candidates, columns=CAND_COLS).applymap(parse).drop_duplicates()
	
	return ent_df, cand_df


def generate_candidates():
	for i in range(1, 1394):
		entities, candidates = process_doc(i)
		entities.to_csv(f'./mentions/{i}.csv', index=False)
		candidates.to_csv(f'./candidates/{i}.csv', index=False)


# def addSummary(start, end):
# 	for i in tqdm(range(start, end + 1)):
# 		print('')
# 		candidates = pd.read_csv(f'./candidates/{i}.csv')
# 		candidates['summary'] = ''
# 		c_urls = candidates['url'].unique()
# 		for url in c_urls:
# 			try:
# 				entity = url[29:]
# 				summary = wiki.summary(entity, sentences=5, auto_suggest=False, redirect=True)
# 				candidates.loc[candidates['url'] == url, 'summary'] = summary.replace('\n', ' ')
# 			except Exception as e:
# 				print("Error fetching", url)
# 				pass
# 		candidates.to_csv(f'./candidates/{i}.csv', index=False)


if __name__ == '__main__':
	generate_candidates()