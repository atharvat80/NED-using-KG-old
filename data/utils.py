import os
import pandas as pd
# import flair.datasets as datasets

from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer


cwd = os.path.dirname(__file__)


def filter_s(x):
    tokenizer = RegexpTokenizer(r'\w+')
    return ' '.join(tokenizer.tokenize(x))


def processDBPediaAbstracts():
    entity_desc = []
    with open(os.path.join(cwd, 'short_abstracts_en.ttl'), 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            if line.startswith('# completed') or line.startswith('# started'):
                pass
            else:
                tag, _, desc = line.split(' ', maxsplit=2)
                tag = tag.split('/')[-1][:-1]
                desc = filter_s(desc[1:-8])
                entity_desc.append([tag, desc])

        pd.DataFrame(entity_desc, columns=['title', 'description']).to_csv(
            os.path.join(cwd, 'DBPedia.csv'), index=False)


# def generateDocs():
# 	aida = datasets.NEL_ENGLISH_AIDA()
# 	aida = aida.train + aida.dev + aida.test
# 	full_text = '\n'.join([i.to_plain_string() for i in aida])
# 	docs = full_text.split('-DOCSTART-')[1:]
# 	for i, doc in enumerate(docs):
# 		with open(f"./docs/{i+1}.txt", 'w', encoding='utf-8') as f:
# 			f.write(doc)


def process_raw_doc(docId):
    ENT_COLS = ['text', 'normalName', 'predictedType',
                'q', 'qid', 'docId', 'origText', 'url']
    CAND_COLS = ['forMention', 'id', 'inCount', 'outCount', 'links', 'url', 'name', 'normalName',
                 'normalWikiTitle', 'predictedType']
    ent_cand, entities, candidates = [], [], []

    with open(os.path.join(cwd, 'raw', f'{docId}'), 'r', encoding='utf-8') as f:
        ent_cand = f.read().split("ENTITY")[1:]

    for i in ent_cand:
        cands = i.split('\n')[:-1]
        entity = cands.pop(0).split('\t')[1:]
        entities.append(entity)
        cands = [j.replace('CANDIDATE', entity[6]) for j in cands]
        candidates += [j.split('\t')[:-1] for j in cands]

    def parse(x): return x[x.index(':') + 1:]
    ent_df = pd.DataFrame(entities, columns=ENT_COLS).applymap(
        parse).drop_duplicates()
    cand_df = pd.DataFrame(candidates, columns=CAND_COLS).applymap(
        parse).drop_duplicates()

    return ent_df, cand_df


def processPPRforNEDData():
    for i in range(1, 1394):
        ground, candidates = process_raw_doc(i)
        # ground.to_csv(f'./ground/{i}.csv', index=False)
        candidates.to_csv(os.path.join(
            cwd, 'candidates', f'{i}.csv'), index=False)


def getCandidates(doc, mention=None):
    """
    Get candidates for tagged mentions in specified documents
    """
    df = pd.read_csv(os.path.join(cwd, 'candidates', f'{doc}.csv'))
    if mention is None:
        return df
    else:
        return df[df['forMention'] == mention]


def getDocument(num):
    with open(os.path.join(cwd, 'docs', f'{num}.txt'), 'r') as f:
        return f.read().replace('\n', ' ')
