import pandas as pd
import flair.datasets as datasets
from tqdm import tqdm
from test import get_candidates
from src.utils import getEntityInfo

aida = datasets.NEL_ENGLISH_AIDA()
DBPedia_desc = pd.read_csv('DBPedia.csv')
DBPedia_entites = DBPedia_desc['title'].unique()


def get_desc(entity):
    try:
        return DBPedia_desc[DBPedia_desc['title'] == entity]['description'].values[0]
    except Exception as e:
        return ''


# Get all valid candidate entities for the test set
entities = []
entity_desc = []
missing = []
invalid = []

for i in range(1163, 1394):
    cands = get_candidates(i)
    entities += list(cands['url'].unique())

for i in aida.test:
    entities += [j.tag for j in i.get_spans()]

entities = [i.split('/')[-1] for i in set(entities)]
entities = list(set(entities))

for i in tqdm(entities):
    desc = get_desc(i).replace('\n', ' ')
    if desc == '':
        missing.append(i)
    else:
        entity_desc.append([i, desc])

for i in tqdm(missing):
    info, desc = getEntityInfo(i, inc_extract=True)
    if info != 'Topics referred to by the same term' and desc != '':
        entity_desc.append([i, desc.replace('\n', ' ')])
    else:
        invalid.append(i)

entity_desc_df = pd.DataFrame(entity_desc, columns=['entity', 'description'])
entity_desc_df.to_csv('data/entity_desc.csv', index=False)
