import pandas as pd
from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')
entity_desc = []


def filter_s(x):
    return ' '.join(tokenizer.tokenize(x))


with open('short_abstracts_en.ttl', 'r', encoding='utf-8') as f:
    for line in tqdm(f):
        if line.startswith('# completed') or line.startswith('# started'):
            pass
        else:
            tag, _, desc = line.split(' ', maxsplit=2)
            tag = tag.split('/')[-1][:-1]
            desc = filter_s(desc[1:-8])
            entity_desc.append([tag, desc])

pd.DataFrame(entity_desc, columns=['title', 'description']).to_csv(
    'DBPedia.csv', index=False)
