import warnings
import sys
import pandas as pd
import wikipedia as wiki
from tqdm import tqdm

warnings.filterwarnings('ignore')

def addSummary(start, end):
	for i in tqdm(range(start, end + 1)):
		print('')
		candidates = pd.read_csv(f'./candidates/{i}.csv')
		candidates['summary'] = ''
		c_urls = candidates['url'].unique()
		for url in c_urls:
			try:
				entity = url[29:]
				summary = wiki.summary(entity, sentences=5, auto_suggest=False, redirect=True)
				candidates.loc[candidates['url'] == url, 'summary'] = summary.replace('\n', ' ')
			except Exception as e:
				print("Error fetching", url)
				pass
		candidates.to_csv(f'./candidates/{i}.csv', index=False)

if __name__ == '__main__':
	addSummary(int(sys.argv[1]), int(sys.argv[2]) + 1)