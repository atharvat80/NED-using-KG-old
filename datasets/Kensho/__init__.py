import os
import pandas as pd

class Kensho:
	def __init__(self):
		cwd = os.path.dirname(__file__)
		self.items = pd.read_csv(os.path.join(cwd, 'item.csv.zip'), keep_default_na=False)
		self.item_aliases = pd.read_csv(os.path.join(cwd, 'item_aliases.csv.zip'), keep_default_na=False)

	def wikipedia_url_from_title(self, title):
		return 'https://en.wikipedia.org/wiki/{}'.format(title.replace(' ', '_'))

	def wikipedia_url_from_page_id(self, page_id):
		return 'https://en.wikipedia.org/?curid={}'.format(page_id)

	def wikidata_url_from_item_id(self, item_id):
		return 'https://www.wikidata.org/entity/Q{}'.format(item_id)