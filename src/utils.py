import os
import json
import urllib
import pandas as pd
import wikipedia as wiki
import wikipediaapi

from bs4 import BeautifulSoup
from googlesearch import search


# ---------------------------------------
#  Candidate Generation Functions
# ---------------------------------------

API_KEY = open(os.path.join(os.path.dirname(__file__), '.api_key')).read()
wiki_api = wikipediaapi.Wikipedia('en')


def GoogleKBSearch(query, num_res=10, as_df=False):
	service_url = 'https://kgsearch.googleapis.com/v1/entities:search'
	columns = ['name', 'description', 'type', 'detailedDescription', 'url']
	params = {
		'query': query,
		'limit': num_res,
		'indent': True,
		'key': API_KEY,
	}
	url = service_url + '?' + urllib.parse.urlencode(params)
	response = json.loads(urllib.request.urlopen(url).read())
	parsed = []
	for i in response['itemListElement']:
		result = i['result']
		if 'description' in result.keys():
			parsed.append([
				result['name'], result['description'], ','.join(
					result['@type']),
				result['detailedDescription']['articleBody'], result['detailedDescription']['url'],
			])
	return pd.DataFrame(parsed, columns=columns) if as_df else parsed


def GoogleSearch(query, num_results=10, desc=False):
	results = search(f"{query} site:en.wikipedia.org", num_results=num_results)
	if desc:
		return [{'label': i[30:], 'description': getEntitySummary(i)} for i in results]
	else:
		return [{'label': i[30:]} for i in results]


# https://stackoverflow.com/questions/27452656/wikidata-entity-value-from-name
# https://www.wikidata.org/w/api.php?action=help&modules=wbsearchentities
def WikiDataSearch(query, num_results=12):
	# Get WikiData entities from wikidata
	service_url = 'https://www.wikidata.org/w/api.php'
	params = {
		'action': 'wbsearchentities',
		'search': query,
		'language': 'en',
		'format': 'json',
		'limit': num_results,
	}
	url = service_url + '?' + urllib.parse.urlencode(params)
	response = json.loads(urllib.request.urlopen(url).read())
	results = {}
	for i in response['search']:
		if 'description' in i.keys() and 'disambiguation' not in i['description']:
			results[i['id']] = {'label': i['label'],
				'description': i['description']}

	# add wikipedia url to results
	params = {
		'action': 'wbgetentities',
		'ids': '|'.join([i for i in results.keys()]),
		'props': 'sitelinks',
		'sitefilter': 'enwiki',
		'format': 'json',
	}
	url = service_url + '?' + urllib.parse.urlencode(params)
	response = json.loads(urllib.request.urlopen(url).read())
	if 'entities' in response.keys():
		to_remove = []
		for i in results.keys():
			try:
				results[i]['label'] = response['entities'][i]['sitelinks']['enwiki']['title'].replace(
					' ', '_')
			except:
				to_remove.append(i)
		for i in to_remove:
			del results[i]
		return [results[i] for i in results.keys()]
	else:
		return []


def WikiSearch(query, num_results=20, inc_extract=False):
	service_url = 'https://en.wikipedia.org/w/api.php'
	search_params = {
		'action':'opensearch',
		'search':query,
		'namespace':0,
		'limit':num_results,
		'redirects':'resolve',
	}
	
	results = make_request(service_url, search_params)[1]
	results = [i for i in results if 'disambiguation' not in i.lower()]
	candidates = []
	# for i in results:
	# 	desc, extract = get_info(i, inc_extract=inc_extract)
	# 	if inc_extract:
	# 		candidates.append({
	# 			'label': i.replace(' ', '_'),
	# 			'description': desc,
	# 			'extract': extract
	# 		})
	# 	else:
	# 		candidates.append({
	# 			'label': i.replace(' ', '_'),
	# 			'description': desc
	# 		})

	# return candidates
	return results


# ---------------------------------------
#  Other
# ---------------------------------------

def make_request(service_url, params):
	url = service_url + '?' + urllib.parse.urlencode(params)
	response = json.loads(urllib.request.urlopen(url).read())
	return response


def get_info(entity_title, inc_extract=False):
	service_url = 'https://en.wikipedia.org/w/api.php'
	params = {
		'action': 'query',
		'titles': entity_title,
		'prop': 'description|extracts' if inc_extract else 'description',
		'redirects': 1,
		'format': 'json',
	}
	
	if inc_extract:
		params['explaintext'] = 1
		params['exsectionformat'] = "plain"
		params['exsentences'] = 1
	
	res = make_request(service_url, params)['query']['pages']
	res = res[list(res.keys())[0]]
	desc = res['description'] if 'description' in res.keys() else ''
	extract = res['extract'] if 'extract' in res.keys() else ''
	return desc, extract


def getEntitySummary(url, num_sentences=3):
	try:
		return wiki.summary(url.split('/')[-1], redirect=True, auto_suggest=False,
							sentences=num_sentences).replace('\n', ' ')
	except:
		return ''

# ---------------------------------------
#  reference queries
# ---------------------------------------
# https://en.wikipedia.org/w/api.php?action=query&redirects=1&titles=Japan+national+football+team&format=json&prop=extracts|description|pageprops&explaintext=1&exsectionformat=plain&exsentences=2
# https://en.wikipedia.org/w/api.php?action=opensearch&search=Japan+Football&namespace=0&limit=20&redirects=resolve
