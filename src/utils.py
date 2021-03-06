import json
import urllib

from googlesearch import search


def GoogleSearch(query, num_results=10, desc=False):
	results = search(f"{query} site:en.wikipedia.org", num_results=num_results)
	if desc:
		return [{'label': i[30:], 'description': getEntityInfo(i)} for i in results]
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
				results[i]['label'] = response['entities'][i]['sitelinks']['enwiki']['title']
			except:
				to_remove.append(i)
		for i in to_remove:
			del results[i]
		return [results[i] for i in results.keys()]
	else:
		return []


def WikiSearch(query, num_results=20):
	service_url = 'https://en.wikipedia.org/w/api.php'
	search_params = {
		'action': 'opensearch',
		'search': query,
		'namespace': 0,
		'limit': num_results,
		'redirects': 'resolve',
	}

	results = make_request(service_url, search_params)[1]
	results = [i.replace(' ', '_')
			   for i in results if 'disambiguation' not in i.lower()]
	return results


def getLinks(entity):
	def parse_response(data):
		pages = data["query"]["pages"]
		page_titles = []
		for _, val in pages.items():
			for link in val["links"]:
				page_titles.append(link["title"])
		return page_titles

	url = "https://en.wikipedia.org/w/api.php"
	params = {
		"action": "query",
		"format": "json",
		"titles": entity,
		"prop": "links",
		"pllimit": "max",
		"plnamespace": 0
	}
	# make the inital request
	data = make_request(url, params)
	page_titles = parse_response(data)	
	# keep going until the last page
	while "continue" in data:
		plcontinue = data["continue"]["plcontinue"]
		params["plcontinue"] = plcontinue
		data = make_request(url, params)
		page_titles += parse_response(data)

	return page_titles

# ---------------------------------------
#  Other
# ---------------------------------------

def make_request(service_url, params):
	url = service_url + '?' + urllib.parse.urlencode(params)
	response = json.loads(urllib.request.urlopen(url).read())
	return response


def getEntityInfo(entity_title, inc_extract=True, num_sentences=1):
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
		params['exsentences'] = num_sentences

	res = make_request(service_url, params)['query']['pages']
	res = res[list(res.keys())[0]]
	desc = res['description'] if 'description' in res.keys() else ''
	extract = res['extract'] if 'extract' in res.keys() else ''
	return desc, extract

# ---------------------------------------
#  reference queries
# ---------------------------------------
# https://en.wikipedia.org/w/api.php?action=query&redirects=1&titles=Japan+national+football+team&format=json&prop=extracts|description|pageprops&explaintext=1&exsectionformat=plain&exsentences=2
# https://en.wikipedia.org/w/api.php?action=opensearch&search=Japan+Football&namespace=0&limit=20&redirects=resolve
