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
def WikiDataSearch(query, num_results=10):
    service_url = 'https://www.wikidata.org/w/api.php'
    params = {
        'action': 'wbsearchentities',
        'search': query,
        'language': 'en',
        'format': 'json',
        'limit': num_results,
        # 'props': '',
    }
    url = service_url + '?' + urllib.parse.urlencode(params)
    response = json.loads(urllib.request.urlopen(url).read())
    results = [i for i in response['search'] if 'disambiguation' not in i['description']]
    # add wikipedia url to results
    params = {
        'action': 'wbgetentities',
        'ids': '|'.join([i['id'] for i in results]),
        'props': 'sitelinks',
        'sitefilter': 'enwiki',
        'format': 'json',
    }
    url = service_url + '?' + urllib.parse.urlencode(params)
    response = json.loads(urllib.request.urlopen(url).read())
    response = list(response['entities'].values())
    for i in range(len(results)):
        results[i]['label'] = response[i]['sitelinks']['enwiki']['title']
        
    return results


def WikiSearch(query, num_results=10):
    BASE_URL = "https://en.wikipedia.org/w/index.php"
    params = {
        "search": query,
        "title": "Special:Search",
        "profile": "advanced",
        "fulltext": "1"
    }
    url = BASE_URL + '?' + urllib.parse.urlencode(params)
    response = urllib.request.urlopen(url).read().decode('utf-8')
    soup = BeautifulSoup(response, 'html.parser')

    results = []
    for i in soup.find_all('li'):
        if 'class' in i.attrs.keys() and 'mw-search-result' in i.attrs['class']:
            results.append(i)
    candidates = [{
        'label': i.findAll('a')[0].attrs['title'].replace(' ', '_'),
        'description': i.findAll('div')[1].text
    } for i in results[:num_results]]

    return candidates


# ---------------------------------------
#  Other
# ---------------------------------------

def getEntitySummary(url, num_sentences=3):
    try:
        return wiki.summary(url.split('/')[-1], redirect=True, auto_suggest=False,
                            sentences=num_sentences).replace('\n', ' ')
    except:
        return ''
