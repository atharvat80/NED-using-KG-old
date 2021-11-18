import os
import urllib
import gensim.downloader as api

# Load models from 
print("Downloading wikipedia2vec...")
urllib.urlretrieve(
	"http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_win10_100d.txt.bz2", 
	"wiki2vec_w10_100d.txt.bz2"
)
print("Extracting downloaded file...")
os.system("bzip2 -d wiki2vec_w10_100d.txt.bz2")
print("Done.")

# Load models about gensim
models = ["fasttext-wiki-news-subwords-300", "glove-wiki-gigaword-300", "word2vec-google-news-300",]
for i in models:
	print("Loading", i)
	model = api.load(i)
	model.save(i, pickle_protocol=4)
	print("Done.")