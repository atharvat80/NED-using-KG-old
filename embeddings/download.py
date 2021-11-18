import gensim.downloader as api

model = api.load("fasttext-wiki-news-subwords-300")
model.save("fasttext-wiki-news-subwords-300", pickle_protocol=4)

model = api.load("glove-wiki-gigaword-300")
model.save("glove-wiki-gigaword-300", pickle_protocol=4)

model = api.load("word2vec-google-news-300")
model.save("word2vec-google-news-300", pickle_protocol=4)