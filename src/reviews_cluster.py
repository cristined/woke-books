import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import load_data


class ReviewClusters(object):
    def __init__(self):
        self.reviews = None
        self.cluster_num = None
        self.max_features = None

    def fit(self, reviews, cluster_num, max_features):
        self.reviews = reviews
        self.cluster_num = cluster_num
        self.max_features = max_features
        self._vectorize()
        self._cluster()

    def _vectorize(self):
        vect = TfidfVectorizer(stop_words='english', tokenizer=self.tokenizer,
                               max_features=self.max_features)
        vector_matrix = vect.fit_transform(self.reviews)

        self.vectors = vector_matrix.toarray()
        self.cols = vect.get_feature_names()

    def _cluster(self):
        self.kmeans = KMeans(n_clusters=self.cluster_num).fit(self.vectors)
        self.kmeans.labels_

        self.k_books = pd.DataFrame(list(zip(list(self.kmeans.labels_),
                                    list(self.reviews.index))),
                                    columns=['k_label', 'book_id'])

    def save_k_books(self, path):
        self.k_books.set_index('k_label').to_csv(path)

    def print_centroid_vocab(self, n):
        centroids = self.kmeans.cluster_centers_
        for ind, c in enumerate(centroids):
            print(ind)
            indices = c.argsort()[-1:-n-1:-1]
            print([self.cols[i] for i in indices])
            print("=="*20)

    def tokenizer(self, doc):
        stops = set(nltk.corpus.stopwords.words('english'))
        stemmer = WordNetLemmatizer()
        doc = word_tokenize(doc.lower())
        tokens = [''.join([char for char in tok if char not in
                  string.punctuation]) for tok in doc]
        tokens = [tok for tok in tokens if tok]
        if stops:
            tokens = [tok for tok in tokens if (tok not in stops)]
        if stemmer:
            tokens = [stemmer.lemmatize(tok) for tok in tokens]
        return tokens

    def print_top_books(self, n):
        k_books = pd.DataFrame(list(zip(list(self.kmeans.labels_),
                               list(self.reviews.index))),
                               columns=['k_label', 'book_id'])
        self.k_counter = Counter(k_books['k_label'])
        df_books = load_data.get_books()
        self.df_k_books = pd.merge(df_books[['best_book_id', 'title']], k_books, how='inner',
                                   left_on='best_book_id', right_on='book_id')
        for i in range(self.cluster_num):
            print(i)
            print(list(self.df_k_books[self.df_k_books['k_label'] == i]['title'])[:n])
            print("=="*20)


if __name__ == '__main__':
    # Created from Amazon Review file
    a_reviews_file = '../data/limited_amazon_reviews.csv'

    df_reviews_agg = load_data.get_amazon_review_text(a_reviews_file)

    rc = ReviewClusters()
    rc.fit(df_reviews_agg, 13, 1000)
    rc.save_k_books('../data/13_kmeans_book_id.csv')
    rc.print_centroid_vocab(15)
    rc.print_top_books(15)

    rc = ReviewClusters()
    rc.fit(df_reviews_agg, 10, 1000)
    rc.save_k_books('../data/10_kmeans_book_id.csv')
    rc.print_centroid_vocab(15)
    rc.print_top_books(15)
