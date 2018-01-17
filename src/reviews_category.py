import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df_reviews = pd.read_csv('../old_data/new_less_reviews.csv', header=None,
                         names=['asin','reviewerID','helpful','overall','summary',
                                'reviewText','unixReviewTime'])

    df_reviews = df_reviews[df_reviews['reviewText'].isnull() == False]

    df_books = pd.read_csv('../data/updated_books.csv')

    df_books_review = pd.merge(df_books, df_reviews, left_on='isbn', right_on='asin', how='left')
    df_books_review = df_books_review[df_books_review['reviewerID'].isnull() == False]

    reviewText_sample = df_reviews['reviewText'].sample(frac=0.1, random_state=200, axis=0)

    docs = df_reviews.groupby('asin')['reviewText'].agg(lambda x: ' '.join(x))

    snowball = SnowballStemmer('english')
    docs_snowball = [[snowball.stem(word) for word in words] for words in docs]

    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    vectors = vectorizer.fit_transform(docs_snowball).toarray()
    cols = vectorizer.get_feature_names()

    model = NMF(n_components=10, init='random', random_state=0)
    W = model.fit_transform(vectors)
    H = model.components_

    for i in range(10):
        print('--'*10)
        print(' - '.join(np.array(cols)[np.argsort(H[i])[::-1]][:20]))
