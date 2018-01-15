import numpy as np
import pandas as pd


class NMF(object):

    def __init__(self, max_iter=50, alpha=0.5, eps=1e-6):
        self.max_iter = max_iter
        self.alpha = alpha
        self.eps = eps

    def _fit_one(self):
        '''
        Do one ALS iteration. This method updates self.W
        and returns None.
        '''
        for i, rating in enumerate(self.V.T):
            if rating:
                W_new = np.linalg.lstsq(self.H.T[i].reshape((1, -1)),
                                        np.array([rating]))[0].T.clip(min=1e-6)
                self.W = self.W * (1.0 - self.alpha) + W_new * self.alpha

    def fit(self, V, H):
        self.V = V
        self.H = H
        self.W = np.random.rand(self.V.shape[0], self.H.shape[0])
        i = 0
        while i < self.max_iter:
            i += 1
            self._fit_one()

    def reconstruction_error(self):
        '''
        Compute and return the reconstruction error of `V`
        '''
        est = []
        for i, rating in enumerate(self.V.T):
            if rating:
                estimate = np.dot(self.W, self.H.T[i])
                est.append(rating - estimate)
        return (np.array(est) ** 2).sum()

if __name__ == '__main__':

    user_matrix = np.load('../data/user_matrix.npy')
    print(user_matrix[400])
    items_matrix = np.load('../data/item_matrix.npy')
    items_matrix_books = items_matrix[::, 0]
    items_matrix_factors = items_matrix[::, 1]

    df_user_ratings = pd.read_csv('../data/goodbooks-10k/ratings.csv')
    df_user_ratings = df_user_ratings[df_user_ratings['user_id'] == 4010]
    df_books_gr = pd.read_csv('../data/goodbooks-10k/books.csv')
    d = df_books_gr.set_index('book_id')['best_book_id'].to_dict()
    df_user_ratings['book_id'] = df_user_ratings['book_id'].map(lambda x: d.get(x, 0))
    dict_u_rate = df_user_ratings.set_index('book_id')['rating'].to_dict()
    user_ratings = [dict_u_rate.get(book, None) for book in items_matrix_books]

    V = np.array(user_ratings).reshape(1, -1)
    H = np.array([factors for factors in items_matrix_factors]).T

    print(V.shape)
    nmf = NMF()
    nmf.fit(V, H)

    print(nmf.W)
