import numpy as np
import pandas as pd


class NGD(object):

    def __init__(self, num_iterations=50, alpha=0.1, eps=1e-6):
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.eps = eps

    def _fit_one(self):
        '''
        Do one ALS iteration. This method updates self.u
        and returns None.
        '''
        for i, rating in enumerate(self.x.T):
            if rating:
                u_new = np.linalg.lstsq(self.V.T[i].reshape((1, -1)),
                                        np.array([rating]))[0].T.clip(min=self.eps)
                self.u = self.u * (1.0 - self.alpha) + u_new * self.alpha

    def fit(self, x, V):
        self.x = x
        self.V = V
        self.u = np.random.rand(self.x.shape[0], self.V.shape[0])
        i = 0
        while i < self.num_iterations:
            i += 1
            self._fit_one()

    def reconstruction_error(self):
        '''
        Compute and return the reconstruction error of `x`
        '''
        est = []
        for i, rating in enumerate(self.x.T):
            if rating:
                estimate = np.dot(self.u, self.V.T[i])
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

    x = np.array(user_ratings).reshape(1, -1)
    V = np.array([factors for factors in items_matrix_factors]).T

    print(x.shape)
    ngd = NGD()
    ngd.fit(x, V)

    print(ngd.u)
