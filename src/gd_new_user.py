import numpy as np
import pandas as pd
import load_data
import get_user



class GD(object):
    """
    Perform the non-negative gradient descent optimization algorithm
    """

    def __init__(self, num_iterations=50, alpha=0.1, eps=1e-6, negative=True):
        """Initialize the instance attributes of a NGD object.
        Parameters
        ----------
        alpha: float
            The learning rate.
        num_iterations: integer.
            Number of iterations to use in the descent.
        eps: float
            epsilon for clipping
        Returns
        -------
        self:
            The initialized NGD object.
        """
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.eps = eps
        self.item_books = None
        self.item_factors = None
        self.user_ratings = None
        self.user_factors = None
        self._fit_one = self._fit_one_non_negative
        if negative:
            self._fit_one = self._fit_one_gradient

    def fit(self, df_user_rating, item_matrix):
        '''
        Do number of iterations of ALS iteration.
        '''
        self.item_books = item_matrix[::, 0]
        self.item_factors = np.array([factors for factors
                                      in item_matrix[::, 1]]).T
        self.user_factors = np.random.uniform(low=0.0, high=1.0,
                                              size=(1, self.item_factors.shape[0]))
        self.user_ratings = df_user_rating.set_index('book_id')['rating'].to_dict()
        for i in range(self.num_iterations):
            self._fit_one()
        return self

    def _fit_one_non_negative(self):
        '''
        Do one ALS iteration. This method updates self.u
        and returns None.
        '''
        for book, rating in self.user_ratings.items():
            try:
                i = np.where(self.item_books == book)[0][0]
                u_new = np.linalg.lstsq(self.item_factors.T[i].reshape((1, -1)),
                                        np.array([rating]))[0].T.clip(min=self.eps)
                self.user_factors = self.user_factors * (1.0 - self.alpha) + u_new * self.alpha
            except IndexError:
                pass

    def _fit_one_gradient(self):
        for book, rating in self.user_ratings.items():
            grad = self._gradient(book, rating)
            self.user_factors = self.user_factors - (self.alpha * grad)

    def _gradient(self, book, rating):
        """Run the gradient descent algorithm for one repititions.
        Parameters
        ----------
        x: ndarray, rating matrix
            The training data for the optimization.
        V: ndarray, items data
            The training response for the optimization.
        u
        Returns
        -------
        The next gradient in U
        """
        grad = np.zeros_like(self.user_factors)
        try:
            i = np.where(self.item_books == book)[0][0]
        except IndexError:
            return grad
        for j, item in enumerate(grad[0]):
            s = - 2 * self.item_factors[j][i] * (rating - np.dot(self.user_factors, self.item_factors.T[i]))
            grad[0][j] = float(s)
        return grad

    def reconstruction_error(self):
        '''
        Compute and return the reconstruction error of `x`
        '''
        est = []
        for book, rating in self.user_ratings.items():
            try:
                i = np.where(self.item_books == book)[0][0]
                estimate = np.dot(self.user_factors, self.item_factors.T[i])
                est.append(rating - estimate)
                # print(book, rating, estimate)
            except IndexError:
                pass
        return (np.array(est) ** 2 / len(est)).sum()


if __name__ == '__main__':
    items_matrix = np.load('../data/item_matrix.npy')
    items_matrix_books = items_matrix[::, 0]
    items_matrix_factors = items_matrix[::, 1]

    # Created from Amazon Review file for ASIN and GoodReads API
    asin_best_file = '../data/asin_best_book_id.csv'
    df_isbn_best_book_id = load_data.get_isbn_to_best_book_id(asin_best_file)
    import os
    api_key = os.environ['GOODREADS_API_KEY']

    df_user = get_user.get_user_read_books(2624891, api_key, df_isbn_best_book_id)


    gd = GD(alpha=.01, num_iterations=100, negative=True)
    gd.fit(df_user, items_matrix)
    print(gd.user_factors)
    print(gd.reconstruction_error())

    gd_neg = GD(alpha=.01, num_iterations=100, negative=False)
    gd_neg.fit(df_user, items_matrix)
    print(gd_neg.user_factors)
    print(gd_neg.reconstruction_error())
