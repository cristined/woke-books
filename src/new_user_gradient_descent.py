import numpy as np
import pandas as pd
import get_user
import load_data

# def gradient(x, V, u):
#     """Run the gradient descent algorithm for one repititions.
#     Parameters
#     ----------
#     x: ndarray, dense as fuck rating matrix
#         The training data for the optimization.
#     V: ndarray, items data
#         The training response for the optimization.
#     u
#     Returns
#     -------
#     The next gradient in U
#     """
#     grad = np.zeros_like(u)
#     for i, rating in enumerate(x):
#         if rating:
#             for j, item in enumerate(grad[0]):
#                 s = - 2 * V[j][i] * (rating - np.dot(u, V.T[i]))
#                 grad[0][j] = float(s)
#     return grad
#
#
# class GradientDescent(object):
#     """
#     Perform the gradient descent optimization algorithm
#     """
#
#     def __init__(self, gradient, alpha=0.01, num_iterations=100):
#         """Initialize the instance attributes of a GradientDescent object.
#         Parameters
#         ----------
#         alpha: float
#             The learning rate.
#         num_iterations: integer.
#             Number of iterations to use in the descent.
#         Returns
#         -------
#         self:
#             The initialized GradientDescent object.
#         """
#         self.gradient = gradient
#         self.u = None
#         self.alpha = alpha
#         self.num_iterations = num_iterations
#
#     def fit(self, x, V):
#         """Run the gradient descent algorithm for num_iterations repititions.
#         Parameters
#         ----------
#         x: ndarray, sparse rating matrix
#             The training data for the optimization.
#         V: ndarray, items data
#             The training response for the optimization.
#         Returns
#         -------
#         self:
#             The fit GradientDescent object.
#         """
#         self.u = np.random.uniform(low=0.0, high=1.0, size=(1, V.shape[0]))
#         print("Begin, U = {}".format(self.u))
#         for i in range(self.num_iterations):
#             grad = self.gradient(x, V, self.u)
#             print("Grad: {}\nSame as last U: {}".format(grad, self.u))
#             self.u = self.u - (self.alpha * grad)
#             print("Grad: {}\nU: {}".format(grad, self.u))
#             print("After {}, U = {}".format(i, self.u))
#         return self


# def gradient(x, V, u):
#     """Run the gradient descent algorithm for one repititions.
#     Parameters
#     ----------
#     x: ndarray, dense as fuck rating matrix
#         The training data for the optimization.
#     V: ndarray, items data
#         The training response for the optimization.
#     u
#     Returns
#     -------
#     The next gradient in U
#     """
#     grad = np.zeros_like(u)
#     for i, rating in enumerate(x):
#         if rating:
#             for j, item in enumerate(grad[0]):
#                 s = - 2 * V[j][i] * (rating - np.dot(u, V.T[i]))
#                 grad[0][j] = float(s)
#     return grad


class GradientDescent(object):
    """
    Perform the gradient descent optimization algorithm
    """

    def __init__(self, alpha=0.01, num_iterations=100):
        """Initialize the instance attributes of a GradientDescent object.
        Parameters
        ----------
        alpha: float
            The learning rate.
        num_iterations: integer.
            Number of iterations to use in the descent.
        Returns
        -------
        self:
            The initialized GradientDescent object.
        """
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.item_books = None
        self.item_factors = None
        self.user_ratings = None
        self.user_factors = None

    def fit(self, df_user_rating, item_matrix):
        """Run the gradient descent algorithm for num_iterations repititions.
        Parameters
        ----------
        x: ndarray, sparse rating matrix
            The training data for the optimization.
        V: ndarray, items data
            The training response for the optimization.
        Returns
        -------
        self:
            The fit GradientDescent object.
        """

        self.item_books = item_matrix[::, 0]
        self.item_factors = np.array([factors for factors
                                      in item_matrix[::, 1]]).T
        self.user_factors = np.random.uniform(low=0.0, high=1.0,
                                              size=(1, self.item_factors.shape[0]))
        self.user_ratings = df_user_rating.set_index('book_id')['rating'].to_dict()
        for i in range(self.num_iterations):
            self._fit_one()
        return self

    def _fit_one(self):
        for book, rating in self.user_ratings.items():
            grad = self.gradient(book, rating)
            self.user_factors = self.user_factors - (self.alpha * grad)

    def gradient(self, book, rating):
        """Run the gradient descent algorithm for one repititions.
        Parameters
        ----------
        x: ndarray, dense as fuck rating matrix
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


if __name__ == '__main__':

    # user_matrix = np.load('../data/user_matrix.npy')
    # print(user_matrix[400])
    items_matrix = np.load('../data/item_matrix.npy')
    items_matrix_books = items_matrix[::, 0]
    items_matrix_factors = items_matrix[::, 1]

    # df_user_ratings = pd.read_csv('../data/goodbooks-10k/ratings.csv')
    # df_books_gr = pd.read_csv('../data/goodbooks-10k/books.csv')
    # d = df_books_gr.set_index('book_id')['best_book_id'].to_dict()
    #
    # df_user_ratings = df_user_ratings[df_user_ratings['user_id'] == 4010]
    # df_user_ratings['book_id'] = df_user_ratings['book_id'].map(lambda x: d.get(x, None))
    # dict_u_rate = df_user_ratings.set_index('book_id')['rating'].to_dict()
    # user_ratings = [dict_u_rate.get(book, None) for book in items_matrix_books]
    # x = np.array(user_ratings)
    # V = np.array([factors for factors in items_matrix_factors]).T

    # Created from Amazon Review file for ASIN and GoodReads API
    asin_best_file = '../data/asin_best_book_id.csv'
    df_isbn_best_book_id = load_data.get_isbn_to_best_book_id(asin_best_file)
    import os
    api_key = os.environ['GOODREADS_API_KEY']

    df_cristine = get_user.get_user_read_books(2624891, api_key)
    dict_isbn_best_id = df_isbn_best_book_id.set_index(['isbn'])['best_book_id'].to_dict()
    df_cristine['book_id'] = df_cristine['isbn'].map(lambda x: dict_isbn_best_id.get(x))
    df_cristine = df_cristine[df_cristine['book_id'].isnull() == False]
    # print(x.shape)

    # gd = GradientDescent(gradient, num_iterations=1)
    # gd.fit(x, V)

    gd = GradientDescent(alpha=.01, num_iterations=100)
    # gd.fit(df_user_ratings, items_matrix)
    gd.fit(df_cristine, items_matrix)
    print(gd.user_factors)
