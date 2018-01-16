import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class GradientDescent(object):
    """
    Perform the gradient descent optimization algorithm
    """

    def __init__(self,
                 alpha=0.01,
                 num_iterations=1000
                 ):
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
        self.u = None
        self.alpha = alpha
        self.num_iterations = num_iterations

    def fit(self, x, V):
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
        self.u = np.random.uniform(low=0.0, high=1.0, size=(1, V.shape[0]))
        for i in range(self.num_iterations):
            grad = self.gradient(x, V)
            self.u = self.u - self.alpha * grad
        return self

    def gradient(self, x, V):
        """Run the gradient descent algorithm for one repititions.
        Parameters
        ----------
        x: ndarray, sparse rating matrix
            The training data for the optimization.
        V: ndarray, items data
            The training response for the optimization.
        Returns
        -------
        self:
            The next gradient in U
        """
        grad = self.u
        for i, rating in enumerate(x):
            if rating:
                for j, item in enumerate(u[0]):
                    s = - 2 * V[j][i] * (rating - np.dot(self.u, V.T[i]))
                    grad[0][j] -= (s * self.alpha)
        return grad


if __name__ == '__main__':

    user_matrix = np.load('../data/user_matrix.npy')
    print(user_matrix[400])
    items_matrix = np.load('../data/item_matrix.npy')
    items_matrix_books = items_matrix[::, 0]
    items_matrix_factors = items_matrix[::, 1]

    df_user_ratings = pd.read_csv('../data/goodbooks-10k/ratings.csv')
    df_books_gr = pd.read_csv('../data/goodbooks-10k/books.csv')
    d = df_books_gr.set_index('book_id')['best_book_id'].to_dict()

    df_user_ratings = df_user_ratings[df_user_ratings['user_id'] == 4010]
    df_user_ratings['book_id'] = df_user_ratings['book_id'].map(lambda x: d.get(x, None))
    dict_u_rate = df_user_ratings.set_index('book_id')['rating'].to_dict()
    user_ratings = [dict_u_rate.get(book, None) for book in items_matrix_books]
    x = np.array(user_ratings)
    V = np.array([factors for factors in items_matrix_factors]).T


    gd = GradientDescent(num_iterations=1000)
    gd.fit(x, V)

    print(gd.u)
