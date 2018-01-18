import numpy as np
import pandas as pd
import random
# import matplotlib.pyplot as plt
import load_data
from new_user_gradient_descent import GradientDescent
from new_user_non_neg_grad_desc import NGD


def get_recommender_data():
    """
    Get the data from the recommender we are testing on
    """
    user_matrix = np.load('../data/user_matrix.npy')
    items_matrix = np.load('../data/item_matrix.npy')
    items_matrix_books = items_matrix[::, 0]
    items_matrix_factors = items_matrix[::, 1]
    return user_matrix, items_matrix, items_matrix_books, items_matrix_factors


def get_books_data():
    """
    Get the books data we are testing on
    """
    df_user_ratings = pd.read_csv('../data/goodbooks-10k/ratings.csv')
    df_books_gr = pd.read_csv('../data/goodbooks-10k/books.csv')
    d_gb_best_id = df_books_gr.set_index('book_id')['best_book_id'].to_dict()
    return df_user_ratings, df_books_gr, d_gb_best_id


def user_ratings_by_id(user_id, df_user_ratings, d_gb_best_id, items_matrix_books):
    """
    Get user's ratings from the recommenders training data using
    the goodreads user ID
    """
    df_uratings = df_user_ratings[df_user_ratings['user_id'] == user_id]
    df_uratings['best_book_id'] = df_uratings['book_id'].map(lambda x: d_gb_best_id.get(x, 0))
    dict_u_rate = df_uratings.set_index('best_book_id')['rating'].to_dict()
    user_ratings = [dict_u_rate.get(book, 0) for book in items_matrix_books]
    return user_ratings


def test_gd_ngd_actuals(user_row, num_iterations=100, alpha=0.01):
    """
    Get user matrix (u) for GD and NGD and the actuals
    """
    user_id = user_matrix[user_row][0]
    x = np.array(user_ratings_by_id(user_id, df_user_ratings, d_gb_best_id, items_matrix_books))
    gd = GradientDescent(num_iterations=num_iterations, alpha=alpha)
    gd.fit(x, V)
    ngd = NGD(num_iterations=num_iterations, alpha=alpha)
    ngd.fit(x.reshape(1, -1), V)
    # Return actuals, gd, ngd
    return np.array(user_matrix[user_row][1]), gd.u[0], ngd.u[0]


def plot_gd_ngd_actuals(user_row):
    """
    Plot user matrix (u) for GD and NGD and the actuals
    """
    user_id = user_matrix[user_row][0]
    actuals_u, gd_u, ngd_u = test_gd_ngd_actuals(user_row)
    plt.plot(gd_u)
    plt.plot(ngd_u)
    plt.plot(actuals_u)
    plt.title("User ID: {}, User Row: {}".format(user_id, user_row))
    plt.legend(['GD', 'NGD', 'Actuals'])
    plt.show()
    print("Actuals: ", actuals_u)
    print("GD: ", gd_u)
    print("GD RMSE: ", ((actuals_u - gd_u) ** 2).sum() ** .5)
    print("NGD: ", ngd_u)
    print("NGD RMSE: ", ((actuals_u - ngd_u) ** 2).sum() ** .5)


def test_rmse(random_rows, num_iterations=100, alpha=0.01):
    """
    For given number of observations get the error of the user matrix (u)
    for GD and NGD vs the actuals
    """
    actuals_u = []
    gd_u = []
    ngd_u = []
    no_obs = len(random_rows)
    for rand_row in random_rows:
        actuals_row, gd_row, ngd_row = test_gd_ngd_actuals(rand_row, num_iterations, alpha)
        actuals_u.append(actuals_row)
        gd_u.append(gd_row)
        ngd_u.append(ngd_row)
    gd_err = ((np.argsort(np.array(actuals_u)) - np.argsort(np.array(gd_u))) ** 2).sum() ** .5
    ngd_err = ((np.argsort(np.array(actuals_u)) - np.argsort(np.array(ngd_u))) ** 2).sum() ** .5
    return gd_err/no_obs, ngd_err/no_obs


def grid_search(num_obs, num_iters, alphas):
    """
    Grid search number of iterations and alphas to find optimal model
    """
    min_err = float('inf')
    best_iters = None
    best_alpha = None
    best_model = None
    random_rows = np.random.choice(len(user_matrix), num_obs, replace=False)
    for iters in num_iters:
        for a in alphas:
            gd_err, ngd_err = test_rmse(random_rows, num_iterations=iters, alpha=a)
            print('num_iterations={}, alpha={}'.format(iters, a))
            print("GD Error - ", gd_err)
            print("NGD Error - ", ngd_err)
            print("--"*20)
            if gd_err < min_err:
                min_err = gd_err
                best_iters = iters
                best_alpha = a
                best_model = 'GD'
            if ngd_err < min_err:
                min_err = ngd_err
                best_iters = iters
                best_alpha = a
                best_model = 'NGD'
    print("Best Model: {}\nAlpha: {}\n# Iters: {}\nError: {}".format(best_model,
                                                                     best_alpha,
                                                                     best_iters,
                                                                     min_err))


if __name__ == '__main__':
    user_matrix, items_matrix, items_matrix_books, items_matrix_factors = get_recommender_data()
    df_user_ratings, df_books_gr, d_gb_best_id = get_books_data()

    V = np.array([factors for factors in items_matrix_factors]).T

    num_iters = [100, 500, 1000]
    alphas = [.01, .1]
    num_obs = 250

    grid_search(num_obs, num_iters, alphas)
