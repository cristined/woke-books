import numpy as np
import pandas as pd
import random
# import matplotlib.pyplot as plt
import load_data
from gd_new_user import GD


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


def user_ratings_by_id(user_id, df_user_ratings, d_gb_best_id):
    """
    Get user's ratings from the recommenders training data using
    the goodreads user ID
    """
    df_uratings = df_user_ratings[df_user_ratings['user_id'] == user_id]
    df_uratings['book_id'] = df_uratings['book_id'].map(lambda x: d_gb_best_id.get(x, 0))
    return df_uratings


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


def grid_search(num_obs, num_iters, alphas, negatives):
    """
    Grid search number of iterations and alphas to find optimal model
    """
    min_err = float('inf')
    best_iters = None
    best_alpha = None
    best_model = None
    random_rows = np.random.choice(len(user_matrix), num_obs, replace=False)
    for negative in negatives:
        for iters in num_iters:
            for a in alphas:
                gd_ord_err, gd_err = test_rmse(random_rows, num_iterations=iters,
                                   alpha=a, negative=negative)
                print('num_iterations={}, alpha={}, negative={}'.format(iters,
                                                                        a,
                                                                        negative))
                print("GD Error - ", gd_err)
                print("GD Ord Error - ", gd_ord_err)
                print("--"*20)
                if gd_ord_err < min_err:
                    min_err = gd_ord_err
                    best_iters = iters
                    best_alpha = a
                    best_model = 'NGD'
                    if negative:
                        best_model = 'GD'
    print("Best Model: {}\nAlpha: {}\n# Iters: {}\nError: {}".format(best_model,
                                                                     best_alpha,
                                                                     best_iters,
                                                                     min_err))


def test_rmse(random_rows, num_iterations=100, alpha=0.01, negative=True):
    """
    For given number of observations get the error of the user matrix (u)
    for GD and NGD vs the actuals
    """
    gd_ord_errs = []
    gd_errs = []
    no_obs = len(random_rows)
    for rand_row in random_rows:
        actuals_row = np.array(user_matrix[rand_row][1])
        gd_row = test_gd(rand_row, num_iterations, alpha, negative)
        actual_sort = np.argsort(actuals_row)
        gd_sort = np.argsort(gd_row)
        gd_ord_errs.append((((np.array(actual_sort)) - np.array(gd_sort)) ** 2).sum() ** .5)
        gd_errs.append((((np.array(actuals_row)) - np.array(gd_row)) ** 2).sum() ** .5)
    return np.array(gd_ord_errs).sum()/no_obs, np.array(gd_ord_errs).sum()/no_obs


def test_gd(user_row, num_iterations=100, alpha=0.01, negative=True):
    """
    Get user matrix (u) for GD and NGD and the actuals
    """
    user_id = user_matrix[user_row][0]
    df_user = user_ratings_by_id(user_id, df_user_ratings, d_gb_best_id)
    gd = GD(num_iterations=num_iterations, alpha=alpha, negative=negative)
    gd.fit(df_user, items_matrix)
    return gd.user_factors[0]


if __name__ == '__main__':
    user_matrix, items_matrix, items_matrix_books, items_matrix_factors = get_recommender_data()
    df_user_ratings, df_books_gr, d_gb_best_id = get_books_data()

    V = np.array([factors for factors in items_matrix_factors]).T

    negatives = [True, False]
    num_iters = [100]
    alphas = [.01]
    num_obs = 2
    # num_iters = [100, 500, 1000]
    # alphas = [.01, .1]
    # num_obs = 250

    grid_search(num_obs, num_iters, alphas, negatives)
