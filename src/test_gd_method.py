import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import load_data
from gd_new_user import GD
import time
import multiprocessing




def get_books_data():
    """
    Get the books data we are testing on
    """
    df_user_ratings = pd.read_csv('../data/goodbooks-10k/val-ratings.csv')
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


# def plot_gd_ngd_actuals(user_row):
#     """
#     Plot user matrix (u) for GD and NGD and the actuals
#     """
#     user_id = user_matrix[user_row][0]
#     actuals_u, gd_u, ngd_u = test_gd_ngd_actuals(user_row)
#     plt.plot(gd_u)
#     plt.plot(ngd_u)
#     plt.plot(actuals_u)
#     plt.title("User ID: {}, User Row: {}".format(user_id, user_row))
#     plt.legend(['GD', 'NGD', 'Actuals'])
#     plt.show()
#     print("Actuals: ", actuals_u)
#     print("GD: ", gd_u)
#     print("GD RMSE: ", ((actuals_u - gd_u) ** 2).sum() ** .5)
#     print("NGD: ", ngd_u)
#     print("NGD RMSE: ", ((actuals_u - ngd_u) ** 2).sum() ** .5)


def load_matrix(rank):
    user_train_npy = '../data/k-matrix/{}_train_user_matrix.npy'.format(rank)
    item_train_npy = '../data/k-matrix/{}_train_item_matrix.npy'.format(rank)
    user_test_npy = '../data/k-matrix/{}_test_user_matrix.npy'.format(rank)
    item_test_npy = '../data/k-matrix/{}_test_item_matrix.npy'.format(rank)
    users_train_matrix = np.load(user_train_npy)
    items_train_matrix = np.load(item_train_npy)
    users_test_matrix = np.load(user_test_npy)
    items_test_matrix = np.load(item_test_npy)
    return users_train_matrix, items_train_matrix, users_test_matrix, items_test_matrix


def grid_search(user_ids, num_iters, alphas, negatives, rank):
    """
    Grid search number of iterations and alphas to find optimal model
    """
    min_err = float('inf')
    min_ord_err = float('inf')
    min_recon_err = float('inf')
    metrics = []
    best_iters = None
    best_alpha = None
    best_model = None
    users_train_matrix, items_train_matrix, users_test_matrix, items_test_matrix = load_matrix(rank)
    for negative in negatives:
        for iters in num_iters:
            for a in alphas:
                rank_sim = test_rank_sim(user_ids, users_train_matrix, items_train_matrix, users_test_matrix, items_test_matrix, num_iterations=iters,
                                   alpha=a, negative=negative)
                print('num_iterations={}, alpha={}, negative={}'.format(iters,
                                                                        a,
                                                                        negative))
                print("GD Error - ", gd_err)
                print("GD Ord Error - ", gd_ord_err)
                print("GD Recon Error - ", gd_recon_err)
                print("--"*20)
                metrics.append([rank, negative, iters, a, gd_ord_err, gd_err, gd_recon_err])
                if gd_recon_err < min_recon_err:
                    min_err = gd_err
                    min_ord_err = gd_ord_err
                    min_recon_err = gd_recon_err
                    best_iters = iters
                    best_alpha = a
                    best_model = 'NGD'
                    if negative:
                        best_model = 'GD'
    print("Best Model: {}\nAlpha: {}\n# Iters: {}\nError: {}".format(best_model,
                                                                     best_alpha,
                                                                     best_iters,
                                                                     min_recon_err))
    np.save(str(rank) + '_metrics', np.array(metrics))
    return best_model, best_alpha, best_iters, min_err


def test_rank_sim(user_ids, users_train_matrix, items_train_matrix, users_test_matrix, items_test_matrix, num_iterations=100, alpha=0.01, negative=True):
    """
    For given number of observations get the error of the user matrix (u)
    for GD and NGD vs the actuals
    """
    gd_ord_errs = []
    gd_errs = []
    gd_recon_errs = []
    no_obs = len(user_ids)
    #lambda function with matrix and param coded in to test_gd function
    # test_gd_pool = lambda x: test_gd(x, users_train_matrix, items_train_matrix, users_test_matrix, items_test_matrix, num_iterations, alpha, negative)
    # # call pool with lambda function - will return a list
    # pool = multiprocessing.Pool(4) #multiprocessing.cpu_count()
    # rank_sim_list = pool.map(test_gd_pool, user_ids)
    # pool.close()
    # pool.join()
    for user_id in user_ids:
        rank_dff = test_gd(user_id, users_train_matrix, items_train_matrix, users_test_matrix, items_test_matrix, num_iterations, alpha, negative)
        rank_diff_list.append(rank_dff)
    #     user_row = np.where(users_test_matrix[::,0] == user_id)[0][0]
    #     actuals_row = np.array(users_test_matrix[user_row][1])
    #     actual_sort = np.argsort(actuals_row)
    #     gd_sort = np.argsort(gd_row)
    #     gd_ord_errs.append(((((np.array(actual_sort)) - np.array(gd_sort)) ** 2).sum() / len(actual_sort))** .5)
    #     gd_errs.append(((((np.array(actuals_row)) - np.array(gd_row)) ** 2).sum() / len(actuals_row)) ** .5)
    #     gd_recon_errs.append(gd_recon_err)
    return np.array(rank_diff_list).mean()


def test_gd(user_id, users_train_matrix, items_train_matrix, users_test_matrix, items_test_matrix, num_iterations=100, alpha=0.01, negative=True):
    """
    Get user matrix (u) for GD and NGD and the actuals
    """
    df_user = user_ratings_by_id(user_id, df_user_ratings, d_gb_best_id)
    gd = GD(num_iterations=num_iterations, alpha=alpha, negative=negative)
    gd.fit(df_user, items_train_matrix)
    gd_row = gd.user_factors[0]
    user_row = np.where(users_test_matrix[::,0] == user_id)[0][0]
    actuals_row = np.array(users_test_matrix[user_row][1])
    rank_diff = get_results_vect(actuals_row, gd_row, items_train_matrix, items_test_matrix)
    return rank_diff


def get_results_vect(actuals_row, gd_row, items_train_matrix, items_test_matrix):
    actuals_row = actuals_row.reshape((1, -1))
    gd_row = gd_row.reshape((1, -1))
    items_train_matrix = np.array([i for i in items_train_matrix[::,1]]).T
    items_test_matrix = np.array([i for i in items_test_matrix[::,1]]).T
    print(actuals_row.shape)
    print(gd_row.shape)
    print(items_train_matrix.shape)
    print(items_test_matrix.shape)
    actual_results = np.dot(actuals_row, items_train_matrix)
    gd_results = np.dot(gd_row, items_test_matrix)
    rank_sim = rank_similarity(actual_results, gd_results)
    return rank_sim


def rank_similarity(actuals_row, gd_row):
    actual_sort = np.argsort(actuals_row).reshape((-1))
    gd_sort = np.argsort(gd_row).reshape((-1))
    score = []
    for i in range(len(actual_sort)):
        sim = len(set(actual_sort[:i+1]).intersection(set(gd_sort[:i+1])))
        score.append(sim / (i +1))
    print(np.array(score).mean())
    return np.array(score).mean()


if __name__ == '__main__':
    df_user_ratings, df_books_gr, d_gb_best_id = get_books_data()

    negatives = [True, False]
    num_iters = [100, 500, 1000]
    alphas = [.01, .1]

    np.random.seed(0)
    user_ids = np.random.choice(df_user_ratings['user_id'].unique(), 2, replace=False)


    print(11)
    # we need to compare ranking between the actual user recommendations dot with the item factors
        # and the ones we made up with the test item set
    # rank_list = range(11, 42)[::2]
    rank = 11
    start = time.time()
    grid_search(user_ids, num_iters, alphas, negatives, rank)
    end = time.time()
    print(end - start)
