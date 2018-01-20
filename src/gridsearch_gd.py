import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import load_data
from gd_new_user import GD
import time
import copy_reg
import types
import multiprocessing
import warnings
warnings.filterwarnings("ignore")


def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

class GridSearchGD(object):
    def __init__(self):
        self.rank = None
        self.df_user_ratings = None
        self.df_books_gr = None
        self.d_gb_best_id = None
        self.users_train_matrix = None
        self.items_train_matrix = None
        self.users_test_matrix = None
        self.items_test_matrix = None

    def fit(self, rank):
        self.rank = rank
        self._get_books_data()
        self._load_matrix()

    def _pickle_method(self, m):
        if m.im_self is None:
            return getattr, (m.im_class, m.im_func.func_name)
        else:
            return getattr, (m.im_self, m.im_func.func_name)

    def _get_books_data(self):
        """
        Get the books data we are testing on
        """
        self.df_user_ratings = pd.read_csv('../data/goodbooks-10k/val-ratings.csv')
        self.df_books_gr = pd.read_csv('../data/goodbooks-10k/books.csv')
        self.d_gb_best_id = self.df_books_gr.set_index('book_id')['best_book_id'].to_dict()

    def _load_matrix(self):
        user_train_npy = '../data/k-matrix/{}_train_user_matrix.npy'.format(self.rank)
        item_train_npy = '../data/k-matrix/{}_train_item_matrix.npy'.format(self.rank)
        user_test_npy = '../data/k-matrix/{}_test_user_matrix.npy'.format(self.rank)
        item_test_npy = '../data/k-matrix/{}_test_item_matrix.npy'.format(self.rank)
        self.users_train_matrix = np.load(user_train_npy)
        self.items_train_matrix = np.load(item_train_npy)
        self.users_test_matrix = np.load(user_test_npy)
        self.items_test_matrix = np.load(item_test_npy)

    def user_ratings_by_id(self, user_id):
        """
        Get user's ratings from the recommenders training data using
        the goodreads user ID
        """
        df_uratings = self.df_user_ratings[self.df_user_ratings['user_id'] == user_id]
        df_uratings['book_id'] = df_uratings['book_id'].map(lambda x: self.d_gb_best_id.get(x, 0))
        return df_uratings

    def grid_search(self, user_ids, num_iters, alphas, negatives, rank):
        """
        Grid search number of iterations and alphas to find optimal model
        """
        max_ratings_rank_sim = 0
        metrics = []
        best_iters = None
        best_alpha = None
        best_model = None
        for negative in negatives:
            for iters in num_iters:
                for a in alphas:
                    # ratings_rank_sim = self.test_ratings_rank_sim(user_ids, num_iterations=iters,
                    #                    alpha=a, negative=negative)

                    pool = multiprocessing.Pool(multiprocessing.cpu_count())
                    copy_reg.pickle(types.MethodType, _pickle_method)
                    test_gd_var = [(user_id, iters, a, negative) for user_id in user_ids]
                    ratings_rank_sim_list = pool.map(self.test_gd, test_gd_var)
                    pool.close()
                    pool.join()
                    ratings_rank_sim = np.array(ratings_rank_sim_list).mean()
                    print('num_iterations={}, alpha={}, negative={}'.format(iters,
                                                                            a,
                                                                            negative))
                    print("Rank Sim - ", ratings_rank_sim)
                    print("--"*20)
                    metrics.append([self.rank, negative, iters, a, ratings_rank_sim])
                    if max_ratings_rank_sim < ratings_rank_sim:
                        max_ratings_rank_sim = ratings_rank_sim
                        best_iters = iters
                        best_alpha = a
                        best_model = 'NGD'
                        if negative:
                            best_model = 'GD'
        print("Best Model: {}\nAlpha: {}\n# Iters: {}\nMax Sim: {}".format(best_model,
                                                                         best_alpha,
                                                                         best_iters,
                                                                         max_ratings_rank_sim))
        np.save(str(self.rank) + '_metrics', np.array(metrics))
        return best_model, best_alpha, best_iters, max_ratings_rank_sim


    def test_ratings_rank_sim(self, user_ids, num_iterations=100, alpha=0.01, negative=True):
        """
        For given number of observations get the error of the user matrix (u)
        for GD and NGD vs the actuals
        """
        # gd_ord_errs = []
        # gd_errs = []
        # gd_recon_errs = []
        # no_obs = len(user_ids)
        #lambda function with matrix and param coded in to test_gd function
        # test_gd_pool = lambda x: self.test_gd(x, num_iterations, alpha, negative)
        # call pool with lambda function - will return a list
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        copy_reg.pickle(types.MethodType, _pickle_method)
        test_gd_var = [(user_id, True, 100, .01) for user_id in user_ids]
        ratings_rank_sim_list = pool.map(self.test_gd, test_gd_var)
        pool.close()
        pool.join()
        # for user_id in d_list:
        #     user_row = np.where(users_test_matrix[::,0] == user_id)[0][0]
        #     actuals_row = np.array(users_test_matrix[user_row][1])
        #     actual_sort = np.argsort(actuals_row)
        #     gd_sort = np.argsort(gd_row)
        #     gd_ord_errs.append(((((np.array(actual_sort)) - np.array(gd_sort)) ** 2).sum() / len(actual_sort))** .5)
        #     gd_errs.append(((((np.array(actuals_row)) - np.array(gd_row)) ** 2).sum() / len(actuals_row)) ** .5)
        #     gd_recon_errs.append(gd_recon_err)
        return np.array(ratings_rank_sim_list).mean()

    def test_gd(self, test_gd_var):
        """
        Get user matrix (u) for GD and NGD and the actuals
        """
        user_id, iters, a, negative = test_gd_var
        df_user = self.user_ratings_by_id(user_id)
        gd = GD(num_iterations=iters, alpha=a, negative=negative)
        gd.fit(df_user, self.items_train_matrix)
        gd_row = gd.user_factors[0]
        user_row = np.where(self.users_test_matrix[::, 0] == user_id)[0][0]
        actuals_row = np.array(self.users_test_matrix[user_row][1])
        rank_diff = self.get_results_vect(actuals_row, gd_row)
        return rank_diff

    def get_results_vect(self, actuals_row, gd_row):
        actuals_row = actuals_row.reshape((1, -1))
        gd_row = gd_row.reshape((1, -1))
        items_train_matrix = np.array([i for i in self.items_train_matrix[::,1]]).T
        items_test_matrix = np.array([i for i in self.items_test_matrix[::,1]]).T
        actual_results = np.dot(actuals_row, items_train_matrix)
        gd_results = np.dot(gd_row, items_test_matrix)
        ratings_rank_sim = self.ratings_rank_similarity(actual_results, gd_results)
        return ratings_rank_sim

    def ratings_rank_similarity(self, actuals_row, gd_row):
        actual_sort = np.argsort(actuals_row).reshape((-1))[::-1]
        gd_sort = np.argsort(gd_row).reshape((-1))[::-1]
        score = []
        for i in range(len(actual_sort)):
            sim = len(set(actual_sort[:i+1]).intersection(set(gd_sort[:i+1]))) / (i +1)
            score.append(sim)
        return np.array(score).mean()


if __name__ == '__main__':
    # df_user_ratings, df_books_gr, d_gb_best_id = get_books_data()

    negatives = [True, False]
    num_iters = [100, 500, 1000]
    alphas = [.01, .1]

    # user_ids = np.random.choice(df_user_ratings['user_id'].unique(), 2, replace=False)


    print(11)
    # we need to compare ranking between the actual user recommendations dot with the item factors
        # and the ones we made up with the test item set
    # rank_list = range(11, 42)[::2]
    rank = 11
    print("=="*20)
    print(rank)
    start = time.time()
    grid_gd = GridSearchGD()
    grid_gd.fit(rank)
    np.random.seed(0)
    user_ids = np.random.choice(grid_gd.df_user_ratings['user_id'].unique(), 2, replace=False)
    grid_gd.grid_search(user_ids, num_iters, alphas, negatives, rank)
    end = time.time()
    print(end - start)
