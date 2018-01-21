import pandas as pd
import numpy as np
import os
import load_data
import get_user
from gd_new_user import GD
# from tabulate import tabulate

class UserRecs(object):
    def __init__(self):
        self._get_books_data()

    def fit(self, user_id, api_key, rank, negative=True):
        self.rank = rank
        self.user_id = user_id
        self._get_items_matrix()
        self.get_user_data(user_id, api_key)
        self.get_gd_user(negative)
        self.get_recommendations()

    def _get_books_data(self):
        # Created from GoodReads API, should be the top 10K rated books
        book_file = '../data/updated_books.csv'
        # From Goodbooks data
        gr_book_file = '../data/goodbooks-10k/books.csv'
        # Created from GoodReads API, and manual classification
        author_file = '../data/classified_authors.csv'
        # Created from GoodReads API
        author_book_file = '../data/author_books.csv'
        # Created from Amazon Review file for ASIN and GoodReads API
        asin_best_file = '../data/asin_best_book_id.csv'
        self.df_books = load_data.get_books(book_file)
        self.df_gr_books = pd.read_csv(gr_book_file)
        self.df_authors = load_data.get_classified_authors(author_file)
        self.df_authors_books = load_data.get_books_to_authors(author_book_file)
        self.df_isbn_best_book_id = load_data.get_isbn_to_best_book_id(asin_best_file)
        df_books_classified = load_data.merge_to_classify_books(self.df_authors_books,
                                                                self.df_authors,
                                                                self.df_books)
        df_books_classified['authorbook_id'] = df_books_classified['best_book_id'].map(str) + ' ' + df_books_classified['author_id'].map(str)
        self.df_books_classified = df_books_classified
        df_ab_classified = df_books_classified.groupby(['race','gender'])['authorbook_id'].nunique().reset_index()
        df_ab_classified['percentage'] = df_ab_classified['authorbook_id'] / df_ab_classified['authorbook_id'].sum()
        df_ab_classified['race_gender'] = df_ab_classified['race'] + ' ' + df_ab_classified['gender']
        self.df_ab_classified = df_ab_classified

    def _get_items_matrix(self):
        item_test_npy = '../data/k-matrix/{}_test_item_matrix.npy'.format(self.rank)
        self.items_matrix = np.load(item_test_npy)
        self.items_matrix_books = self.items_matrix[::, 0]
        self.items_matrix_factors = self.items_matrix[::, 1]

    def get_user_data(self, user_id, api_key):
        self.df_user_ratings, self.books_read_10k, self.books_read = get_user.get_user_read_books(user_id, api_key, self.df_isbn_best_book_id, self.df_books)
        df_user_ab_classified = get_user.create_user_authorbook_classified(self.df_isbn_best_book_id,
                                                                           self.df_user_ratings,
                                                                           self.df_books_classified)
        df_user_ab_classified['race_gender'] = df_user_ab_classified['race'].map(str) + ' ' + df_user_ab_classified['gender'].map(str)
        self.df_user_ab_classified = df_user_ab_classified
        df_user_v_goodreads = pd.merge(self.df_ab_classified, df_user_ab_classified, left_on='race_gender', right_on='race_gender', how='left')
        df_user_v_goodreads = df_user_v_goodreads[['race_gender','race_x','gender_x','authorbook_id_x','percentage_x','authorbook_id_y','percentage_y']]
        df_user_v_goodreads.columns = ['race_gender', 'race', 'gender', 'gr_count',
                                       'gr_percentage', 'user_count',
                                       'user_percentage']
        df_user_v_goodreads['gr_count'] = df_user_v_goodreads['gr_count'] + 1000
        df_user_v_goodreads['gr_percentage'] = df_user_v_goodreads['gr_count'] / df_user_v_goodreads['gr_count'].sum()
        df_user_v_goodreads['user_count'].fillna(0, inplace=True)
        df_user_v_goodreads['user_percentage'].fillna(0.00001, inplace=True)
        df_user_v_goodreads['user_gr_perc'] = df_user_v_goodreads['user_percentage'] / df_user_v_goodreads['gr_percentage']
        df_user_v_goodreads['user_gr_perc_norm'] = 1 / (1 + df_user_v_goodreads['user_gr_perc'])
        self.df_user_v_goodreads = df_user_v_goodreads

    def plot_user_data(self):
        print("{} out of {} books that you have read are in the top 10,000 books on goodreads".format(self.books_read_10k, self.books_read))
        get_user.plot_user_authorbook_classified(self.df_user_ab_classified)

    def get_gd_user(self, negative=False):
        gd = GD(num_iterations=100, alpha=0.01, negative=negative)
        gd.fit(self.df_user_ratings, self.items_matrix)
        self.gd = gd

    def get_recommendations(self):
        book_factors = np.array([factors for factors in self.items_matrix_factors]).T
        recommendations = np.dot(self.gd.user_factors, book_factors)
        book_recs_arr = np.dstack((self.items_matrix_books.reshape((-1)), recommendations.reshape((-1))))[0]
        df_book_rec = pd.DataFrame(book_recs_arr, columns=['best_book_id','rating_guess'])
        df_books_rec_ratings = pd.merge(df_book_rec, self.df_user_ratings[['book_id','rating']], left_on=['best_book_id'], right_on=['book_id'], how='left')
        df_books_unread = df_books_rec_ratings[df_books_rec_ratings.rating.isnull()]
        df_books_unread_classified = pd.merge(df_books_unread, self.df_books_classified, left_on='best_book_id', right_on='best_book_id', how='inner')
        dict_user_goodreads_boost = self.df_user_v_goodreads.set_index('race_gender')['user_gr_perc_norm'].to_dict()
        df_books_unread_classified['race_gender'] = df_books_unread_classified['race'] + ' ' + df_books_unread_classified['gender']
        df_books_unread_classified['rating_guess'] = 5 * df_books_unread_classified['rating_guess'] / df_books_unread_classified['rating_guess'].max()
        df_books_unread_classified['boost'] = df_books_unread_classified['race_gender'].map(lambda x: dict_user_goodreads_boost.get(x, 0))
        df_books_unread_classified['boosted_ratings'] = df_books_unread_classified['boost'] + df_books_unread_classified['rating_guess']
        self.df_recommendations = df_books_unread_classified.sort_values('boosted_ratings', ascending=False)
        self.get_final_rec_df()

    def get_final_rec_df(self):
        rec_ind = self.df_recommendations[['best_book_id']].reset_index(drop=True
                                                                        ).reset_index()
        rec_ind = rec_ind.groupby(['best_book_id']).min().reset_index()
        rec_ind = pd.merge(rec_ind, self.df_gr_books, how='left',
                           left_on='best_book_id', right_on='best_book_id'
                           ).sort_values('index')
        self.book_recs = rec_ind


def pretty_print(df, length):
    # df = df[['title','name_x','race_gender','rating_guess','boosted_ratings']].head(length)
    df = df[['title', 'race_gender']].head(length)
    print(str(df))


if __name__ == '__main__':

    api_key = os.environ['GOODREADS_API_KEY']

    Catherine = 53106890
    Cristine = 2624891
    Tomas = 5877959
    Moses = 8683925
    Rohit = 76691842

    for rank in [11, 13]:
        Cristine_Recs = UserRecs()
        Cristine_Recs.fit(Cristine, api_key, rank)
        print("Rank of {} for {}".format(rank, 'Catherine'))
        print(pretty_print(Cristine_Recs.df_recommendations, 10))
        Cristine_Recs.plot_user_data()
