import pandas as pd
import numpy as np
import os
import load_data
import get_user
from new_user_gradient_descent import GradientDescent
from new_user_non_neg_grad_desc import NGD


def get_books_data():
    # Created from GoodReads API, should be the top 10K rated books
    book_file = '../data/updated_books.csv'
    # Created from GoodReads API, and manual classification
    author_file = '../data/classified_authors.csv'
    # Created from GoodReads API
    author_book_file = '../data/author_books.csv'
    # Created from Amazon Review file for ASIN and GoodReads API
    asin_best_file = '../data/asin_best_book_id.csv'
    df_books = load_data.get_books(book_file)
    df_authors = load_data.get_classified_authors(author_file)
    df_authors_books = load_data.get_books_to_authors(author_book_file)
    df_isbn_best_book_id = load_data.get_isbn_to_best_book_id(asin_best_file)
    df_books_classified = load_data.merge_to_classify_books(df_authors_books,
                                                            df_authors,
                                                            df_books)
    df_books_classified['authorbook_id'] = df_books_classified['best_book_id'].map(str) + ' ' + df_books_classified['author_id'].map(str)
    df_ab_classified = df_books_classified.groupby(['race','gender'])['authorbook_id'].nunique().reset_index()
    df_ab_classified['percentage'] = df_ab_classified['authorbook_id'] / df_ab_classified['authorbook_id'].sum()
    df_ab_classified['race_gender'] = df_ab_classified['race'] + ' ' + df_ab_classified['gender']
    return df_books, df_authors, df_authors_books, df_isbn_best_book_id, df_books_classified, df_ab_classified


def get_items_matrix():
    items_matrix = np.load('../data/item_matrix.npy')
    items_matrix_books = items_matrix[::, 0]
    items_matrix_factors = items_matrix[::, 1]
    return items_matrix_books, items_matrix_factors


def get_user_data(user_id, api_key, df_isbn_best_book_id, df_books_classified, df_ab_classified, items_matrix_books):
    df_user_ratings = get_user.get_user_read_books(user_id, api_key, df_isbn_best_book_id)
    df_user_ab_classified = get_user.create_user_authorbook_classified(df_isbn_best_book_id,
                                                                       df_user_ratings,
                                                                       df_books_classified)
    df_user_ab_classified['race_gender'] = df_user_ab_classified['race'].map(str) + ' ' + df_user_ab_classified['gender'].map(str)
    df_user_v_goodreads = pd.merge(df_ab_classified, df_user_ab_classified, left_on='race_gender', right_on='race_gender', how='left')
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
    # matrix_u_rate = get_user.user_ratings_for_recommender(df_user_ratings,
    #                                                       df_isbn_best_book_id,
    #                                                       items_matrix_books)
    matrix_u_rate = None
    return df_user_ratings, df_user_v_goodreads, np.array(matrix_u_rate)


def get_ngd_user(items_matrix_factors, matrix_u_rate):
    book_factors = np.array([factors for factors in items_matrix_factors]).T
    ngd = NGD(num_iterations=100, alpha=0.01)
    ngd.fit(matrix_u_rate, book_factors)
    return ngd.u


def get_gd_user(df_user_ratings, items_matrix):
    # book_factors = np.array([factors for factors in items_matrix_factors]).T
    gd = GradientDescent(num_iterations=100, alpha=0.01)
    gd.fit(df_user_ratings, items_matrix)
    return gd.user_factors


def get_recommendations(ngd_user, items_matrix_books, items_matrix_factors, df_user_ratings, df_books_classified, df_user_v_goodreads):
    book_factors = np.array([factors for factors in items_matrix_factors]).T
    recommendations = np.dot(ngd_user, book_factors)
    book_recs_arr = np.dstack((items_matrix_books.reshape((-1)), recommendations.reshape((-1))))[0]
    df_book_rec = pd.DataFrame(book_recs_arr, columns=['best_book_id','rating_guess'])
    df_books_rec_ratings = pd.merge(df_book_rec, df_user_ratings[['book_id','rating']], left_on=['best_book_id'], right_on=['book_id'], how='left')
    df_books_unread = df_books_rec_ratings[df_books_rec_ratings.rating.isnull()]
    df_books_unread_classified = pd.merge(df_books_unread, df_books_classified, left_on='best_book_id', right_on='best_book_id', how='inner')
    dict_user_goodreads_boost = df_user_v_goodreads.set_index('race_gender')['user_gr_perc_norm'].to_dict()
    df_books_unread_classified['race_gender'] = df_books_unread_classified['race'] + ' ' + df_books_unread_classified['gender']
    df_books_unread_classified['boost'] = df_books_unread_classified['race_gender'].map(lambda x: dict_user_goodreads_boost.get(x, 0))
    df_books_unread_classified['boosted_ratings'] = df_books_unread_classified['boost'] + df_books_unread_classified['rating_guess']
    return df_books_unread_classified.sort_values('boosted_ratings', ascending=False)


if __name__ == '__main__':
    df_books, df_authors, df_authors_books, df_isbn_best_book_id, df_books_classified, df_ab_classified = get_books_data()
    items_matrix_books, items_matrix_factors = get_items_matrix()


    api_key = os.environ['GOODREADS_API_KEY']

    df_user_ratings, df_user_v_goodreads, matrix_u_rate = get_user_data(2624891, api_key, df_isbn_best_book_id, df_books_classified, df_ab_classified, items_matrix_books, df_user_v_goodreads)
    ngd_user = get_ngd_user(items_matrix_factors, matrix_u_rate)
    print(ngd_user)
    print(ngd_user.shape)
    df_user = get_recommendations(ngd_user, items_matrix_books, items_matrix_factors, df_user_ratings, df_books_classified)
    print(df_user[['title','rating_guess','boosted_ratings']].head(10))
    gd_user = get_gd_user(items_matrix_factors, matrix_u_rate)
    print(gd_user)
    df_user = get_recommendations(gd_user, items_matrix_books, items_matrix_factors, df_user_ratings, df_books_classified)
    print(df_user[['title','rating_guess','boosted_ratings']].head(10))

    df_user_ratings, df_user_v_goodreads, matrix_u_rate = get_user_data(34338862, api_key, df_isbn_best_book_id, df_books_classified, items_matrix_books)
    ngd_user = get_ngd_user(items_matrix_factors, matrix_u_rate)
    print(ngd_user)
    df_user = get_recommendations(ngd_user, items_matrix_books, items_matrix_factors, df_user_ratings, df_books_classified)
    print(df_user[['title','rating_guess','boosted_ratings']].head(10))
    gd_user = get_gd_user(items_matrix_factors, matrix_u_rate)
    print(gd_user)
    df_user = get_recommendations(gd_user, items_matrix_books, items_matrix_factors, df_user_ratings, df_books_classified)
    print(df_user[['title','rating_guess','boosted_ratings']].head(10))

    df_user_ratings, df_user_v_goodreads, matrix_u_rate = get_user_data(1, api_key, df_isbn_best_book_id, df_books_classified, items_matrix_books)
    ngd_user = get_ngd_user(items_matrix_factors, matrix_u_rate)
    print(ngd_user)
    df_user = get_recommendations(ngd_user, items_matrix_books, items_matrix_factors, df_user_ratings, df_books_classified)
    print(df_user[['title','rating_guess','boosted_ratings']].head(10))
    gd_user = get_gd_user(items_matrix_factors, matrix_u_rate)
    print(gd_user)
    df_user = get_recommendations(gd_user, items_matrix_books, items_matrix_factors, df_user_ratings, df_books_classified)
    print(df_user[['title','rating_guess','boosted_ratings']].head(10))
