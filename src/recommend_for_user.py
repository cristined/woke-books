import numpy as np
import pandas as pd
import load_data
import get_user
from new_user_gradient_descent import GradientDescent


def get_gd_ratings(matrix_u_rate, book_factors):
    gd = GradientDescent(num_iterations=1000)
    gd.fit(matrix_u_rate, book_factors)
    recommendations = np.dot(gd.u, book_factors)
    book_recs_arr = np.dstack((items_matrix_books.reshape((-1)), recommendations.reshape((-1))))[0]
    df_book_rec = pd.DataFrame(book_recs_arr, columns=['best_book_id','rating_guess'])
    return df_book_rec


def get_recommendations(df_book_rec, df_user_ratings, df_books_classified):
    df_books_rec_ratings = pd.merge(df_book_rec,
                                    df_user_ratings[['best_book_id', 'rating']],
                                    left_on=['best_book_id'],
                                    right_on=['best_book_id'],
                                    how='left')
    df_books_unread = df_books_rec_ratings[df_books_rec_ratings.rating.isnull()]
    df_books_unread_classified = pd.merge(df_books_unread, df_books_classified, left_on='best_book_id', right_on='best_book_id', how='inner')
    df_books_unread_classified = df_books_unread_classified.sort_values('rating_guess', ascending=False)[df_books_unread_classified['race'] != 'WHITE']
    df_books_unread_classified
    return df_books_unread_classified


if __name__ == '__main__':
    # Get Recommender Data
    items_matrix = np.load('../data/item_matrix.npy')
    items_matrix_books = items_matrix[::, 0]
    items_matrix_factors = items_matrix[::, 1]
    book_factors = np.array([factors for factors in items_matrix_factors]).T

    # Get User Data
    api_key = os.environ['GOODREADS_API_KEY']
    df_user_ratings = get_user.get_user_read_books(2624891, api_key)
    matrix_u_rate = get_user.user_ratings_for_recommender(df_user_ratings,
                                                          df_isbn_best_book_id,
                                                          items_matrix_books)

    # Run Recommendations
    df_book_rec = get_gd_ratings(matrix_u_rate, book_factors)

    df_books_unread_classified = get_recommendations(df_book_rec,
                                                     df_user_ratings,
                                                     df_books_classified)
