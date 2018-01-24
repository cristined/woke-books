import pandas as pd
from collections import Counter
from sqlalchemy import create_engine


def get_books():
    """
    INPUT: None
    OUTPUT: DataFrame with the following columns:
    'book_id', 'title', 'isbn', 'isbn13', 'country_code', 'language_code',
    'description', 'work_id', 'best_book_id', 'original_title',
    'ratings_count', 'work_ratings_count', 'work_text_reviews_count',
    'ratings_1', 'ratings_2', 'ratings_3', 'ratings_4', 'ratings_5',
    'image_url', 'small_image_url', 'k_label'
    """
    query = """
            SELECT books.*, k_label
            FROM books
            LEFT JOIN
            kmeans
            ON books.best_book_id = kmeans.book_id;
            """
    engine = create_engine('postgresql://postgres@localhost/books')
    return pd.read_sql_query(query, engine)


def get_classified_authors():
    """
    INPUT: None
    OUTPUT: DataFrame with the following columns:
    'author_id', 'name', 'main_author', 'race', 'gender', 'image_url',
    'about', 'influences', 'works_count', 'hometown', 'born_at', 'died_at'
    """
    df_authors = load_table('authors')
    return df_authors


def get_books_to_authors():
    """
    INPUT: None
    OUTPUT: DataFrame with the following columns:
    'book_id', 'author_id', 'name', 'role'
    """
    df_authors_books = load_table('book_authors')
    return df_authors_books


def get_isbn_to_best_book_id():
    """
    INPUT: None
    OUTPUT: DataFrame with the following columns:
    'isbn', 'best_book_id'
    """
    df_isbn_best_book_id = load_table('isbn_book_id')
    return df_isbn_best_book_id


def merge_to_classify_books():
    """
    INPUT: None
    OUTPUT:
    DataFrame with the following columns:
    'book_id', 'title', 'isbn', 'isbn13', 'country_code', 'language_code',
    'description', 'work_id', 'best_book_id', 'original_title',
    'ratings_count', 'work_ratings_count', 'work_text_reviews_count',
    'ratings_1', 'ratings_2', 'ratings_3', 'ratings_4', 'ratings_5',
    'image_url', 'small_image_url', 'author_id', 'name', 'main_author',
    'race', 'gender', 'about', 'influences', 'works_count', 'hometown',
    'born_at', 'died_at', 'k_label'
    """
    query = """
            WITH authors_books_classified AS
                 (SELECT book_id, authors.author_id as author_id, authors.name as name,
                         main_author, race, gender, about, influences,
                         authors.works_count as works_count, hometown, born_at, died_at
                 FROM book_authors
                 INNER JOIN authors
                 ON book_authors.author_id = authors.author_id),
            k_classified AS
                 (SELECT authors_books_classified.*, k_label
                 FROM authors_books_classified
                 LEFT JOIN kmeans
                 ON authors_books_classified.book_id = kmeans.book_id)
            SELECT books.*, author_id, name, main_author, race, gender, about,
                   influences, works_count, hometown, born_at, died_at, k_label
            FROM books
            INNER JOIN
            k_classified
            ON books.best_book_id = k_classified.book_id;
            """
    engine = create_engine('postgresql://postgres@localhost/books')
    return pd.read_sql_query(query, engine)


def get_amazon_review_text(csv_file):
    """
    INPUT:
    - json file with Amazon reveiws
    - dictionary of ISBNs (subset we care about) to GoodReads best book id
    OUTPUT:
    Series of aggregated reviews grouped by GoodReads best book id
    """
    df_reviews = pd.read_csv(csv_file, header=None,
                             names=['best_book_id', 'asin', 'summary',
                                    'review_text'])
    df_reviews = df_reviews[df_reviews['review_text'].isnull() == False]
    df_reviews_agg = df_reviews.groupby('best_book_id')['review_text'].agg(lambda col: ' '.join(col))
    return df_reviews_agg


def get_amazon_ratings(csv_file):
    """
    INPUT:
    - csv file with Amazon reveiws
    - dictionary of ISBNs (subset we care about) to GoodReads best book id
    OUTPUT:
    DataFrame with the following columns:
    'user_id', 'book_id', 'rating'
    """
    df_a_ratings = pd.read_csv(csv_file, header=None,
                               names=['book_id', 'asin', 'user_id',
                                      'helpful', 'rating', 'unix_review_time'])
    df_a_ratings = df_a_ratings[['user_id', 'book_id', 'rating']]
    return df_a_ratings


def get_goodread_data(ratings_csv, books_csv):
    """
    INPUT:
    From Kaggle dataset
    - ratings_csv - book_id (nothing to do with goodreads), user_id (who knows),
      rating
    - book_csv - we will use this to get the Kaggle book ID to the goodreads
      best book id
    OUTPUT:
    DataFrame with the following columns:
    'user_id','book_id','rating'
    """
    df_ratings = pd.read_csv(ratings_csv)
    df_books = pd.read_csv(books_csv, usecols=['book_id', 'best_book_id'])
    dict_best_id = df_books.set_index(['book_id'])['best_book_id'].to_dict()
    df_ratings['book_id'] = df_ratings['book_id'].map(lambda x: dict_best_id.get(x))
    return df_ratings


def load_table(table_name):
    """
    Load tabel and don't have to recreate the engine every time
    """
    engine = create_engine('postgresql://postgres@localhost/books')
    df = pd.read_sql_table(table_name, engine)
    return df


if __name__ == '__main__':

    df_books = get_books()
    df_authors = get_classified_authors()
    df_authors_books = get_books_to_authors()
    df_isbn_best_book_id = get_isbn_to_best_book_id()
    df_books_classified = merge_to_classify_books()


    # From Kaggle's Goodbooks-10K
    gr_rating_file = '../data/goodbooks-10k/ratings.csv'
    gr_book_file = '../data/goodbooks-10k/books.csv'
    # Created from Amazon Review file
    a_ratings_file = '../data/limited_amazon_ratings.csv'
    a_reviews_file = '../data/limited_amazon_reviews.csv'
    # df_gr_ratings = get_goodread_data(gr_rating_file, gr_book_file)
    # df_a_ratings = get_amazon_ratings(a_ratings_file)
    # df_reviews_agg = get_amazon_review_text(a_reviews_file)
