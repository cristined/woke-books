import pandas as pd
from sqlalchemy import create_engine
from collections import Counter


def get_books(updated_csv_file, original_csv_file):
    """
    INPUT:
    csv filename
    OUTPUT:
    DataFrame with the following columns:
    'book_id', 'title','isbn', 'isbn13', 'country_code',
    'language_code', 'description', 'work_id', 'best_book_id', 'original_title'
    """
    df_books = pd.read_csv(updated_csv_file, usecols=['book_id', 'title',
                           'isbn', 'isbn13', 'country_code',
                           'language_code', 'description', 'work_id',
                           'best_book_id', 'original_title'])
    df_books = df_books[df_books['language_code'].map(lambda x: x in ['eng', 'en-US', 'en-GB', 'en-CA', 'en'])]
    df_books_org = pd.read_csv(original_csv_file, usecols=['best_book_id',
                               'ratings_count', 'work_ratings_count',
                               'work_text_reviews_count', 'ratings_1',
                               'ratings_2', 'ratings_3', 'ratings_4',
                               'ratings_5', 'image_url', 'small_image_url'])
    df_books = pd.merge(df_books, df_books_org, how='left',
                        left_on='best_book_id', right_on='best_book_id')
    df_books = df_books.set_index('book_id')
    return df_books


def get_classified_authors(csv_file):
    """
    INPUT:
    csv filename
    OUTPUT:
    DataFrame with the following columns:
    'author_id', 'name', 'race', 'gender', 'image_url',
    'about', 'influences', 'works_count', 'hometown', 'born_at', 'died_at'
    """
    df_authors = pd.read_csv(csv_file, usecols=['author_id', 'name', 'main_author', 'race', 'gender', 'image_url',
       'about', 'influences', 'works_count', 'hometown', 'born_at', 'died_at'])
    df_authors['race'] = df_authors['race'].map(lambda x: upper_strip(x))
    df_authors = df_authors[df_authors['race'].isnull() == False]
    df_authors['gender'] = df_authors['gender'].map(lambda x: upper_strip(x))
    df_authors = df_authors[df_authors['race'].isnull() == False]
    # List of races and genders from EEO values
    races = ['BLACK', 'WHITE', 'ASIAN', 'LATINO', 'NATIVE AMERICAN', 'MIXED',
             'PACIFIC ISLANDER']
    genders = ['FEMALE', 'MALE']
    # Invalid values will be replaced with majority of class
    race_majority = Counter(df_authors['race']).most_common()[0][0]
    gender_majority = Counter(df_authors['gender']).most_common()[0][0]
    df_authors['race'] = df_authors['race'].map(lambda x: replace_invalid_values(x, races, race_majority))
    df_authors['gender'] = df_authors['gender'].map(lambda x: replace_invalid_values(x, genders, gender_majority))
    df_authors = df_authors[df_authors['main_author']]
    df_authors = df_authors.set_index('author_id')
    return df_authors


def upper_strip(value):
    try:
        return value.upper().strip()
    except AttributeError:
        return value


def replace_invalid_values(value, valid_values, replacement):
    if value in valid_values:
        return value
    return replacement


def get_books_to_authors(csv_file):
    """
    INPUT:
    csv filename
    OUTPUT:
    DataFrame with the following columns:
    'book_id', 'author_id', 'name', 'role'
    """
    df_authors_books = pd.read_csv(csv_file, usecols=['book_id',
                                   'author_id', 'name', 'role'])
    df_authors_books = df_authors_books.set_index('book_id')
    return df_authors_books


def get_isbn_to_best_book_id(csv_file, our_best_book_ids=None):
    """
    INPUT:
    csv filename
    set of the best book ID's we care about if we would like to limit the file
    OUTPUT:
    DataFrame with the following columns:
    'isbn', 'best_book_id'
    """
    pass
    df_isbn_best_book_id = pd.read_csv(csv_file, header=None,
                                       names=['isbn', 'best_book_id'])
    df_isbn_best_book_id = df_isbn_best_book_id[df_isbn_best_book_id['best_book_id'].isnull() == False]
    if our_best_book_ids:
        df_isbn_best_book_id = df_isbn_best_book_id[df_isbn_best_book_id['best_book_id'].map(lambda x: x in our_best_book_ids)]
    df_isbn_best_book_id['best_book_id'] = df_isbn_best_book_id['best_book_id'].map(lambda x: int(x))
    df_isbn_best_book_id = df_isbn_best_book_id.set_index('isbn')
    return df_isbn_best_book_id


def get_kmeans_books(csv_file):
    """
    INPUT:
    csv filename
    set of the best book ID's we care about if we would like to limit the file
    OUTPUT:
    DataFrame with the following columns:
    'k_label', 'book_id'
    """
    return pd.read_csv(csv_file).set_index('k_label')


def pd_to_sql(df, name):
    engine = create_engine('postgresql+psycopg2://postgres@localhost/books')
    df.to_sql(name, engine)


if __name__ == '__main__':
    # Created from GoodReads API
    book_file = '../data/updated_books.csv'
    # From Kaggles Goodbooks 10k data set
    org_book_file = '../data/goodbooks-10k/books.csv'
    pd_to_sql(get_books(book_file, org_book_file), 'books')

    # Created from GoodReads API, and manual classification
    author_file = '../data/classified_authors.csv'
    pd_to_sql(get_classified_authors(author_file), 'authors')

    # Created from GoodReads API
    author_book_file = '../data/author_books.csv'
    pd_to_sql(get_books_to_authors(author_book_file), 'book_authors')

    # Created from Amazon Review file for ASIN and GoodReads API
    asin_best_file = '../data/asin_best_book_id.csv'
    pd_to_sql(get_isbn_to_best_book_id(asin_best_file), 'isbn_book_id')

    # Created from the reviews_cluster.py
    kmeans_book_id = '../data/kmeans_book_id.csv'
    pd_to_sql(get_kmeans_books(kmeans_book_id), 'kmeans')
