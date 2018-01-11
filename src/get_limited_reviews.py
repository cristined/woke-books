import pandas as pd
import gzip
import json
import csv
import os
import requests
from xml.etree import ElementTree
import time


def asin_we_care_about(path, to_path, dict_isbn_best_id, api_key):
    '''
    INPUT
    - Path to json.gz file to open and read
    - Path you would like to create the csv at
    - Dictionary of isbn to goodreads best book ID you care about
    - GoodReads API key
    OUTPUT
    - csv file with only the reviews for best book IDs we care about
    '''
    best_id_set = set(dict_isbn_best_id.values())
    with gzip.open(path, 'rb') as f, open(to_path, 'w') as to:
        to = csv.writer(to)
        for line in f:
            json_line = json.loads(line)
            asin = json_line['asin']
            try:
                best_book_id = dict_isbn_best_id[asin]
            except KeyError:
                book_id, best_book_id = get_book_data(asin, api_key)
                dict_isbn_best_id[asin] = best_book_id
            if best_book_id in best_id_set:
                to.writerow([best_book_id,
                            asin,
                            json_line['reviewerID'],
                            json_line['helpful'],
                            json_line['overall'],
                            json_line['summary'],
                            json_line['reviewText'],
                            json_line['unixReviewTime']])


def create_asin_best_book_id_csv(path, to_path, dict_isbn_best_id, api_key):
    '''
    INPUT
    - Path to json.gz file to open and read
    - Path you would like to create the csv at
    - Dictionary of isbn to goodreads best book ID you care about
    - GoodReads API key
    OUTPUT
    - csv file with only the reviews for best book IDs we care about
    '''
    i = 0
    with gzip.open(path, 'rb') as f, open(to_path, 'w') as to:
        to = csv.writer(to)
        for asin, best_book_id in dict_isbn_best_id.items():
            to.writerow([asin, best_book_id])
            i += 1
        for line in f:
            json_line = json.loads(line)
            asin = json_line['asin']
            if asin not in dict_isbn_best_id:
                try:
                    best_book_id = dict_isbn_best_id[asin]
                except KeyError:
                    book_id, best_book_id = get_book_data(asin, api_key)
                    dict_isbn_best_id[asin] = best_book_id
                to.writerow([asin,
                            best_book_id])
                i += 1
                print(i)


def get_book_data(isbn, api_key):
    '''
    INPUT
    - isbn to look up
    - GoodReads API key
    OUTPUT
    - goodreads book ID and goodreads best book ID
    '''
    isbn_to_id_url = 'https://www.goodreads.com/book/isbn_to_id/{}?key={}'.format(isbn, api_key)
    book_id = requests.get(isbn_to_id_url).text
    book_url = 'https://www.goodreads.com/book/show/{}.xml?key={}'.format(book_id, api_key)
    try:
        response = requests.get(book_url)
    except requests.exceptions.SSLError:
        time.sleep(10)
        response = requests.get(book_url)
    try:
        tree = ElementTree.fromstring(response.content)
        best_book_id = get_text(tree.find('book/work/best_book_id'))
        print(book_id, best_book_id)
        return book_id, best_book_id
    except ElementTree.ParseError:
        return None, None


def get_text(item):
    '''
    INPUT
    - XML item
    OUTPUT
    - Text if possible, otherwise None
    '''
    try:
        return item.text
    except AttributeError:
        return None


def get_review_text_to_csv(path, to_path, dict_isbn_best_id):
    '''
    INPUT
    - Path to json.gz file to open and read
    - Path you would like to create the csv at
    - Dictionary of isbn to goodreads best book ID you care about
    OUTPUT
    - csv file with only the reviews for best book IDs we care about
    '''
    best_id_set = set(dict_isbn_best_id.values())
    with gzip.open(path, 'rb') as f, open(to_path, 'w') as to:
        to = csv.writer(to)
        for line in f:
            json_line = json.loads(line)
            asin = json_line['asin']
            try:
                best_book_id = dict_isbn_best_id[asin]
            except KeyError:
                book_id, best_book_id = get_book_data(asin, api_key)
                dict_isbn_best_id[asin] = best_book_id
            if best_book_id in best_id_set:
                to.writerow([best_book_id,
                            asin,
                            json_line['summary'],
                            json_line['reviewText']])


def get_amazon_ratings(json_file, dict_isbn_best_id, output_filename):
    """
    INPUT:
    - json file with Amazon reveiws
    - dictionary of ISBNs (subset we care about) to GoodReads best book id
    OUTPUT:
    csv file with the following columns:
    'user_id', 'best_book_id', 'rating'
    """
    best_id_set = set(dict_isbn_best_id.values())
    with gzip.open(path, 'rb') as f, open(to_path, 'w') as to:
        to = csv.writer(to)
        for line in f:
            json_line = json.loads(line)
            asin = json_line['asin']
            try:
                best_book_id = dict_isbn_best_id[asin]
            except KeyError:
                book_id, best_book_id = get_book_data(asin, api_key)
                dict_isbn_best_id[asin] = best_book_id
            if best_book_id in best_id_set:
                to.writerow([best_book_id,
                            asin,
                            json_line['reviewerID'],
                            json_line['helpful'],
                            json_line['overall'],
                            json_line['unixReviewTime']])


if __name__ == '__main__':
    print('getting reviews')
    # df_books_1 = pd.read_csv('../data/updated_books.csv')
    df_books = pd.read_csv('asin_best_book_id_take_2.csv', header=None,
                           names=['asin', 'best_book_id'])
    dict_isbn_best_id = df_books.set_index(['asin'])['best_book_id'].to_dict()

    api_key = os.environ['GOODREADS_API_KEY']

    # asin_we_care_about('reviews_Books_5.json.gz',
    #                    'best_id_reviews.csv',
    #                    dict_isbn_best_id,
    #                    api_key)
    # # asin_we_care_about('../data/reviews_Books_5.json.gz',
    # #                    '../data/best_id_reviews.csv',
    # #                    dict_isbn_best_id,
    # #                    api_key)

    create_asin_best_book_id_csv('reviews_Books_5.json.gz',
                                 'asin_best_book_id_take_3.csv',
                                 dict_isbn_best_id,
                                 api_key)
    # create_asin_best_book_id_csv('../data/reviews_Books_5.json.gz',
    #                            '../data/asin_best_book_id.csv',
    #                            dict_isbn_best_id,
    #                            api_key)
