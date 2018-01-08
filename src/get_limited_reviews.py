import pandas as pd
import gzip
import json
import csv
import os
import requests
from xml.etree import ElementTree


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
    response = requests.get(book_url)
    try:
        tree = ElementTree.fromstring(response.content)
        best_book_id = tree.find('book/work/best_book_id').text
        print(book_id, best_book_id)
        return book_id, best_book_id
    except ElementTree.ParseError:
        return None, None


if __name__ == '__main__':
    print('getting reviews')
    df_books = pd.read_csv('../data/updated_books.csv')
    dict_isbn_best_id = df_books.set_index(['isbn'])['best_book_id'].to_dict()

    api_key = os.environ['GOODREADS_API_KEY']
    asin_we_care_about('../data/reviews_Books_5.json.gz',
                       '../data/best_id_reviews.csv',
                       dict_isbn_best_id,
                       api_key)
