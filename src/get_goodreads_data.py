import pandas as pd
import numpy as np
import requests
from xml.etree import ElementTree
import time
import os


def get_book_data(df, save_dir, api_key):
    '''
    INPUT
    - dataframe to process needs to include goodreads_book_id
    - directory to save the XML in
    - Goodreads API key
    OUTPUT
    - XML saved in the directory
    '''
    for i, book in df.iterrows():
        book_id = book['goodreads_book_id']
        book_url = 'https://www.goodreads.com/book/show/{}.xml?key={}'.format(book_id, api_key)
        print(book_id)
        cur_path = '{}/{}{}'.format(save_dir, book_id, '.xml')
        r = requests.get(book_url).text.encode('utf-8')
        with open(cur_path, 'wb') as f:
            f.write(r)
        time.sleep(1)


def get_author_data(df, save_dir, api_key):
    '''
    INPUT
    - dataframe to process needs to include goodreads_book_id and authors
    - directory to save the XML in
    - Goodreads API key
    OUTPUT
    - XML saved in the directory
    '''
    processed_authors = []
    for i, book in df.iterrows():
        book_id = book['goodreads_book_id']
        book_authors = book['authors']
        if book_authors not in processed_authors:
            book_url = 'https://www.goodreads.com/book/show/{}.xml?key={}'.format(book_id, api_key)
            content = requests.get(book_url).content
            time.sleep(1)
            save_author_xml(content, save_dir, api_key)
            processed_authors.append(book_authors)


def save_author_xml(content, save_dir, api_key):
    '''
    INPUT
    - XML content
    - directory to save the file in
    - GoodReads API key
    OUTPUT
    - Saved XML file
    '''
    root = ElementTree.fromstring(content)
    for author in root.findall('book/authors/author'):
        author_id = author.find('id').text
        author_name = author.find('name').text
        author_url = 'https://www.goodreads.com/author/show/{}?format=xml&key={}'.format(author_id, api_key)
        print(author_name)
        cur_path = '{}/{}{}'.format(save_dir, author_id, '.xml')
        r = requests.get(author_url).text.encode('utf-8')
        with open(cur_path, 'w') as f:
            f.write(r)
        time.sleep(1)


if __name__ == '__main__':
    df_books = pd.read_csv('../data/books.csv')
    api_key = os.environ['GOODREADS_API_KEY']
    save_dir = '../data/book_data'
    get_book_data(df_books, save_dir, api_key)
