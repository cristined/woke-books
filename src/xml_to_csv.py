import pandas as pd
import numpy as np
import requests
from xml.etree import ElementTree
from collections import Counter
import time
import os
import csv


def get_text(item):
    try:
        return item.text
    except AttributeError:
        return None


def get_author_book_csv(file, csvfile):
    parser = ElementTree.XMLParser(encoding="utf-8")
    tree = ElementTree.parse(file, parser=parser)
    root = tree.getroot()
    book_id = root.find('book/id')
    for author in root.findall('book/authors/author'):
        author_id = author.find('id')
        name = author.find('name')
        print(get_text(name))
        role = author.find('role')
        csvfile.append(map(get_text, [book_id, author_id, name, role]))


def get_book_csv(file, csvfile):
    parser = ElementTree.XMLParser(encoding="utf-8")
    tree = ElementTree.parse(file, parser=parser)
    root = tree.getroot()
    for book in root.findall('book'):
        book_id = book.find('id')
        title = book.find('title')
        print(get_text(title))
        title_without_series = book.find('title_without_series')
        isbn = book.find('isbn')
        isbn13 = book.find('isbn13')
        asin = book.find('asin')
        kindle_asin = book.find('kindle_asin')
        country_code = book.find('country_code')
        language_code = book.find('language_code')
        description = book.find('description')
        work_id = book.find('work/id')
        best_book_id = book.find('work/best_book_id')
        original_title = book.find('work/original_title')
        csvfile.append(map(get_text, [book_id, title, title_without_series, isbn, isbn13, asin, country_code, language_code, description, work_id, best_book_id, original_title]))


def create_csv(xml_func, directory, col, new_csvname):
    csv_list = []
    files_data = os.listdir(directory)
    error_lst = []
    for f in files_data:
        filename = '{}/{}'.format(directory, f)
        try:
            xml_func(filename, csv_list)
        except ElementTree.ParseError:
            error_lst.append(f)
    df = pd.DataFrame(csv_list, columns=col)
    df.to_csv(new_csvname)
    pd.DataFrame(error_lst).to_csv('errors.csv')


if __name__ == '__main__':
    # directory = '../data/book_data'
    # new_csvname = '../data/updated_books.csv'
    directory = 'book_data'
    new_csvname = 'updated_books.csv'
    col = ['book_id', 'title', 'title_without_series', 'isbn', 'isbn13', 'asin', 'country_code', 'language_code', 'description', 'work_id', 'best_book_id', 'original_title']
    create_csv(get_book_csv, directory, col, new_csvname)

    # directory = '../data/book_data'
    # new_csvname = '../data/author_books.csv'

    directory = 'book_data'
    new_csvname = 'author_books.csv'
    col = ['book_id', 'author_id', 'name', 'role']
    create_csv(get_author_book_csv, directory, col, new_csvname)
