import pandas as pd
import gzip
import json
import csv

def this_file_is_too_big(path, to_path, isbn_set):
    with gzip.open(path, 'rb') as f, open(to_path, 'w') as to:
        to = csv.writer(to)
        for line in f:
            json_line = json.loads(line)
            if json_line['asin'] in isbn_set:
                to.writerow([json_line['asin'],
                          json_line['reviewerID'],
                          json_line['helpful'],
                          json_line['overall'],
                          json_line['summary'],
                          json_line['reviewText'],
                          json_line['unixReviewTime']])


if __name__ == '__main__':
    print('getting reviews')
    df_books = pd.read_csv('updated_books.csv')
    isbn_set = set(df_books['isbn'].unique())
    print(isbn_set)
    df_reviews = this_file_is_too_big('reviews_Books_5.json.gz',
                                      'new_less_reviews.txt', isbn_set)
