import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyspark
from pyspark.sql.types import *
from pyspark.ml.tuning import TrainValidationSplit
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import load_data
import get_user


def train_recommender(train):
    als_model = ALS(userCol='user_id',
                    itemCol='book_id',
                    ratingCol='rating',
                    nonnegative=True,
                    regParam=0.1,
                    rank=10
                    )
    recommender = als_model.fit(train)
    return recommender


def save_matrix(recommender):
    np.save('item_matrix', recommender.itemFactors.toPandas().as_matrix())
    np.save('user_matrix', recommender.userFactors.toPandas().as_matrix())


def recommender_rmse(recommender, test):
    predictions = recommender.transform(test)
    predictions_df = predictions.toPandas()
    train_df = train.toPandas()
    predictions_df = predictions.toPandas().fillna(train_df['rating'].mean())
    predictions_df['squared_error'] = (predictions_df['rating'] - predictions_df['prediction'])**2
    return np.sqrt(sum(predictions_df['squared_error']) / len(predictions_df))


def main():
    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    sc = spark.sparkContext
    spark, sc

    # Created from GoodReads API
    book_file = 'updated_books.csv'
    # Created from GoodReads API, and manual classification
    author_file = 'classified_authors.csv'
    # Created from GoodReads API
    author_book_file = 'author_books.csv'
    # Created from Amazon Review file for ASIN and GoodReads API
    asin_best_file = 'asin_best_book_id_take_3.csv'
    # From Kaggle's Goodbooks-10K
    k_rating_file = 'ratings.csv'
    k_book_file = 'books.csv'

    df_books = load_data.get_books(book_file)
    df_authors = load_data.get_classified_authors(author_file)
    df_authors_books = load_data.get_books_to_authors(author_book_file)
    df_isbn_best_book_id = load_data.get_isbn_to_best_book_id(asin_best_file)

    df_books_classified = load_data.merge_to_classify_books(df_authors_books, df_authors,
                                                  df_books)

    df_k_ratings = load_data.get_goodread_data(k_rating_file, k_book_file)

    ratings_df = spark.createDataFrame(df_k_ratings)

    train, test = ratings_df.randomSplit([0.8, 0.2], seed=427471138)

    recommender = train_recommender(train)
    save_matrix(recommender)

    print("RMSE = {}".format(recommender_rmse(recommender,
                                              test)))

if __name__ == "__main__":
    main()
