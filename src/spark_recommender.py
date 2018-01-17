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


def train_recommender(train, regParam=.1, rank=10):
    als_model = ALS(userCol='user_id',
                    itemCol='book_id',
                    ratingCol='rating',
                    nonnegative=True,
                    regParam=regParam,
                    rank=rank
                    )
    recommender = als_model.fit(train)
    return recommender


def grid_search_rec(train, test, regParam_list, rank_list):
    min_err = float('inf')
    best_regParam = None
    best_rank = None
    best_recommender = None
    for regParam in regParam_list:
        for rank in rank_list:
            recommender = train_recommender(train, regParam=regParam, rank=rank)
            rmse = recommender_rmse(recommender, test)
            print("regParam: {}, rank: {}\nRMSE: {}".format(regParam, rank, rmse))
            print("--"*20)
            if rmse < min_err:
                min_err = rmse
                best_regParam = regParam
                best_rank = rank
                best_recommender = recommender
    print("Best regParam: {}\nBest rank: {}\nBest RMSE: {}".format(best_regParam, best_rank, min_err))
    return best_recommender


def save_matrix(recommender):
    np.save('item_matrix', recommender.itemFactors.toPandas().as_matrix())
    np.save('user_matrix', recommender.userFactors.toPandas().as_matrix())


def recommender_rmse(recommender, test):
    predictions = recommender.transform(test)
    predictions_df = predictions.toPandas()
    train_df = train.toPandas()
    predictions_df = predictions.toPandas().fillna(train_df['rating'].mean())
    predictions_df['squared_error'] = (predictions_df['rating'] -
                                       predictions_df['prediction']) ** 2
    return np.sqrt(sum(predictions_df['squared_error']) / len(predictions_df))


def load_books_data():
    # Created from GoodReads API
    book_file = 'updated_books.csv'
    # Created from GoodReads API, and manual classification
    author_file = 'classified_authors.csv'
    # Created from GoodReads API
    author_book_file = 'author_books.csv'
    # Created from Amazon Review file for ASIN and GoodReads API
    asin_best_file = 'asin_best_book_id.csv'
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
    return df_books, df_authors, df_authors_books, df_isbn_best_book_id, df_books_classified, df_k_ratings


if __name__ == "__main__":
    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    sc = spark.sparkContext
    spark, sc

    df_books, df_authors, df_authors_books, df_isbn_best_book_id, df_books_classified, df_k_ratings = load_books_data()

    ratings_df = spark.createDataFrame(df_k_ratings)

    train, validate, test = ratings_df.randomSplit([0.6, 0.2, 0.2], seed=72)

    regParam_list = [.01, 1]
    rank_list = [5, 10, 15, 20]
    best_recommender = grid_search_rec(train, validate, regParam_list, rank_list)

    # save_matrix(best_recommender)

    # print("RMSE = {}".format(recommender_rmse(recommender, test)))
