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
    min_err_top_5 = float('inf')
    best_regParam = None
    best_rank = None
    best_recommender = None
    for regParam in regParam_list:
        for rank in rank_list:
            recommender = train_recommender(train, regParam=regParam, rank=rank)
            rmse, rmse_top_5 = recommender_rmse(recommender, train, test)
            print("regParam: {}, rank: {}\nRMSE: {}\nTop 5 RMSE: {}".format(regParam, rank, rmse, rmse_top_5))
            print("--"*20)
            save_matrix(recommender, str(rank))
            if rmse_top_5 < min_err_top_5:
                min_err = rmse
                min_err_top_5 = rmse_top_5
                best_regParam = regParam
                best_rank = rank
                best_recommender = recommender
    print("Best regParam: {}\nBest rank: {}\nBest RMSE: {}\nTop 5 Best RMSE: {}".format(best_regParam, best_rank, min_err, min_err_top_5))
    return best_regParam, best_rank


def grid_search_rec_per_rank(train_df, train_val_df, regParam_list, rank_list):
    metrics = []
    train, test = train_df.randomSplit([0.7, 0.3], seed=72)
    for rank in rank_list:
        min_err = float('inf')
        min_err_top_5 = float('inf')
        best_regParam = None
        for regParam in regParam_list:
            recommender = train_recommender(train, regParam=regParam, rank=rank)
            rmse, rmse_top_5 = recommender_rmse(recommender, train, test)
            metrics.append([rank, regParam, rmse, rmse_top_5])
            print("regParam: {}, rank: {}\nRMSE: {}\nTop 5 RMSE: {}".format(rank, regParam, rmse, rmse_top_5))
            print("--"*20)
            if rmse_top_5 < min_err_top_5:
                min_err = rmse
                min_err_top_5 = rmse_top_5
                best_regParam = regParam
        recommender = train_recommender(train_df, regParam=best_regParam, rank=rank)
        save_matrix(recommender, str(rank) + "_train")
        recommender_val = train_recommender(train_val_df, regParam=best_regParam, rank=rank)
        save_matrix(recommender_val, str(rank) + "_test")
    print(metrics)
    return metrics

def save_matrix(recommender, prefix):
    np.save(prefix + '_item_matrix', recommender.itemFactors.toPandas().as_matrix())
    np.save(prefix + '_user_matrix', recommender.userFactors.toPandas().as_matrix())
    print("Saved: {} matrix".format(prefix))


def recommender_rmse(recommender, train, test):
    predictions = recommender.transform(test)
    predictions_df = predictions.toPandas()
    train_df = train.toPandas()
    predictions_df = predictions.toPandas().fillna(train_df['rating'].mean())
    predictions_df['squared_error'] = (predictions_df['rating'] -
                                       predictions_df['prediction']) ** 2
    rmse = np.sqrt(sum(predictions_df['squared_error']) / len(predictions_df))
    g = predictions_df.groupby('user_id')
    top_5 = g.rating.transform(lambda x: x >= x.quantile(.95))
    predictions_df = predictions_df[top_5 == 1]
    predictions_df['squared_error_top_5'] = (predictions_df['rating'] -
                                             predictions_df['prediction']) ** 2
    rmse_top_5 = np.sqrt(sum(predictions_df['squared_error_top_5']) / len(predictions_df))
    return rmse, rmse_top_5


def load_books_data():
    # Created from GoodReads API
    book_file = '../data/updated_books.csv'
    # Created from GoodReads API, and manual classification
    author_file = '../data/classified_authors.csv'
    # Created from GoodReads API
    author_book_file = '../data/author_books.csv'
    # Created from Amazon Review file for ASIN and GoodReads API
    asin_best_file = '../data/asin_best_book_id.csv'
    # From Kaggle's Goodbooks-10K
    k_rating_file = '../data/goodbooks-10k/ratings.csv'
    k_book_file = '../data/goodbooks-10k/books.csv'

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

    # Only training using GoodReads data because we will be predicting on
    # GoodReads data and the datasets look different
    # GoodReads Data:
    # Books/User: 111.87
    # 5 star 33%
    # 4 star 36%
    # 3 star 23%
    # 2 star 6%
    # 1 star 2%
    # Amazon Data:
    # Books/User: 3.98
    # 5 star 56%
    # 4 star 24%
    # 3 star 11%
    # 2 star 5%
    # 1 star 4%

    df_train = pd.read_csv('../data/goodbooks-10k/train-ratings.csv')
    df_train_val = pd.read_csv('../data/goodbooks-10k/train_val-ratings.csv')

    train_df = spark.createDataFrame(df_train)

    train_val_df = spark.createDataFrame(df_train_val)

    regParam_list = [.01, .05, .1]
    rank_list = range(11, 42)[::2]
    metrics = grid_search_rec_per_rank(train_df, regParam_list, train_val_df, rank_list)
