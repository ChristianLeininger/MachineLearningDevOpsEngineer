# copyright 2023
# author: Christian Leininger
# Email: leininic@tf.uni-freiburg.de
# date: 18.06.2023


import pandas as pd
from typing import List, Dict, Tuple
import logging


def compute_median(df: pd.DataFrame, list_colmns: List) -> None:
    """ Compute the median of the dataframe

    Args:
        param1(pd.DataFrame): dataframe
        param2(List): list of columns

    """
    for column in list_colmns:
        try:
            median = df[column].median()
        except Exception as e:
            logging.info(f"error {e} occured")
            logging.error(f"could not compute median of {column}")
            continue
        for idx, value in enumerate(df[column]):
            if value >= median:
                df.loc[idx, column] = 'high'
            else:
                df.loc[idx, column] = 'low'

        logging.info(f"succesfully computed median of {column}")
        try:
            logging.info(f"{ df.groupby(column).quality.mean()}")
        except Exception as e:
            logging.info(f"error {e} occured")
            logging.error(f"could not compute mean of {column}")
            continue


def main():
    """ Main entry point of the app """
    df = pd.read_csv('winequality-red.csv', sep=';')
    logging.info(df.head())
    labels = list(df.columns)
    for idx in range(len(labels)):
        labels[idx] = labels[idx].replace(' ', '_')
    df.columns = labels
    list_column_names = df.columns.tolist()
    compute_median(df=df, list_colmns=list_column_names)
    logging.info(df.head())


if __name__ == "__main__":
    logging.basicConfig(
            level=logging.DEBUG, format="%(levelname)s: %(message)s")
    """ This is executed when run from the command line """
    main()
