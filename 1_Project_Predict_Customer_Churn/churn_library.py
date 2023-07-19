# library doc string


# import libraries
import os
import shap
import argparse 
import joblib
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report


os.environ['QT_QPA_PLATFORM']='offscreen'


class ChurnLibrary:
    def __init__(self, pth: str):
        assert pth.name.endswith('.csv'), 'File must be a csv'
        self.pth = pth
    
    def import_data(self):
        '''
        uses pandas to import csv to dataframe uses self.pth as path
        and stores dataframe as self.df member variable
        '''	

        try:
             self.df = pd.read_csv(self.pth)
             logging.info(f"Successfully imported {self.pth} to dataframe with shape {self.df.shape}.")
        except FileNotFoundError:
             logging.info(f"Sorry, the file {self.pth} does not exist.")
        except pd.errors.EmptyDataError:
             logging.info(f"The file {self.pth} is empty.")
        except pd.errors.ParserError:
             logging.info(f"Couldn't parse the file {self.pth}.")
        except Exception as e:
             logging.info(f"An unexpected error occurred while reading the file {self.pth}")


def perform_eda(df: pd.DataFrame) -> None:
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    pass


def encoder_helper(df: pd.DataFrame, category_lst: list, response: str):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    pass


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass




def main(args):
    '''
    main function
    '''
    # instantiate logging
    # Map the logging levels to their corresponding attributes
    log_levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }

    # Set the logging level based on the command-line argument
    logging.basicConfig(level=log_levels[args.log_level])
    # instantiate class
    cl = ChurnLibrary(pth=args.input_file)   






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="An advanced argparse example")

    # Optional argument: --input-file or -i
    parser.add_argument("-i", "--input-file", type=str, default="data/BankChurners.csv", help="Path to the input file")

    parser.add_argument("-l", "--log-level", choices=["debug", "info", "warning", "error", "critical"],
                        default="info", help="Set the logging level (default: info)")
    args = parser.parse_args()
    main(args)