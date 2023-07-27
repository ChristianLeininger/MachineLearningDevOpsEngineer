# copyright 2023
# author: Christian Leininger
# Email: leininic@tf.uni-freiburg.de
# date: 20.07.2023


# import libraries
import os
import ast
import shap
import wandb
import hydra

from omegaconf import DictConfig
import argparse
import joblib
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report

from utils import get_categorical_columns, save_plot, save_describtion, save_statistics
from utils import save_histplot, save_correlation_matrix, add_churn_encoded_column


# Turn off interactive mode to suppress plot display
plt.ioff()
os.environ['QT_QPA_PLATFORM']='offscreen'



class ChurnLibrary:
    def __init__(self, cfg: DictConfig):
        """  Constructor for this class.
        Args:
            pth (Path): Path object pointing to csv file
        """
        logging.info("Instantiating ChurnLibrary object...")
        # some assertions
        assert isinstance(cfg, DictConfig), "cfg should be a DictConfig"
        assert isinstance(cfg['seed'], int), "seed should be an integer"
        assert isinstance(cfg.data['split_ratio'], float), "split_ratio should be a float"
        assert isinstance(cfg.data['file_name'], str), "file_name should be a string"
        assert isinstance(cfg.data['file_name'].endswith('.csv'), bool), "file_name end .csv"
        assert isinstance(cfg['pth'], str), "pth should be a string"
        assert isinstance(cfg['auto'], bool), "auto should be a boolean"
        assert isinstance(cfg['track'], bool), "track should be a boolean"
        assert isinstance(cfg.models.log_reg["solver"], str), "solver should be a string"
        assert isinstance(cfg.models.log_reg["max_iter"], int), "max_iter should be an integer"
        assert isinstance(cfg.models.log_reg["verbose"], int), "verbose should be an integer"

        assert isinstance(cfg.wandb['project_name'], str), "project_name should be a string"

        # check if directory exists
        assert os.path.exists(cfg['pth']), "pth should be a valid path change in hydra config file"
        self.seed = cfg['seed']
        self.split_ratio = cfg.data['split_ratio']
        self.auto = cfg['auto']
        self.log_reg_solver = cfg.models.log_reg["solver"]
        self.log_reg_max_iter = cfg.models.log_reg["max_iter"]
        self.verbose = cfg.models.log_reg["verbose"]     # 0 = no output, 1 = some, 2 = all
        self.path = cfg['pth']
        self.param_grid = cfg.models.random_forest["param_grid"]
        self.data_path = os.path.join(self.path, 'data')
        self.data_file = os.path.join(self.data_path, cfg.data['file_name'])
        # check file exists
        assert os.path.exists(self.data_file), "file_name not valid change in hydra config file"
        self.image_path = os.path.join(self.path, 'images')
        self.model_path = os.path.join(self.path, 'models')
        self.logs_path = os.path.join(self.path, 'logs')
        # plot data
        self.figsize = ast.literal_eval(cfg.plot['figsize'])
        self.alpha = cfg.plot['alpha']
        self.cat_columns = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'
            ]
        self.quant_columns = [
            'Customer_Age',
            'Dependent_count',
            'Months_on_book',
            'Total_Relationship_Count',
            'Months_Inactive_12_mon',
            'Contacts_Count_12_mon',
            'Credit_Limit',
            'Total_Revolving_Bal',
            'Avg_Open_To_Buy',
            'Total_Amt_Chng_Q4_Q1',
            'Total_Trans_Amt',
            'Total_Trans_Ct',
            'Total_Ct_Chng_Q4_Q1',
            'Avg_Utilization_Ratio'
            ]
        self.keep_cols = ['Customer_Age',
                          'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
             'Income_Category_Churn', 'Card_Category_Churn']
        # create images folder if doesnt already exist
        if not os.path.exists(self.image_path):
            os.makedirs(self.image_path)
        # create models folder if doesnt already exist
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        # create logs folder if doesnt already exist
        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)
        logging.info("Successfully instantiated ChurnLibrary object.")
        # initialize wandb
        self.track = cfg['track']
        if self.track:
            self.init_wandb(project_name=cfg.wandb['project_name'], experiment_name=cfg.wandb['experiment_name'])

    def init_wandb(self, project_name: str, experiment_name: str) -> None:
        """ Initialize wandb
        Args:
            project_name (str): Name of project
            experiment_name (str): Name of experiment
        """
        logging.info("Initializing wandb...")
        assert isinstance(project_name, str), "project_name should be a string"
        assert isinstance(experiment_name, str), "experiment_name should be a string"
        # initialize wandb
        self.run = wandb.init(project=project_name,
                   name=experiment_name,
                   )
        logging.info("Successfully initialized wandb.")

    def import_data(self):
        '''
        uses pandas to import csv to dataframe uses self.pth as path
        and stores dataframe as self.df member variable
        Output: None
        '''

        self.df = pd.read_csv(self.data_file)
        logging.info(f"Successfully imported {self.data_file} to dataframe with shape {self.df.shape}.")



    def perform_eda(self) -> None:
        '''
        perform eda on df and save figures to images folder

        output:
                None
        '''
        logging.info("Performing EDA...")
        # create a new column called Churn that is 1 if Attrition_Flag is "Attrited Customer" and 0 otherwise
        self.df['Churn'] = self.df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)

        save_plot(df=self.df, feature='Churn', save_path=self.image_path)
        save_plot(df=self.df, feature='Customer_Age', save_path=self.image_path)
        save_describtion(df=self.df, save_path=self.image_path, file_name='describe')
        save_statistics(df=self.df, column="Marital_Status", save_path=self.image_path)
        save_histplot(df=self.df, column="Total_Trans_Ct", save_path=self.image_path)
        save_correlation_matrix(df=self.df, save_path=self.image_path)
        logging.info("Successfully performed eda and saved figures to images folder.")
        # import pdb; pdb.set_trace()


    def encoder_helper(self, category_lst: list, response: str):
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
        import pdb; pdb.set_trace()


    def perform_feature_engineering(self, response: str = 'Churn') -> tuple:
        '''
        input:
                response: string of response name [optional argument that could be used for naming variables or index y column]

        output:
                X_train: X training data
                X_test: X testing data
                y_train: y training data
                y_test: y testing data
        '''

        # create a list of categorical columns to dummy
        if self.auto:
            cat_cols = get_categorical_columns(self.df)
            logging.info(f"Automatically detected categorical columns: {cat_cols}")
        else:
            cat_cols = self.cat_columns
            logging.info(f"Using provided categorical columns: {cat_cols}")
        # add churn encoded column
        self.df = add_churn_encoded_column(df=self.df, column_names=cat_cols)
        # import pdb; pdb.set_trace()
        self.df = self.df[self.keep_cols + [response]]
        logging.info(f"Successfully performed feature engineering.")
        logging.info(f"Final df shape: {self.df.shape} features: {self.df.columns}")

        # split into train and test
        X_train, X_test, y_train, y_test = train_test_split(self.df.drop(columns=[response]), self.df[response], test_size=self.split_ratio, random_state=self.seed)
        logging.info(f"Successfully split data into train and test sets.")
        logging.info(f"X_train shape: {X_train.shape} y_train shape: {y_train.shape}")
        logging.info(f"X_test shape: {X_test.shape} y_test shape: {y_test.shape}")
        logging.info(f"test size: {self.split_ratio} random state: {self.seed}")

        return X_train, X_test, y_train, y_test

    def load_model(self):
        """  """
        # load models
        logging.info(f"Loading models from {self.model_path}...")
        assert os.path.exists(self.model_path), "model_path should be a valid path change in hydra config file"
        assert os.path.exists(os.path.join(self.model_path, 'rf_model.joblib.pkl')), "rf_model.joblib.pkl"
        assert os.path.exists(os.path.join(self.model_path, 'lrc_model.joblib.pkl')), "lrc_model.joblib.pkl"
        rfc = load(os.path.join(self.model_path, 'rf_model.joblib.pkl'))
        lrc = load(os.path.join(self.model_path, 'lrc_model.joblib.pkl'))
        logging.info(f"Successfully loaded models from {self.model_path}.")
        return rfc, lrc


    def classification_report_image(self,
                                    lrc,
                                    rfc,
                                    x_train,
                                    x_test,
                                    y_train,
                                    y_test
                                    ):
        '''
        produces classification report for training and testing results and stores report as image
        in images folder
        input:
                x_train: training data
                x_test:  test data
                y_train: training response values
                y_test:  test response values


        output:
                None
        '''
        logging.info("Creating classification report image...")
        
        logging.info("Creating ROC Curve...")
        lrc_plot = plot_roc_curve(lrc, x_test, y_test)

        plt.figure(figsize=self.figsize)

        ax = plt.gca()
        lrc_plot.plot(ax=ax, alpha=self.alpha)
        rfc_disp = plot_roc_curve(rfc, x_test, y_test, ax=ax, alpha=0.8)
        plt.title('ROC Curve')
        plt.savefig(os.path.join(self.image_path, 'Roc_curve.png'))
        plt.clf()
        plt.close("all")
        # import pdb; pdb.set_trace()
        explainer = shap.TreeExplainer(rfc)
        shap_values = explainer.shap_values(x_test)
        logging.info("Creating SHAP Summary Plot...")
        shap.summary_plot(shap_values, x_test, plot_type="bar", show=False)
        plt.title("SHAP Summary Plot")
        # Save the plot to a PNG file
        plt.savefig(os.path.join(self.image_path, "Shap_summary_plot.png"), bbox_inches='tight')
        # import pdb; pdb.set_trace()
        plt.clf()
        logging.info("Successfully created SHAP Summary Plot.")



    def feature_importance_plot(self, 
                                rfc,
                                lrc, 
                                X_train,
                                X_test,
                                y_train,
                                y_test,
                                ):
        '''
        creates and stores the feature importances in pth
        input:
                model: model object containing feature_importances_
                X_test: X testing data
                X_train: X training data
                y_train: y training data
                y_test: y testing data
        output:
                None
        '''
        # plots
        # Calculate feature importances
        logging.info("Creating feature importance plot...")
        logging.info("Calculating y_test_preds_rf...")
        y_train_preds_rf = rfc.predict(X_train)
        y_test_preds_rf = rfc.predict(X_test)
        y_train_preds_lrc = lrc.predict(X_train)
        y_test_preds_lrc = lrc.predict(X_test)
        importances = rfc.feature_importances_
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]
        # Rearrange feature names so they match the sorted feature importances
        names = [X_test.columns[i] for i in indices]
        # Create plot
        plt.figure(figsize=self.figsize)
        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel('Importance')
        # Add bars
        plt.bar(range(X_test.shape[1]), importances[indices])
        # Add feature names as x-axis labels
        plt.xticks(range(self.df.shape[1]), names, rotation=90)
        # Save plot
        plt.savefig(os.path.join(self.image_path, 'Feature_importance.png'),bbox_inches='tight')
        plt.clf()
        plt.close("all")
        plt.rc('figure', figsize=self.figsize)
        #plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
        plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10}, fontproperties = 'monospace')
        plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
        plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10}, fontproperties = 'monospace')
        plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
        plt.axis('off')
        # save plot
        plt.savefig(os.path.join(self.image_path, 'Random_Forest_Classification_report'),bbox_inches='tight')
        plt.clf()
        plt.close("all")
        plt.rc('figure', figsize=self.figsize)
        plt.text(0.01, 1.25, str('Logistic Regression Train'), {'fontsize': 10}, fontproperties = 'monospace')
        plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lrc)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
        plt.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10}, fontproperties = 'monospace')
        plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lrc)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
        plt.axis('off')
        # save plot
        plt.savefig(os.path.join(self.image_path, 'Logistic_Regression_Classification_report'),bbox_inches='tight')
        logging.info("Successfully created feature importance plot.")
        # import pdb; pdb.set_trace()



    def train_models(self, X_train , X_test, y_train, y_test):
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
        # some assertions
        assert isinstance(X_train, pd.DataFrame), "X_train should be a Pandas DataFrame"
        assert isinstance(X_test, pd.DataFrame), "X_test should be a Pandas DataFrame"
        assert isinstance(y_train, pd.Series), "y_train should be a Pandas Series"
        assert isinstance(y_test, pd.Series), "y_test should be a Pandas Series"
        # grid search
        logging.info("Init RandomForest ...")
        rfc = RandomForestClassifier(random_state=self.seed)
        # Use a different solver if the default 'lbfgs' fails to converge
        # Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
        lrc = LogisticRegression(
            solver=self.log_reg_solver , max_iter=self.log_reg_max_iter, random_state=self.seed, verbose=self.verbose)


        logging.info("Performing grid search...")
        cv_rfc = GridSearchCV(estimator=rfc, param_grid=self.param_grid, cv=self.cv)
        cv_rfc.fit(X_train, y_train)
        logging.info("Successfully performed grid search.")
        logging.info("Performing logistic regression...")
        lrc.fit(X_train, y_train)
        logging.info("Successfully performed logistic regression.")

        y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
        y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

        y_train_preds_lr = lrc.predict(X_train)
        y_test_preds_lr = lrc.predict(X_test)
        logging.info(f"Successfully trained models.")
        # scores
        logging.info('random forest results')
        logging.info('test results')
        logging.info(classification_report(y_test, y_test_preds_rf))
        logging.info('train results')
        logging.info(classification_report(y_train, y_train_preds_rf))
        logging.info('logistic regression results')
        logging.info('test results')
        logging.info(classification_report(y_test, y_test_preds_lr))
        logging.info('train results')
        logging.info(classification_report(y_train, y_train_preds_lr))
        logging.info("Creating classification report image...")

        report_rf = classification_report(y_test, y_test_preds_rf, output_dict=True)
        report_df = pd.DataFrame(report_rf).transpose()
        #save report
        report_df.to_csv(os.path.join(self.logs_path, 'report_rf.csv'))
        # save model
        dump(cv_rfc.best_estimator_, os.path.join(self.model_path, 'rf_model.joblib.pkl'))
        # save report
        report_lr = classification_report(y_test, y_test_preds_lr, output_dict=True)
        report_df = pd.DataFrame(report_lr).transpose()
        report_df.to_csv(os.path.join(self.logs_path, 'report_lr.csv'))
        # save model
        dump(lrc, os.path.join(self.model_path, 'lrc_model.joblib.pkl'))
        logging.info(f"Successfully saved models to {self.model_path} and reports to {self.logs_path}.")
        # save shap
        explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
        shap_values = explainer.shap_values(X_train)
        shap.summary_plot(shap_values, X_train, plot_type="bar")
        plt.savefig(os.path.join(self.image_path, 'Shap_values_treeExplainer.png'))
        logging.info(f"Successfully saved shap values to {self.image_path}.")
        logging.info(f"Successfully created classification report image.")
        logging.info(f"... completed training and saving models and reports.")



@hydra.main(config_path="config", config_name="stable_train")
def main(cfg):
    '''
    main function
    Args:
        cfg (DictConfig): Hydra config
    '''
    # instantiate logging
    # Map the logging levels to their corresponding attributes
    # import pdb; pdb.set_trace()
    log_levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }

    # Set the logging level based on the command-line argument
    logging.basicConfig(level=log_levels[cfg['log_level']])
    # instantiate class
    cly = ChurnLibrary(cfg=cfg)
    cly.import_data()  # import data
    cly.perform_eda()  # perform eda
    # perform feature engineering
    # cly.encoder_helper
    X_train, X_test, y_train, y_test = cly.perform_feature_engineering(response='Churn')
    if cfg['train_models']:
        cly.train_models(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    else:
        rfc, lrc = cly.load_model()  # load models

    # cly.classification_report_image(lrc=lrc, rfc=rfc, x_train=X_train, x_test=X_test, y_train=y_train, y_test=y_test)
    cly.feature_importance_plot(rfc=rfc, lrc=lrc, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    logging.info("Successfully completed main function.")


if __name__ == '__main__':
    main()
