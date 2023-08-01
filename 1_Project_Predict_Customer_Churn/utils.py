from logging.handlers import RotatingFileHandler
import plotly.graph_objects as go
from datetime import datetime
import wandb
import logging
import os
from omegaconf import DictConfig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def get_logger(cfg: DictConfig) -> logging.Logger:
    """ Get logger

    Args:
        cfg (DictConfig): Configuration
    Returns:
        logging.Logger: Logger
    """
    # some assertions
    assert isinstance(cfg, DictConfig), "cfg should be a DictConfig"

    # Create a custom logger
    logger = logging.getLogger(cfg.job_logging.name)
    # Set to your desired logging level.
    logger.setLevel(cfg.job_logging.handlers.file.level)

    # Create a file handler
    current_data_time = datetime.now().strftime("%A,_%d_%B_%Y_%H:%M:%S")
    file_handler = RotatingFileHandler(
        os.path.join(
            cfg.pth,
            "logs",
            f"{cfg.job_logging.name}_{current_data_time}.log"),
        maxBytes=2000,
        backupCount=5)
    # Set to your desired logging level. Here, DEBUG is chosen.
    file_handler.setLevel(cfg.job_logging.handlers.file.level)

    # Create a console handler
    console_handler = logging.StreamHandler()
    # Set to your desired logging level. Here, ERROR is chosen.
    console_handler.setLevel(cfg.job_logging.handlers.file.level)

    # Create a formatter and set it for both handlers

    formatter = logging.Formatter(cfg.job_logging.formatters.basic.format)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add both file and console handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_categorical_columns(df: pd.DataFrame) -> list:
    """ Get list of categorical columns from a Pandas DataFrame

    Args:
        df (pd.DataFrame): Pandas DataFrame
    Returns:
        list: List of categorical columns
    """
    # Select columns of type 'object' and 'category'
    categorical_columns = df.select_dtypes(
        include=['object', 'category']).columns
    return list(categorical_columns)


def save_df_head(df: pd.DataFrame, save_path: str, figsize: tuple) -> None:
    """ Save head of Pandas DataFrame to image file
    Args:
        df (pd.DataFrame): Pandas DataFrame
        save_path (str): Path to save plot
        figsize (tuple): Size of plot
    """

    logging.info(f"Saving head of Pandas DataFrame to {save_path} ....")

    fig, ax = plt.subplots()
    # Adjust bbox values as needed to center the table
    # Create the table and set cell padding
    data = df.head().T.round(2)
    table = plt.table(cellText=data.values,
                      colLabels=df.index,
                      rowLabels=df.columns,
                      cellLoc='center',
                      loc='center',
                      )

    table.auto_set_column_width(col=list(range(len(df.columns))))
    ax.axis('off')
    plt.savefig(os.path.join(save_path, 'Head.png'), bbox_inches='tight')
    plt.close()
    plt.clf()
    logging.info(f"Saved head of Pandas DataFrame to {save_path}")


def save_plot(
        df: pd.DataFrame,
        feature: str,
        save_path: str,
        figsize: tuple) -> None:
    """ Save plot to file

    Args:
        df (pd.DataFrame): Pandas DataFrame
        feature (str): Feature to plot
        save_path (str): Path to save plot
    """
    # some assertions
    assert isinstance(df, pd.DataFrame), "df should be a Pandas DataFrame"
    assert isinstance(feature, str), "feature should be a string"
    assert feature in df.columns, "feature should be in df"
    assert isinstance(save_path, str), "save_path should be a string"

    save_path = os.path.join(save_path, f"{feature}_histogram.png")
    # if already image already exists return
    if os.path.exists(save_path):
        logging.info(f"Image already exists at {save_path}")
        return

    # Create a figure
    plt.figure(figsize=figsize)
    df[feature].hist()
    # Add axis labels
    plt.xlabel(f"{feature}")
    plt.ylabel('Count')
    # Add a title
    plt.title(f'{feature} Distribution')
    # Add a legend
    plt.legend([feature])
    # Add grid lines
    plt.grid(True)
    # Customize the tick labels
    plt.xticks(rotation=45)
    # Save the plot to a file
    plt.savefig(save_path)
    logging.info(f"Saved plot to {save_path}")


def save_describtion(df: pd.DataFrame, save_path: str, file_name: str) -> None:
    """ Save descriptive statistics to image file
    Args:
        df (pd.DataFrame): Pandas DataFrame
        save_path (str): Path to save plot
        file_name (str): Name of file
    """
    # some assertions
    assert isinstance(df, pd.DataFrame), "df should be a Pandas DataFrame"
    assert isinstance(save_path, str), "save_path should be a string"
    assert isinstance(file_name, str), "file_name should be a string"

    # drop column
    # import pdb; pdb.set_trace()
    drop_features = [
        f'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_{i}' for i in range(
            1,
            3)]
    for feature in drop_features:
        if feature in df.columns:
            df = df.drop(columns=[feature])
            logging.info(f"Dropped {feature} from df")
        else:
            logging.info(f"{feature} not in df")

    save_path = os.path.join(save_path, 'Descriptive_statistics.png')
    # if already image already exists return
    if os.path.exists(save_path):
        logging.info(f"Image already exists at {save_path}")
        return

    desc_stats = df.describe().round(2)

    fig, ax = plt.subplots()
    # Adjust bbox values as needed to center the table
    # Create the table and set cell padding
    table = plt.table(cellText=desc_stats.values.T,
                      colLabels=desc_stats.index,
                      rowLabels=desc_stats.columns,
                      cellLoc='center',
                      loc='center',
                      )

    table.auto_set_column_width(col=list(range(len(desc_stats.columns))))
    ax.axis('off')
    plt.savefig(save_path, bbox_inches='tight')
    logging.info(f"Saved plot to {save_path}")


def save_statistics(
    df: pd.DataFrame,
    column: str,
    save_path: str,
    size: tuple = (
        20,
        10)) -> None:
    """ Plot statistics for a column
    Args:
        df (pd.DataFrame): Pandas DataFrame
        column (str): Column to plot
        save_path (str): Path to save plot
        size (tuple): Size of plot
    Returns:
        None
    """
    # some assertions
    assert isinstance(df, pd.DataFrame), "df should be a Pandas DataFrame"
    assert isinstance(column, str), "column should be a string"
    assert column in df.columns, "column should be in df"
    assert isinstance(size, tuple), "size should be a tuple"
    save_path = os.path.join(save_path, f'{column}.png')
    plt.figure(figsize=size)
    df[column].value_counts('normalize').plot(kind='bar')
    plt.title(f"Statistics for {column}")
    plt.ylabel('Frequency')
    plt.xlabel(f"{column}")
    plt.savefig(save_path, bbox_inches='tight')
    logging.info(f"Saved plot to {save_path}")


def save_histplot(
    df: pd.DataFrame,
    column: str = 'Total_Trans_Ct',
    save_path: str = "default",
    size: tuple = (
        20,
        10)) -> None:
    """ Plot statistics for a column
    Args:
        df (pd.DataFrame): Pandas DataFrame
        column (str): Column to plot
        save_path (str): Path to save plot
        size (tuple): Size of plot
    Returns:
        None
    """
    # some assertions
    assert isinstance(df, pd.DataFrame), "df should be a Pandas DataFrame"
    assert isinstance(column, str), "column should be a string"
    assert column in df.columns, "column should be in df"
    assert isinstance(size, tuple), "size should be a tuple"
    assert isinstance(save_path, str), "save_path should be a string"
    if save_path is "default":
        save_path = os.path.join(os.getcwd(), 'images')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        logging.info("Set save_path its default value")
        logging.info(f"save_path set to {save_path}")
    save_path = os.path.join(save_path, f'{column}.png')
    plt.figure(figsize=size)
    # Show distributions of 'Total_Trans_Ct' and add a smooth curve obtained
    # using a kernel density estimate
    sns.histplot(df[column], stat='density', kde=True)
    plt.title(f"Distribution of {column}")
    plt.ylabel('Frequency')
    plt.xlabel(f"{column}")
    plt.savefig(save_path, bbox_inches='tight')
    logging.info(f"Saved plot to {save_path}")


def save_correlation_matrix(
        df: pd.DataFrame,
        save_path: str = "default",
        size: tuple = (
            20,
            10),
        max_name_length: int = 30) -> None:
    """ Plot statistics for a column
    Args:
        df (pd.DataFrame): Pandas DataFrame
        save_path (str): Path to save plot
        size (tuple): Size of plot
        max_name_length (int): Maximum length of column name
    Returns:
        None
    """
    # some assertions
    assert isinstance(df, pd.DataFrame), "df should be a Pandas DataFrame"
    assert isinstance(size, tuple), "size should be a tuple"
    assert isinstance(save_path, str), "save_path should be a string"
    assert isinstance(
        max_name_length, int), "max_name_length should be an integer"
    if save_path is "default":
        save_path = os.path.join(os.getcwd(), 'images')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        logging.info("Set save_path its default value")
        logging.info(f"save_path set to {save_path}")
    save_path = os.path.join(save_path, 'Correlation_matrix.png')
    # rename to long column names
    df.rename(columns=lambda x: x if len(x) <=
              max_name_length else x[:max_name_length], inplace=True)
    plt.figure(figsize=size)
    sns.heatmap(df.corr(), annot=True, cmap='Dark2_r', linewidths=2)
    plt.savefig(save_path, bbox_inches='tight')


def wandb_track_images(run: wandb.run, image_path: str = "default") -> None:
    """ Track images with wandb
    Args:
        run (wandb.run): Wandb run
        image_path (str): Path to save plot
    Returns: None
    """
    # some assertions
    assert isinstance(
        run, wandb.sdk.wandb_run.Run), "run should be a wandb.run"
    assert isinstance(image_path, str), "save_path should be a string"
    if image_path is "default":
        logging.info("Set image_path its default value")
        logging.info("Set image_path to images")
        return
    # Get all files in image_path
    image_files = [
        file for file in os.listdir(image_path) if file.endswith(
            (".png", ".jpg", ".jpeg"))]
    # Log all files to wandb
    for image_file in image_files:
        run.log({f"{image_file}": wandb.Image(
            os.path.join(image_path, image_file))})
        logging.info(f"Logged {image_file} to wandb")
    logging.info(f"Logged {len(image_files)} images to wandb")
