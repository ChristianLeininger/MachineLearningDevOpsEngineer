import os
import pandas as pd
import matplotlib.pyplot as plt
import logging


def get_categorical_columns(df: pd.DataFrame) -> list:
    """ Get list of categorical columns from a Pandas DataFrame

    Args:
        df (pd.DataFrame): Pandas DataFrame
    Returns:
        list: List of categorical columns
    """
    # Select columns of type 'object' and 'category'
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    return list(categorical_columns)



def save_plot(df: pd.DataFrame, feature: str, save_path: str) -> None:
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
    plt.figure(figsize=(20, 10))
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