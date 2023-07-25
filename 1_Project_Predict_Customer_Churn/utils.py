import os
import numpy as np
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
    drop_features = [f'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_{i}' for i in range(1, 3)]
    for feature in drop_features:
        if feature in df.columns:
            df = df.drop(columns=[feature])
            logging.info(f"Dropped {feature} from df")
        else:
            logging.info(f"{feature} not in df")

    
    save_path = os.path.join(save_path, 'descriptive_statistics.png')
    # if already image already exists return
    if os.path.exists(save_path):
        logging.info(f"Image already exists at {save_path}")
        return
    # Get descriptive statistics
    desc_stats = df.describe().round(2)

   
    # import pdb; pdb.set_trace()
    # Split the columns into two sets
    # TODO fix this plots are false
    # Save the tables as images
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









   
    



