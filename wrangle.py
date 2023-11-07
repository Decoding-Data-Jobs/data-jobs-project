# IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import nltk
from nltk.tokenize import word_tokenize, MWETokenizer
from nltk.corpus import stopwords
import string
from collections import Counter
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import matplotlib.dates as mdates
import ipywidgets as widgets
from IPython.display import display, clear_output
from ipywidgets import interact
import plotly.express as px

# Ensure necessary resources are available
nltk.download("punkt")
nltk.download("stopwords")




# ----------------------------------- #
# ACQUIRE
# ----------------------------------- #

def check_columns(DataFrame, reports=False, graphs=False, dates=False):
    """
    This function takes a pandas dataframe as input and returns
    a dataframe with information about each column in the dataframe.
    """

    dataframeinfo = []

    # Check information about the index
    index_dtype = DataFrame.index.dtype
    index_unique_vals = DataFrame.index.unique()
    index_num_unique = DataFrame.index.nunique()
    index_num_null = DataFrame.index.isna().sum()
    index_pct_null = index_num_null / len(DataFrame.index)

    if pd.api.types.is_numeric_dtype(index_dtype) and not isinstance(
        DataFrame.index, pd.RangeIndex
    ):
        index_min_val = DataFrame.index.min()
        index_max_val = DataFrame.index.max()
        index_range_vals = (index_min_val, index_max_val)
    elif pd.api.types.is_datetime64_any_dtype(index_dtype):
        index_min_val = DataFrame.index.min()
        index_max_val = DataFrame.index.max()
        index_range_vals = (
            index_min_val.strftime("%Y-%m-%d"),
            index_max_val.strftime("%Y-%m-%d"),
        )

        # Check for missing dates in the index if dates kwarg is True
        if dates:
            full_date_range = pd.date_range(
                start=index_min_val, end=index_max_val, freq="D"
            )
            missing_dates = full_date_range.difference(DataFrame.index)
            if not missing_dates.empty:
                print(
                    f"Missing dates in index: ({len(missing_dates)} Total) {missing_dates.tolist()}"
                )
    else:
        index_range_vals = None

    dataframeinfo.append(
        [
            "index",
            index_dtype,
            index_num_unique,
            index_num_null,
            index_pct_null,
            index_unique_vals,
            index_range_vals,
        ]
    )

    print(f"Total rows: {DataFrame.shape[0]}")
    print(f"Total columns: {DataFrame.shape[1]}")

    if reports:
        describe = DataFrame.describe().round(2)
        print(describe)

    if graphs:
        DataFrame.hist(figsize=(10, 10))
        plt.subplots_adjust(hspace=0.5)
        plt.show()

    for column in DataFrame.columns:
        dtype = DataFrame[column].dtype
        num_null = DataFrame[column].isna().sum()
        pct_null = DataFrame[column].isna().mean().round(5)

        try:
            unique_vals = DataFrame[column].unique()
            num_unique = DataFrame[column].nunique()
        except TypeError:
            unique_vals = "Column contains multiple lists"
            num_unique = "ERROR"

        if pd.api.types.is_numeric_dtype(dtype):
            min_val = DataFrame[column].min()
            max_val = DataFrame[column].max()
            mean_val = DataFrame[column].mean()
            range_vals = (min_val, max_val, mean_val)
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            min_val = DataFrame[column].min()
            max_val = DataFrame[column].max()
            range_vals = (min_val.strftime("%Y-%m-%d"), max_val.strftime("%Y-%m-%d"))

            if dates:
                full_date_range_col = pd.date_range(
                    start=min_val, end=max_val, freq="D"
                )
                missing_dates_col = full_date_range_col.difference(DataFrame[column])
                if not missing_dates_col.empty:
                    print(
                        f"Missing dates in column '{column}': ({len(missing_dates_col)} Total) {missing_dates_col.tolist()}"
                    )
                else:
                    print(f"No missing dates in column '{column}'")

        else:
            range_vals = None

        dataframeinfo.append(
            [column, dtype, num_unique, num_null, pct_null, unique_vals, range_vals]
        )

    return pd.DataFrame(
        dataframeinfo,
        columns=[
            "col_name",
            "dtype",
            "num_unique",
            "num_null",
            "pct_null",
            "unique_values",
            "range (min, max, mean)",
        ],
    )


# ----------------------------------- #
# PREPARE
# ----------------------------------- #


def standardize_titles(df):
    """
    This function takes a DataFrame as a parameter.
    It standardizes the job titles in the 'title' column based on certain keywords.

    Parameters:
    df (DataFrame): The input DataFrame.

    Returns:
    DataFrame: The DataFrame with standardized job titles.
    """

    # Define the mapping of keywords to standardized titles
    title_mapping = {
        "scie": "Data Scientist",
        "eng": "Data Engineer",
        "ana": "Data Analyst",
    }

    # Apply the mapping to the 'title' column
    df["title_cleaned"] = df["title"].apply(
        lambda x: next((v for k, v in title_mapping.items() if k in x.lower()), x)
    )

    # If a title_cleaned isn't one of those 3, make it Other
    df["title_cleaned"] = df["title_cleaned"].apply(
        lambda x: "Other" if x not in title_mapping.values() else x
    )

    return df

# Tokenize, normalize and lemmatize the descriptions
def tokenize_normalize_lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    # Convert to lowercase, remove punctuation and stopwords, and lemmatize
    tokens = [
        lemmatizer.lemmatize(word.lower())
        for word in tokens
        if word.isalpha() and word.lower() not in stopwords.words("english")
    ]
    return tokens

def create_sector_column(df):
    """
    This function takes a DataFrame as a parameter.
    It creates a new 'sector' column in the DataFrame based on certain keywords in the 'description' column.

    Parameters:
    df (DataFrame): The input DataFrame.

    Returns:
    DataFrame: The DataFrame with the new 'sector' column.
    """

    # Define the mapping of keywords to sectors
    sector_mapping = {
        "finance": "Finance",
        "business": "Business",
        "healthcare": "Healthcare",
        "patient": "Healthcare",
        "technology": "Technology",
        "education": "Education",
        "retail": "Retail",
        "property": "Real Estate",
    }

    # Apply the mapping to the 'description' column
    df["sector"] = df["description_cleaned"].apply(
        lambda x: next(
            (v for k, v in sector_mapping.items() if k in " ".join(x).lower()), "Other"
        )
    )

    return df

# ----------------------------------- #
# EXPLORE
# ----------------------------------- #

def eda_plot(df, column, topn=20):
    """
    This function takes a DataFrame, a column name, and a number as parameters.
    It creates a bar plot of the top 'n' most frequent values in the specified column.

    Parameters:
    df (DataFrame): The input DataFrame.
    column (str): The name of the column to plot.
    topn (int): The number of top values to include in the plot.

    Returns:
    None
    """

    # Create a figure
    plt.figure(figsize=(12, 4))

    # Get the top 'n' most frequent values in the column
    top_values = df[column].value_counts().nlargest(topn)

    # Create a bar plot of the top values
    # Use the 'hsv' colormap to get different colors for each bar
    sns.barplot(x=top_values.index, y=top_values.values, palette="hsv")

    # Set the title and labels
    plt.title(f"'{column}' column value counts", fontsize=20)
    plt.ylabel("Counts", fontsize=15)
    plt.xlabel(column, fontsize=15)

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")

    display(jobs_df_cleaned[column].value_counts(dropna=True).head(topn))

    # Display the plot
    plt.show()

def plot_most_common_words(df, column_name, n):
    """
    Get the n most common words from a specific column of a dataframe and plot them.

    Parameters:
    - df (DataFrame): The dataframe containing the data.
    - column_name (str): The name of the column containing lists of words.
    - n (int): The number of most common words to retrieve.
    """

    # Ensure the column is interpreted as lists, but only if the value is a string
    df[column_name] = df[column_name].apply(
        lambda x: eval(x) if isinstance(x, str) else x
    )

    # Get the frequency of each word in each row
    word_frequency = df[column_name].apply(
        lambda words_list: {word: 1 for word in set(words_list)}
    )

    # Combine all the dictionaries into one big dictionary
    combined_frequency = Counter({key: 0 for key in set().union(*word_frequency)})
    for freq in word_frequency:
        combined_frequency.update(freq)

    # Sort the dictionary by values in descending order and get the top n words
    most_common_words = combined_frequency.most_common(n)

    # Convert to DataFrame
    result_df = pd.DataFrame(most_common_words, columns=["word", "count"])

    # Sort by highest occurence first
    result_df = result_df.sort_values(by="count", ascending=True)

    # Calculate the frequency
    total_rows = df.shape[0]
    result_df["frequency"] = result_df["count"] / total_rows

    # Generate a list of colors
    colors = sns.color_palette("hsv", len(result_df))

    # Plot common words
    plt.figure(figsize=(15, 10))
    plt.barh(result_df["word"], result_df["count"], color=colors)
    plt.title("Most Common Words in Job Descriptions")
    plt.xlabel("Count")
    plt.ylabel("Word")
    plt.show()

def filtered_keywords(jobs_filtered, keywords, title=None, head=10):
    # get keywords in a column
    count_keywords = (
        pd.DataFrame(jobs_filtered.description_tokens.sum())
        .value_counts()
        .rename_axis("keywords")
        .reset_index(name="counts")
    )

    # get frequency of occurence of word (as word only appears once per line)
    length = len(jobs_filtered)  # number of job postings
    count_keywords["percentage"] = 100 * count_keywords.counts / length

    # plot the results
    count_keywords = count_keywords[count_keywords.keywords.isin(keywords)]
    count_keywords = count_keywords.head(head)
    g = plt.bar(
        x="keywords",
        height="percentage",
        data=count_keywords,
        color=np.random.rand(len(count_keywords.keywords), 3),
    )
    plt.xlabel("")
    plt.ylabel("Likelyhood to be in job posting (%)")
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Top {head} {title} for Data Jobs")
    plt.show(g)

def plot_monthly_postings(df):
    """
    Plot the number of postings per month based on the DataFrame's datetime index.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    """

    # Resample the datetime index to monthly frequency and count the number of postings
    monthly_postings = df.resample("M").size()

    # Change the format of monthly_postings index to be MMM-YYYY
    monthly_postings.index = monthly_postings.index.strftime("%b-%Y")

    # Drop November 2023
    monthly_postings = monthly_postings.drop("Nov-2023")

    # Plot the monthly postings
    fig, ax = plt.subplots(figsize=(12, 4))
    monthly_postings.plot(kind="bar", ax=ax)
    ax.set_title("Value counts by month")
    ax.set_ylabel("Counts")

    plt.xticks(rotation=45, ha="right")
    plt.show()

def get_top_skills(df, qty):
    """
    This function takes a DataFrame and a quantity as parameters.
    It returns a DataFrame with the top skills by number of postings, their frequency, and their average yearly salary.
    The quantity parameter determines the number of top skills to return.

    Parameters:
    df (DataFrame): The input DataFrame.
    qty (int): The number of top skills to return.

    Returns:
    DataFrame: A DataFrame with the top skills, their number of postings, their frequency, and their average yearly salary.
    """

    # Initialize an empty dictionary to store the skills and their counts
    skills_counts = {}

    # Loop over the values in the 'description_tokens' column
    for val in df.description_tokens.values:
        # Convert the list of skills in each posting to a set to remove duplicates
        unique_skills = set(val)
        # Increment the count for each unique skill
        for skill in unique_skills:
            if skill in skills_counts:
                skills_counts[skill] += 1
            else:
                skills_counts[skill] = 1

    # Get the top skills and their counts
    top_skill_count = sorted(skills_counts.items(), key=lambda x: -x[1])[:qty]

    # Separate the skills and counts into two lists
    top_skills = list(map(lambda x: x[0], top_skill_count))
    top_counts = list(map(lambda x: x[1], top_skill_count))

    # Initialize an empty list to store the average salaries
    salaries = []

    # Loop over the top skills and calculate their average salary
    for skill in top_skills:
        skill_df = df[df.description_tokens.apply(lambda x: skill in x)]
        if skill_df.avg_salary.isna().all():  # If all salaries for this skill are NaN
            salaries.append(np.nan)  # Append NaN to the salaries list
        else:
            salaries.append(skill_df.avg_salary.mean())

    # Create a DataFrame with the top skills, their number of postings, and their average yearly salary
    top_skills_df = pd.DataFrame(
        {
            "skill": top_skills,
            "number_of_postings": top_counts,
            "avg_yearly_salary": [
                round(s) if s == s else np.nan for s in salaries
            ],  # Only round the salary if it is not NaN
        }
    )

    # Calculate the frequency of each skill
    top_skills_df["frequency (%)"] = round(
        (top_skills_df["number_of_postings"] / df.shape[0]) * 100, 2
    )

    # Sort the DataFrame by average yearly salary in descending order
    top_skills_df = top_skills_df.sort_values("number_of_postings", ascending=False)

    # Remove 'none'
    top_skills_df = top_skills_df[top_skills_df["skill"] != "none"]

    # Round values to 2 decimals
    top_skills_df["avg_yearly_salary"] = top_skills_df["avg_yearly_salary"].round()

    return top_skills_df


# ----------------------------------- #
# MODEL
# ----------------------------------- #
