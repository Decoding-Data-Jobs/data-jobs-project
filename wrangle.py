# Standard library imports
import os
import string
from collections import Counter
import ast

# Third-party library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import nltk
import ipywidgets as widgets
from IPython.display import display, clear_output
import plotly.express as px

# Specific functions from those libraries
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import MWETokenizer, word_tokenize
from ipywidgets import interact, widgets

# +------------------------------------------+
# |                                          |
# |   D E F I N I N G     K E Y W O R D S    |
# |                                          |
# +------------------------------------------+

# Picked out keywords based on all keywords (only looked words with 100+ occurrences)
keywords_programming = [
    "sql",
    "python",
    "r",
    "c",
    "c#",
    "javascript",
    "js",
    "java",
    "scala",
    "sas",
    "matlab",
    "c++",
    "c/c++",
    "perl",
    "go",
    "typescript",
    "bash",
    "html",
    "css",
    "php",
    "powershell",
    "rust",
    "kotlin",
    "ruby",
    "dart",
    "assembly",
    "swift",
    "vba",
    "lua",
    "groovy",
    "delphi",
    "objective-c",
    "haskell",
    "elixir",
    "julia",
    "clojure",
    "solidity",
    "lisp",
    "f#",
    "fortran",
    "erlang",
    "apl",
    "cobol",
    "ocaml",
    "crystal",
    "javascript/typescript",
    "golang",
    "nosql",
    "mongodb",
    "t-sql",
    "no-sql",
    "visual_basic",
    "pascal",
    "mongo",
    "pl/sql",
    "sass",
    "vb.net",
    "mssql",
]
# Pick out ML Algorithm keywords
keywords_ML_Algorithms = [
    x.lower()
    for x in [
        "regression",
        "clustering",
        "classification",
        "predictive",
        "prediction",
        "decision trees",
        "Decision Trees, Random Forests",
        "Convolutional Neural Networks",
        "CNN",
        "Gradient Boosting Machines (xgboost, lightgbm, etc)",
        "Bayesian Approaches",
        "Dense Neural Networks (MLPs, etc)",
        "DNN",
        "Recurrent Neural Networks",
        "RNN",
        "Transformer Networks (BERT, gpt-3, etc)",
        "Graph Neural Networks",
        "Transformer" "Autoencoder Networks (DAE, VAE, etc)",
        "Generative Adversarial Networks",
        "None",
        "Evolutionary Approaches",
    ]
]
# Viz keywords
keyword_viz = [
    x.lower()
    for x in [
        "Matplotlib",
        "Seaborn",
        "Plotly",
        "Ggplot",
        "None",
        "Shiny",
        "Geoplotlib",
        "Bokeh",
        "D3 js",
        "Other",
        "Leaflet / Folium",
        "Pygal",
        "Altair",
        "Dygraphs",
        "Highcharter",
        "tableau",
        "Microsoft Power BI",
        "Google Data Studio",
        "Amazon QuickSight",
        "Qlik Sense",
        "Other",
        "Microsoft Azure Synapse ",
        "Looker",
        "Alteryx ",
        "SAP Analytics Cloud ",
        "TIBCO Spotfire",
        "Domo",
        "Sisense ",
        "Thoughtspot ",
    ]
]
# Computer vision and nlp
keyword_cvnlp = ["computer vision", "natural language processing"]

# Big data keywords
keyword_big_data = [
    "mysql",
    "postgresql",
    "microsoft sql",
    "sqlite",
    "mongodb",
    "bigquery",
    "oracle database",
    "azure sql",
    "amazon rds",
    "google cloud sql",
    "snowflake",
]
# More big data
keyword_big_data_2 = [
    x.lower()
    for x in [
        "MySQL ",
        "PostgreSQL ",
        "Microsoft SQL Server ",
        "SQLite ",
        "MongoDB ",
        "None",
        "Google Cloud BigQuery ",
        "Oracle Database ",
        "Microsoft Azure SQL Database ",
        "Amazon RDS ",
        "Google Cloud SQL ",
        "Snowflake ",
        "Amazon Redshift ",
        "Amazon DynamoDB ",
        "Other",
        "IBM Db2 ",
    ]
]
# Business Intelligence keywords
keyword_bi = [
    x.lower()
    for x in [
        "tableau",
        "Power BI",
        "Power_bi",
        "Google Data Studio",
        "QuickSight",
        "Qlik Sense",
        "Other",
        "Azure Synapse ",
        "Looker",
        "Alteryx ",
        "SAP Analytics Cloud ",
        "TIBCO Spotfire",
        "Domo",
        "Sisense ",
        "Thoughtspot ",
    ]
]
# More business intelligence
keyword_bi_2 = [
    x.lower()
    for x in [
        "tableau",
        "Microsoft Power BI",
        "Google Data Studio",
        "Amazon QuickSight",
        "Qlik Sense",
        "Other",
        "Microsoft Azure Synapse ",
        "Looker",
        "Alteryx ",
        "SAP Analytics Cloud ",
        "TIBCO Spotfire",
        "Domo",
        "Sisense ",
        "Thoughtspot ",
    ]
]
# Analyst tools
keywords_analyst_tools = [
    "excel",
    "tableau",
    "word",
    "powerpoint",
    "looker",
    "powerbi",
    "outlook",
    "azure",
    "jira",
    "twilio",
    "snowflake",
    "shell",
    "linux",
    "sas",
    "sharepoint",
    "mysql",
    "visio",
    "git",
    "mssql",
    "powerpoints",
    "postgresql",
    "spreadsheets",
    "seaborn",
    "pandas",
    "gdpr",
    "spreadsheet",
    "alteryx",
    "github",
    "postgres",
    "ssis",
    "numpy",
    "power_bi",
    "spss",
    "ssrs",
    "microstrategy",
    "cognos",
    "dax",
    "matplotlib",
    "dplyr",
    "tidyr",
    "ggplot2",
    "plotly",
    "esquisse",
    "rshiny",
    "mlr",
    "docker",
    "linux",
    "jira",
    "hadoop",
    "airflow",
    "redis",
    "graphql",
    "sap",
    "tensorflow",
    "node",
    "asp.net",
    "unix",
    "jquery",
    "pyspark",
    "pytorch",
    "gitlab",
    "selenium",
    "splunk",
    "bitbucket",
    "qlik",
    "terminal",
    "atlassian",
    "unix/linux",
    "linux/unix",
    "ubuntu",
    "nuix",
    "datarobot",
]
# Cloud tools
keywords_cloud_tools = [
    "aws",
    "azure",
    "gcp",
    "snowflake",
    "redshift",
    "bigquery",
    "aurora",
    "amazon",
    "ec2",
    "s3",
]
# Not using
keywords_general_tools = [
    "microsoft",
    "slack",
    "apache",
    "ibm",
    "html5",
    "datadog",
    "bloomberg",
    "ajax",
    "persicope",
    "oracle",
]
# Not using
keywords_general = [
    "coding",
    "server",
    "database",
    "cloud",
    "warehousing",
    "scrum",
    "devops",
    "programming",
    "saas",
    "ci/cd",
    "cicd",
    "ml",
    "data_lake",
    "frontend",
    " front-end",
    "back-end",
    "backend",
    "json",
    "xml",
    "ios",
    "kanban",
    "nlp",
    "iot",
    "codebase",
    "agile/scrum",
    "agile",
    "ai/ml",
    "ai",
    "paas",
    "machine_learning",
    "macros",
    "iaas",
    "fullstack",
    "dataops",
    "scrum/agile",
    "ssas",
    "mlops",
    "debug",
    "etl",
    "a/b",
    "slack",
    "erp",
    "oop",
    "object-oriented",
    "etl/elt",
    "elt",
    "dashboarding",
    "big-data",
    "twilio",
    "ui/ux",
    "ux/ui",
    "vlookup",
    "crossover",
    "data_lake",
    "data_lakes",
    "bi",
]

keywords = (
    keywords_programming
    + keywords_ML_Algorithms
    + keywords_analyst_tools
    + keywords_cloud_tools
)

# +--------------------------+
# |                          |
# |  P R E P A R A T I O N   |
# |                          |
# +--------------------------+


def tokenize_normalize_lemmatize(text):
    # Ensure necessary resources are available
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    # Convert to lowercase, remove punctuation and stopwords, and lemmatize
    tokens = [
        lemmatizer.lemmatize(word.lower())
        for word in tokens
        if word.isalpha() and word.lower() not in stopwords.words("english")
    ]
    return tokens


def process_description(description, keywords):
    """
    This function processes a job description by tokenizing the words, handling multi-word tokenization,
    removing duplicates, filtering for keywords only, and replacing certain keywords.

    Parameters:
    description (str): The job description to process.
    keywords (list): The list of keywords to filter for.

    Returns:
    list: The processed job description as a list of keywords.
    """
    # Convert the description to lowercase
    detail = description.lower()

    # Tokenize the description into individual words
    detail = word_tokenize(detail)

    # Define multi-word tokens
    multi_tokens = [
        ("power", "bi"),
        ("data", "lake"),
        ("data", "lakes"),
        ("machine", "learning"),
        ("objective", "c"),
        ("visual", "basic"),
        ("predictive", "prediction"),
        ("plotly", "express"),
        ("ggplot", "ggplot"),
        ("d3", "js"),
    ]

    # Initialize a multi-word tokenizer with the defined tokens
    tokenizer = MWETokenizer(multi_tokens)

    # Tokenize the description with the multi-word tokenizer
    detail = tokenizer.tokenize(detail)

    # Remove duplicate words
    detail = list(set(detail))

    # Filter the description for the specified keywords
    detail = [word for word in detail if word in keywords]

    # Define tokens to replace
    replace_tokens = {"powerbi": "power_bi", "spreadsheets": "spreadsheet"}

    # Replace the defined tokens in the description
    for key, value in replace_tokens.items():
        detail = [d.replace(key, value) for d in detail]

    # Replace "c/c++" and "c++" with "c++"
    detail = ["c++" if skill in ["c/c++", "c++"] else skill for skill in detail]

    return detail


def prepare_jobs(df, prepped_csv=False):
    """
    1. Selects a subset of columns from the DataFrame.
    2. Drops duplicate rows based on the "job_id" column.
    3. Drops the "job_id" column.
    4. Fills null values in the "work_from_home" column with False.
    5. Creates a new "cleaned_salary" column and performs several transformations on it
        to standardize the salary information.
    6. Extracts the pay rate from the "cleaned_salary" column.
    7. Drops all letters from the "cleaned_salary" column.
    8. Splits the "cleaned_salary" into "min_salary" and "max_salary" columns.
    9. Creates an "avg_salary" column by averaging the "min_salary" and "max_salary" columns.
    10. Adjusts the salary columns based on the pay rate.
    11. Drops the original "salary" column.
    12. Cleans the "location" column and replaces state abbreviations with full names.
    13. Renames the "date_time" column to "date_scraped" and converts it to datetime.
    14. Converts the "posted_at" column to a timedelta.
    15. Creates a new "posting_created" column by subtracting "posted_at" from "date_scraped".
    16. Standardizes the job titles in the "title" column.
    17. Tokenizes the job descriptions and filters for certain keywords.
    18. Creates a new 'sector' column based on certain keywords in the 'description' column.
    19. Filters the DataFrame to only include full-time jobs.

    Parameters:
    df (DataFrame): The input DataFrame.

    Returns:
    A prepped dataframe
    """
    file_path = "./support_files/prepped_jobs.csv"

    if os.path.isfile(file_path):
        # Read in CSV
        jobs_df_cleaned = pd.read_csv(file_path)

        # Convert the strings in 'description_cleaned' and 'description_tokens' back into lists
        jobs_df_cleaned["description_cleaned"] = jobs_df_cleaned[
            "description_cleaned"
        ].apply(ast.literal_eval)
        jobs_df_cleaned["description_tokens"] = jobs_df_cleaned[
            "description_tokens"
        ].apply(ast.literal_eval)

        # Make the index date time
        jobs_df_cleaned.index = pd.to_datetime(jobs_df_cleaned["posting_created"])

        return jobs_df_cleaned
    try:
        jobs_df_cleaned = df[
            [
                "title",
                "company_name",
                "location",
                "via",
                "description",
                "posted_at",
                "schedule_type",
                "work_from_home",
                "salary",
                "job_id",
                "date_time",
                "salary_pay",
                "salary_rate",
            ]
        ]

        # Drop duplicates on the unique identifier
        jobs_df_cleaned = jobs_df_cleaned.drop_duplicates(subset=["job_id"])

        # Drop the column since we're not using it anymore
        jobs_df_cleaned = jobs_df_cleaned.drop(columns=["job_id"])

        # Fill nulls in work from home with False
        jobs_df_cleaned["work_from_home"] = jobs_df_cleaned["work_from_home"].fillna(
            False
        )

        # Create a salary cleaned column out of a copy of salary
        jobs_df_cleaned["cleaned_salary"] = jobs_df_cleaned["salary"]

        # Remove decimals and numeric character until you hit a - or [a-zA-Z]
        jobs_df_cleaned["cleaned_salary"] = jobs_df_cleaned[
            "cleaned_salary"
        ].str.replace(r"\.\d+(?=[a-zA-Z-])", "", regex=True)

        # Replace 'K' or 'k' in the 'cleaned_salary' column with ',000'
        jobs_df_cleaned["cleaned_salary"] = (
            jobs_df_cleaned["cleaned_salary"]
            .str.replace("K", "000", case=False, regex=True)
            .str.replace("k", "000", case=False, regex=True)
        )

        # Remove commas from all entries in the 'cleaned_salary' column
        jobs_df_cleaned["cleaned_salary"] = jobs_df_cleaned[
            "cleaned_salary"
        ].str.replace(",", "", regex=False)

        # Extract pay rate
        jobs_df_cleaned["pay_rate"] = jobs_df_cleaned["cleaned_salary"].str.extract(
            r"(\bhour\b|\bmonth\b|\byear\b)", expand=False
        )

        # Add "ly" to the entire column
        jobs_df_cleaned["pay_rate"] = jobs_df_cleaned["pay_rate"].str.replace(
            r"(\bhour\b|\bmonth\b|\byear\b)", r"\1ly", regex=True
        )

        # Drop all letters from salary cleaned column
        jobs_df_cleaned["cleaned_salary"] = jobs_df_cleaned[
            "cleaned_salary"
        ].str.replace(r"[a-zA-Z]", "", regex=True)

        # Function to get min salary
        def get_min_salary(salary):
            return salary.split("–")[0]

        # Function to get max salary
        def get_max_salary(salary):
            values = salary.split("–")
            if len(values) == 1:
                return values[0]
            return values[1]

        # Make salary cleaned a string
        jobs_df_cleaned["cleaned_salary"] = jobs_df_cleaned["cleaned_salary"].astype(
            str
        )

        # Apply the functions to get min_salary and max_salary columns
        jobs_df_cleaned["min_salary"] = jobs_df_cleaned["cleaned_salary"].apply(
            get_min_salary
        )
        jobs_df_cleaned["max_salary"] = jobs_df_cleaned["cleaned_salary"].apply(
            get_max_salary
        )

        # Make an avg_salary column using the average of min and max
        jobs_df_cleaned["avg_salary"] = (
            jobs_df_cleaned["min_salary"].astype(float)
            + jobs_df_cleaned["max_salary"].astype(float)
        ) / 2

        # If pay rate is hourly, multiply min_salary, max_salary, and avg_salary by 2080
        jobs_df_cleaned.loc[
            jobs_df_cleaned["pay_rate"] == "hourly",
            ["min_salary", "max_salary", "avg_salary"],
        ] = (
            jobs_df_cleaned.loc[
                jobs_df_cleaned["pay_rate"] == "hourly",
                ["min_salary", "max_salary", "avg_salary"],
            ].astype(float)
            * 2080
        )

        # If pay rate is monthly, multiply min_salary, max_salary, and avg_salary by 12
        jobs_df_cleaned.loc[
            jobs_df_cleaned["pay_rate"] == "monthly",
            ["min_salary", "max_salary", "avg_salary"],
        ] = (
            jobs_df_cleaned.loc[
                jobs_df_cleaned["pay_rate"] == "monthly",
                ["min_salary", "max_salary", "avg_salary"],
            ].astype(float)
            * 12
        )

        # Drop the old salary
        jobs_df_cleaned = jobs_df_cleaned.drop(columns=["salary"])

        # Make them floats
        jobs_df_cleaned["min_salary"] = jobs_df_cleaned["min_salary"].astype(float)
        jobs_df_cleaned["max_salary"] = jobs_df_cleaned["max_salary"].astype(float)

        # Replace "nan" in cleaned_salary with nulls
        jobs_df_cleaned["cleaned_salary"] = jobs_df_cleaned["cleaned_salary"].replace(
            "nan", np.nan
        )

        # Make the column a string
        jobs_df_cleaned["location_cleaned"] = jobs_df_cleaned["location"].astype(str)

        # Make a lambda for all the states and apply it to a state column, to reduce location values
        jobs_df_cleaned["location_cleaned"] = jobs_df_cleaned["location_cleaned"].apply(
            lambda x: x.split(",")[1].strip() if "," in x else x
        )

        # Create a dictionary with state abbreviations and full names
        state_dict = {
            "CA": "California",
            "NY": "New York",
            "NJ": "New Jersey",
            "MO": "Missouri",
            "OK": "Oklahoma",
            "KS": "Kansas",
            "AR": "Arkansas",
            "TX": "Texas",
            "MA": "Massachusetts",
            "NE": "Nebraska",
            "PA": "Pennsylvania",
            "DC": "District of Columbia",
            "CT": "Connecticut",
            "NH": "New Hampshire",
        }

        # Replace state abbreviations with full names
        jobs_df_cleaned["location_cleaned"] = jobs_df_cleaned[
            "location_cleaned"
        ].replace(state_dict)

        # If the string has (+X others), change it to "Multiple Locations"
        jobs_df_cleaned["location_cleaned"] = jobs_df_cleaned["location_cleaned"].apply(
            lambda x: "Multiple Locations" if "(" in x else x
        )

        # Remove leading and trailing white space
        jobs_df_cleaned["location_cleaned"] = jobs_df_cleaned[
            "location_cleaned"
        ].str.strip()

        # Replace "nan" with Unkown
        jobs_df_cleaned["location_cleaned"] = jobs_df_cleaned[
            "location_cleaned"
        ].replace("nan", "Unknown")

        # Change date_time to date_scraped
        jobs_df_cleaned.rename(columns={"date_time": "date_scraped"}, inplace=True)

        # Convert "posted_at" to timedelta
        jobs_df_cleaned["posted_at"] = pd.to_timedelta(
            jobs_df_cleaned["posted_at"].str.extract("(\d+)")[0].astype(int), unit="h"
        )

        # Convert "date_scraped" to datetime
        jobs_df_cleaned["date_scraped"] = pd.to_datetime(
            jobs_df_cleaned["date_scraped"]
        )

        # Create "posting_created" column
        jobs_df_cleaned["posting_created"] = (
            jobs_df_cleaned["date_scraped"] - jobs_df_cleaned["posted_at"]
        )

        # Change posting_created to be date time formated with only hours and minutes
        jobs_df_cleaned["posting_created"] = jobs_df_cleaned[
            "posting_created"
        ].dt.strftime("%Y-%m-%d %H:%M")
        # Make the index date time
        jobs_df_cleaned.index = pd.to_datetime(jobs_df_cleaned["posting_created"])

        # Define the mapping of keywords to standardized titles
        title_mapping = {
            "scie": "Data Scientist",
            "eng": "Data Engineer",
            "ana": "Data Analyst",
        }

        # Apply the mapping to the 'title' column
        jobs_df_cleaned["title_cleaned"] = jobs_df_cleaned["title"].apply(
            lambda x: next((v for k, v in title_mapping.items() if k in x.lower()), x)
        )

        # If a title_cleaned isn't one of those 3, make it Other
        jobs_df_cleaned["title_cleaned"] = jobs_df_cleaned["title_cleaned"].apply(
            lambda x: "Other" if x not in title_mapping.values() else x
        )

        jobs_df_cleaned["description_cleaned"] = jobs_df_cleaned["description"].apply(
            tokenize_normalize_lemmatize
        )

        jobs_df_cleaned["description_tokens"] = jobs_df_cleaned["description"].apply(
            lambda x: process_description(x, keywords)
        )

        # If the schedule type does not have "Full-time", drop it
        jobs_df_cleaned = jobs_df_cleaned[
            jobs_df_cleaned["schedule_type"] == "Full-time"
        ]

        # Play a sound when completed
        os.system("afplay /System/Library/Sounds/Ping.aiff")

        return jobs_df_cleaned
    except Exception as e:
        os.system("afplay /System/Library/Sounds/Ping.aiff")
        print("Prep failed.")
        traceback.print_exc()


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


def preprocess_jobs_df(jobs_df):
    import pandas as pd
    import numpy as np
    import nltk
    from nltk.tokenize import word_tokenize, MWETokenizer
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from collections import Counter

    # Ensure necessary resources are available
    nltk.download("punkt")
    nltk.download("stopwords")

    # Define helper functions
    def get_min_salary(salary):
        return salary.split("–")[0]

    def get_max_salary(salary):
        values = salary.split("–")
        if len(values) == 1:
            return values[0]
        return values[1]

    def standardize_titles(df):
        title_mapping = {
            "scie": "Data Scientist",
            "eng": "Data Engineer",
            "ana": "Data Analyst",
        }
        df["title_cleaned"] = df["title"].apply(
            lambda x: next((v for k, v in title_mapping.items() if k in x.lower()), x)
        )
        df["title_cleaned"] = df["title_cleaned"].apply(
            lambda x: "Other" if x not in title_mapping.values() else x
        )
        return df

    def tokenize_normalize_lemmatize(text):
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(text)
        tokens = [
            lemmatizer.lemmatize(word.lower())
            for word in tokens
            if word.isalpha() and word.lower() not in stopwords.words("english")
        ]
        return tokens

    def process_description(description, keywords):
        detail = description.lower()
        detail = word_tokenize(detail)
        multi_tokens = [
            ("power", "bi"),
            ("data", "lake"),
            ("data", "lakes"),
            ("machine", "learning"),
            ("objective", "c"),
            ("visual", "basic"),
            ("predictive", "prediction"),
            ("plotly", "express"),
            ("ggplot", "ggplot"),
            ("d3", "js"),
        ]
        tokenizer = MWETokenizer(multi_tokens)
        detail = tokenizer.tokenize(detail)
        detail = list(set(detail))
        detail = [word for word in detail if word in keywords]
        replace_tokens = {"powerbi": "power_bi", "spreadsheets": "spreadsheet"}
        for key, value in replace_tokens.items():
            detail = [d.replace(key, value) for d in detail]
        detail = ["c++" if skill in ["c/c++", "c++"] else skill for skill in detail]
        return detail

    def create_sector_column(df):
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
        df["sector"] = df["description_cleaned"].apply(
            lambda x: next(
                (v for k, v in sector_mapping.items() if k in " ".join(x).lower()),
                "Other",
            )
        )
        return df

        # Start preprocessing

    jobs_df_cleaned = jobs_df[
        [
            "title",
            "company_name",
            "location",
            "via",
            "description",
            "posted_at",
            "schedule_type",
            "work_from_home",
            "salary",
            "job_id",
            "date_time",
            "salary_pay",
            "salary_rate",
        ]
    ]
    jobs_df_cleaned = jobs_df_cleaned.drop_duplicates(subset=["job_id"])
    jobs_df_cleaned = jobs_df_cleaned.drop(columns=["job_id"])
    jobs_df_cleaned["work_from_home"] = jobs_df_cleaned["work_from_home"].fillna(False)
    jobs_df_cleaned["cleaned_salary"] = jobs_df_cleaned["salary"]
    jobs_df_cleaned["cleaned_salary"] = jobs_df_cleaned["cleaned_salary"].str.replace(
        r"\.\d+(?=[a-zA-Z-])", "", regex=True
    )
    jobs_df_cleaned["cleaned_salary"] = (
        jobs_df_cleaned["cleaned_salary"]
        .str.replace("K", "000", case=False, regex=True)
        .str.replace("k", "000", case=False, regex=True)
    )
    jobs_df_cleaned["cleaned_salary"] = jobs_df_cleaned["cleaned_salary"].str.replace(
        ",", "", regex=False
    )
    jobs_df_cleaned["pay_rate"] = jobs_df_cleaned["cleaned_salary"].str.extract(
        r"(\bhour\b|\bday\b|\bweek\b|\bmonth\b|\byear\b)", expand=False
    )
    jobs_df_cleaned["min_salary"] = jobs_df_cleaned["cleaned_salary"].apply(
        get_min_salary
    )
    jobs_df_cleaned["max_salary"] = jobs_df_cleaned["cleaned_salary"].apply(
        get_max_salary
    )
    jobs_df_cleaned = standardize_titles(jobs_df_cleaned)
    jobs_df_cleaned["description_cleaned"] = jobs_df_cleaned["description"].apply(
        tokenize_normalize_lemmatize
    )
    jobs_df_cleaned = create_sector_column(jobs_df_cleaned)

    # Drop columns we won't be using
    jobs_df_cleaned = jobs_df_cleaned[
        [
            # 'title',
            "company_name",
            # 'location',
            "via",
            # 'description',
            "posted_at",
            "schedule_type",
            "work_from_home",
            # 'date_scraped',
            # 'salary_pay',
            # 'salary_rate',
            # 'cleaned_salary',
            # 'pay_rate',
            # 'min_salary',
            # 'max_salary',
            "avg_salary",
            "location_cleaned",
            "posting_created",
            "title_cleaned",
            "description_cleaned",
            "description_tokens",
        ]
    ]

    return jobs_df_cleaned


# +--------------------------+
# |                          |
# |  E X P L O R A T I O N   |
# |                          |
# +--------------------------+


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


def create_skill_plot(top_skills_df):
    # Create a dropdown selector for the skill category
    skill_list_dropdown = widgets.Dropdown(
        options=[
            ("All Skills", keywords),
            ("Programming Languages", keywords_programming),
            ("ML Algorithms", keywords_ML_Algorithms),
            ("Visualization Tools", keyword_viz + keyword_bi + keyword_bi_2),
            ("Big Data & Cloud", keywords_cloud_tools + keyword_big_data),
        ],
        value=keywords,
        description="Skill Category:",
    )

    # Create a slider to select the number of skills to plot
    num_skills_slider = widgets.IntSlider(
        value=10,
        min=1,
        max=50,
        step=1,
        description="Number of Skills:",
    )

    plot_button = widgets.Button(description="Plot")
    reset_button = widgets.Button(description="Reset")

    def plot_skills_data():
        plot_button.on_click(update_plot)
        reset_button.on_click(reset_selection)

        display(skill_list_dropdown, num_skills_slider, plot_button)

    def update_plot(b):
        clear_output(wait=True)
        # Filter DataFrame based on selected skills
        df = top_skills_df[top_skills_df["skill"].isin(skill_list_dropdown.value)]

        # Select top N skills based on frequency
        df = df.nlargest(num_skills_slider.value, "frequency (%)")

        fig = px.bar(
            df,
            x="skill",
            y="frequency (%)",
            color="avg_yearly_salary",
            color_continuous_scale="Blues",
        )

        fig.update_traces(
            textfont_size=20,
            hovertemplate="""
        <b>Skill:</b> %{x} <br> \
        <b>Frequency:</b> %{y}% <br> \
        <b>Average Salary:</b> $%{marker.color}<extra></extra>""",
            hoverlabel=dict(font_size=20),
        )

        fig.update_layout(
            title_text="Data Jobs - Salaries and Popularity<br><i style='font-size: 15px;'>Hover for details</i>",
            coloraxis_colorbar=dict(title="Average Annual Salary"),
            height=400,
            width=1200,
        )

        fig.show()
        display(reset_button)

    def reset_selection(b):
        clear_output(wait=True)
        plot_skills_data()

    plot_skills_data()


def create_interactive_salary_plot(jobs_df_cleaned, top_skills_df):
    def interactive_skill_salary(b):
        clear_output(wait=True)
        skill = skill_selector.value
        if skill is None:  # Don't plot if skill is None
            return None
        # Filter the dataframe for jobs that mention the selected skill and have a salary
        skill_df = jobs_df_cleaned[
            jobs_df_cleaned["description_tokens"].apply(lambda x: skill in x)
            & jobs_df_cleaned["avg_salary"].notna()
        ].sort_values(by="avg_salary")

        # Reset the index
        skill_df = skill_df.reset_index(drop=True)

        total_df = jobs_df_cleaned[
            jobs_df_cleaned["description_tokens"].apply(lambda x: skill in x)
        ]

        # Calculate the average salary for the selected skill
        avg_salary = skill_df["avg_salary"].mean()

        # Create a bar plot
        fig = px.bar(
            skill_df,
            x=skill_df.index,
            y="avg_salary",
            hover_data=[
                "avg_salary",
                "description_tokens",
                "company_name",
            ],  # Include 'company_name' in the hover data
            labels={"avg_salary": "Salary ($)", "index": "Jobs WITH Salary"},
            title=f"Salary Distribution for Skill: {skill}",
        )

        # Format hover data
        fig.update_traces(
            hovertemplate="<b>%{customdata[1]}</b><br><br>"
            + "<b>Avg Salary:</b> %{y:$,.2f}<br>"
            + "<b>Skills:</b> %{customdata[0]}<br>"
            + "<extra></extra>"
        )

        # Add a line for the average salary
        fig.add_shape(
            type="line",
            line=dict(dash="dash"),
            y0=avg_salary,
            y1=avg_salary,
            x0=0,
            x1=1,
            xref="paper",
            yref="y",
        )

        # Add a text label for the average salary
        fig.add_annotation(
            y=avg_salary + 15000,
            x=0,
            xref="paper",
            yref="y",
            text=f"Average Salary: ${avg_salary:.2f}",
            showarrow=False,
            font=dict(size=20, color="red"),
        )

        # Update layout
        fig.update_layout(
            autosize=True,
            hovermode="closest",
            showlegend=False,
            plot_bgcolor="white",
            yaxis=dict(title="Salary ($)"),
            xaxis=dict(title="Jobs WITH Salary"),
        )

        fig.show()
        display(reset_button)

    def reset_selection(b):
        clear_output(wait=True)
        display(skill_selector, plot_button)

    # Get the top skills sorted alphabetically
    sorted_skills = sorted(top_skills_df["skill"].unique())

    # Interactive dropdown widget
    skill_selector = widgets.Dropdown(options=sorted_skills, description="Skill:")

    plot_button = widgets.Button(description="Plot")
    plot_button.on_click(interactive_skill_salary)

    reset_button = widgets.Button(description="Reset")
    reset_button.on_click(reset_selection)

    display(skill_selector, plot_button)


def create_skill_postings_plot(jobs_df_cleaned):  # , get_top_skills):
    def plot_top_skills(df, qty, title_cleaned=None):
        # Filter the dataframe based on the selected job title
        if title_cleaned:
            df = df[df["title_cleaned"] == title_cleaned]

        # Get the top skills
        top_skills_df = get_top_skills(df, qty)

        # Calculate the ratio of postings the word shows up in and convert to percentage
        top_skills_df["number_of_postings"] = (
            top_skills_df["number_of_postings"] / len(df)
        ) * 100

        # Sort by number of postings
        top_skills_df.sort_values(
            by="number_of_postings", ascending=False, inplace=True
        )

        # Plot it
        fig = px.bar(
            top_skills_df,
            x="skill",
            y="number_of_postings",
            color="avg_yearly_salary",
            color_continuous_scale="Blues",
        )

        fig.update_traces(
            textfont_size=40,
            hovertemplate="""<b>Skill:</b> %{x}<br><b>Postings:</b> %{y}%<br><b>Average Annual Salary:</b> $%{marker.color}<extra></extra>""",
            hoverlabel=dict(font_size=20),
        )

        fig.update_layout(
            title_text=f"<b style='font-size: 30px;'>{title_cleaned if title_cleaned else 'All Jobs'} Skills</b><br><i style='font-size: 20px;'>Salaries and Popularity</i>",
            title_x=0.1,
            font_color="black",
            coloraxis_colorbar=dict(title="Average Annual Salary"),
            yaxis=dict(title="Percentage of Postings"),
        )
        fig.show()

        # Display the 'Go Back' button after the plot
        display(go_back_button)

    def on_update_button_clicked(b):
        qty = qty_slider.value
        title = title_dropdown.value
        if title == "All Data Jobs":
            title = None
        clear_output(wait=True)
        plot_top_skills(jobs_df_cleaned, qty, title)

    def on_go_back_button_clicked(b):
        clear_output(wait=True)
        display(qty_slider, title_dropdown, update_button)

    # Get unique job titles excluding 'Other'
    job_titles = [
        title for title in jobs_df_cleaned["title_cleaned"].unique() if title != "Other"
    ]

    qty_slider = widgets.IntSlider(
        min=1, max=50, step=1, value=10, description="Top skills:"
    )

    # Create the dropdown widget
    title_dropdown = widgets.Dropdown(
        options=["All Data Jobs"] + job_titles, description="Job title:"
    )
    update_button = widgets.Button(description="Update Plot")
    update_button.on_click(on_update_button_clicked)

    go_back_button = widgets.Button(description="Go Back")
    go_back_button.on_click(on_go_back_button_clicked)

    display(qty_slider, title_dropdown, update_button)


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

    display(df[column].value_counts(dropna=True).head(topn))

    # Display the plot
    plt.show()
