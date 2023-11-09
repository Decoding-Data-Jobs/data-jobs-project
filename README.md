<div align="center">

# **Decoding Data Jobs**
##### *Analyzing job postings to find desired skill sets for aspiring data nerds.*

</div>

---
 
Zacharia Schmitz<br>
Joshua Click<br>
October - November 2023<br>

<div align="center">

![No picture yet](/support_files/images/topskills.png)

*Top skills in data job postings*

</div>

--- 

#### **Table of Contents:**

###### *(Jump To)*

[Project Overview](#overview)

[Data Acquisition](#acquire)

[Preparation](#preparing-data)

[Exploration](#exploration-questions)

[Models](#modeling)

[Conclusion](#conclusions)

</p>

---

## **Overview**

1. Decide Source & Scope

2. Acquire Job Postings

3. Data Cleaning

4. Text Preprocessing

5. Feature Extraction

6. Model Training

7. Deliverable / Dashboard (Presenting Model-Validated Findings)


---

## **Project Goal**

**Deliver insights to potential data job applicants.**

--- 

## **Project Description**

**We developed a dynamic tool that empowers aspiring data analysts to navigate the job market with precision and confidence.**

*Our project uses 33,000 job postings that were scraped from Google and then processed with machine learning models to validate the results. We're not just sifting through text; we're decoding the jargon of job descriptions to help future data analysts on the hunt for their dream roles. We provide them with a competitive edge by revealing the skills, qualifications, and trends that matter most.* 

---


<br>

## **Hypotheses**

- Data scientists and engineers have a more technical skillset compared to analysts

- Data scientists and engineers typically make more than analysts

- Data analyst roles oftentimes aren't clearly defined and include more advanced skills like machine learning

<br>


---


<br>

## **Acquire**

We originally intended on pulling all of the data myself using LinkedIn webscraping or another job resource.

We were able to use a scraper for LinkedIn, but after reading into it, they don't like that and have been known to send cease and desist letters.

### **Getting our dataset from Google Jobs**

For decent analysis, we would need a fairly large dataset.

With most job postings not including pay information, this would increase the demand for a large dataset even more.

#### **1. Use their API**

- You'll need an API key from their [dev website](https://developers.google.com/custom-search/v1/overview)

- You can get 100 search queries per day for free.

- At a cost of $5 per 1,000, you can get up to 10,000.

- The downside to using their API, people often complain that their API results, aren't true to what searches are actually returning.

- We also was not able to see if people could use the API for job posting searches.

#### **2. Scrape the normal result pages**

- While Google does not officially allow it, scraping the search engine results page (SERP) is also an option

- Google seems to have very sophisticated technology when it comes to scraping their pages

- If you scrape at a rate higher than 8 keyword requests per hour you risk detection

- If you push it higher than 10 per hour, this will oftentimes get you blocked

- By using multiple IPs you can up the rate (100 IPs = 1,000 requests)

- There is also an [open source search engine scraper](http://scraping.compunect.com) written in PHP, that can manage proxies and other detection avoidance methods

#### **3. Use a Scraping Service**

- There seem to be many services that offer to do the webscraping

- The one that the Kaggle dataset used to scrape was SerpAPI

- Their cost was as low as $50 for 5,000 searches per month or as high as $250 for 30,000 searches a month

---


<br>

### **Data Dictionary:**

<div align="center">

![Alt text](/support_files/images/datadict.png)

### Definitions

| Field Name | Description |
| --- | --- |
| `Unnamed: 0` | Appears to be an auto-incremented identifier. |
| `index` | Another identifier, possibly redundant with "Unnamed: 0". |
| `title` | Job title. |
| `company_name` | Name of the company offering the job. |
| `location` | Location of the job. |
| `via` | Source/platform where the job was posted. |
| `description` | Detailed description of the job. |
| `extensions` | Additional information about the job (e.g., job type, benefits). |
| `job_id` | A unique identifier for the job, possibly encoded. |
| `thumbnail` | URL to a thumbnail image associated with the job/company. |
| `url` | URL for the job posting. |
| `company_description` | Description of the company. |
| `company_rating` | Company's rating. |
| `rating_count` | Number of ratings the company received. |
| `job_type` | Type of the job (e.g., full-time, part-time). |
| `benefits` | List of benefits provided by the company. |
| `posted` | When the job was posted. |
| `deadline` | Application deadline for the job. |
| `employment_type` | Employment type (e.g., full-time, contract). |
| `commute_time` | Information on commute time, if available. |
| `salary_pay` | Salary payment value, if available. |
| `salary_rate` | Salary rate (e.g., per hour, per year), if available. |
| `salary_avg` | Average salary for the job, if available. |
| `salary_min` | Minimum salary for the job, if available. |
| `salary_max` | Maximum salary for the job, if available. |
| `salary_hourly` | Hourly salary, if available. |
| `salary_yearly` | Yearly salary, if available. |
| `salary_standardized` | Standardized salary information, if available. |
| `description_tokens` | List of skills extracted from the job description. |

</div>

</p>

<br>
<br>


---

## **Preparing Data**

- Drop Columns

- Check for Duplicates

- Handling Missing Data

- Work From Home

- Feature Engineering - Standardizing Salary

- Standardize Location Column

- Date Formatting

- Standardize Job Title

- Job Description NLP Processing

- Creation of `description_cleaned`

- Define Keywords for `description_tokenized`

- Schedule Types

![Alt text](/support_files/images/prepped_dict.png)

---
<br>

## **Exploration Questions**

1. What companies have the most job postings?

2. What is the location spread for our dataset?

3. Within the Google Jobs search, which site has the most postings? 

4. What words are most common in data job descriptions

5. What are the overall top skills to learn for data jobs?

6. Do a majority of places allow work from home or want you in the work place?

7. What skills are most prevalent in our postings for programming languages, machine learning methods, tools? 

8. What time of year do we see most data jobs being posted?

9. What are the most desirable skills?



<br>
<br>


---


<br>

## **Modeling**

Utilize GridSearch for best parameters for TF-IDF and LogisticRegression

**Unbalanced DataSet**

```python
# Create a pipeline
pipeline = Pipeline(
    [
        ("tfidf", TfidfVectorizer()),
        ("logreg", LogisticRegression(max_iter=1000, random_state=321)),
    ]
)

param_grid = {
    "logreg__C": [5, 10, 20],
    "logreg__penalty": ["l1", "l2"],
    "tfidf__max_df": [250, 500, 750, 1000],
    "tfidf__max_features": [500, 750, 1000],
    "tfidf__min_df": [50, 100, 150],
    'tfidf__ngram_range': [(1, 1), (1, 2)]
}

```
**Balanced DataSet**

```python
# Create a pipeline
pipeline = Pipeline(
    [
        ("tfidf", TfidfVectorizer()),
        ("logreg", LogisticRegression(max_iter=1000, random_state=321)),
    ]
)

param_grid = {
    "logreg__C": [5, 10, 20],
    "logreg__penalty": ["l1", "l2"],
    "tfidf__max_df": [250, 500, 750, 1000],
    "tfidf__max_features": [500, 750, 1000],
    "tfidf__min_df": [50, 100, 150],
    'tfidf__ngram_range': [(1, 1), (1, 2)]
    }

```



---

### Best GridSearches on Test Set:

**Unbalanced Dataset (92% baseline)**

```python
tfidf = TfidfVectorizer(max_df=1000, max_features=1000, min_df=100, ngram_range=(1, 2))
logit = LogisticRegression(C=5, penalty="l2", random_state=321, max_iter=1000)
```

**Train Set(Mean with 2 cross-validations):96%**

**Test Set(Mean with 2 cross-validations):90%**

---

**Balanced Dataset(33% baseline)**
```python
tfidf = TfidfVectorizer(max_df=1000, max_features=750, min_df=50, ngram_range=(1, 2))
logit = LogisticRegression(C=5, penalty="l2", random_state=321, max_iter=1000)
```

**Train Set(Mean w/ 2 cross-validations):97%**

**Test Set(Mean w/ 2 cross-validations):87%**
<br>
<br>


---


<br>

## **How to Reproduce:**

1. Clone this repo

2. Download CSV into /support_files/
    - [name this "jobs.csv"](https://drive.google.com/file/d/1M5UibWPA48zynbNXbB-ZE9m8LVVn5MRR/view?usp=sharing) - Unprepped .csv (takes 10 minutes to prep)
    - [name this "prepped_jobs.csv"](https://drive.google.com/file/d/1FsXOyZgzOE0AgniY_hdtzTcmOu2KlDni/view?usp=sharing) - Prepped .csv 

2. Run the notebook.

<br>
<br>


---

<br>

## **Conclusions**

### Recommendations

##### - Modeling Takeaways

- We down-sampled our dataset in order to demonstrate an accurate model
    - Another option, with more time, would be to collect more Data Scientist and Engineer positions
    - Data Analyst, will always have more representation than the other two, just due to more Analyst positions<br><br>

- Our model is currently only being used to prove that our analysis of the data job skills are different between the three `titles`

- Further data validation could be performed by including trigrams and quadgrams, but very computationally expensive

##### - Data Collection Takeaways

- Decided to use the 'Full-Time' positions only to deal with outliers
    - Freelance jobs will often pay much more, but don't guarantee employment or have benefits
    - Freelance jobs also are not very applicable to entry level applicants<br><br>

- Trying to categorize by `sector` proved to be too inaccurate from the nature of the descriptions in the job posts
    - Being able to <i>accurately</i> categorize by sector would add value, but would take too much time for this scope<br><br>

- Dataset had very little job positions for engineer/scientists due to the original search term being "Data Analyst"
    - Scraping for all 3 search terms would add insight for the under-represented categories<br><br>

- `location` in the dataset was primarily from one geographic area and did not include positions from the entire U.S.
    - Although the search was for the entire United States, it seems it was limited to a specific region
    - If this was due to IP address, area could be more diversified by using a proxy<br><br>

- `date_posted` provided insights that certain fiscal quarters have increased hiring

- We were able to distiguish skills for each `title` represented in the dataset
    - this was validated by using a classification model to predict the `title`<br><br>

- `salary` was only present in 18% of the job postings. This represents a known issue for job searchers of no salary in the posting

##### - Dashboard & Interactive Plots Takeaways

- Presenting data with an interactive graph can allow for users to answer their own potential questions

- Rather than having scrolls of graphs, it could also be summed up with an interactive graph

##### - Next Steps

**Validation:** 
- Set up a validation framework to periodically test the model on new job scrapings from Google and ensure its predictions remain accurate over time
- If the model suddenly is inaccurate, this could represent a shift in the desired skills over time

**Continuously add data for continued data insights**
- Expand to the entire United States, rather than the limited geographic region
- Continue scraping posts, to potentially identify upward and downward trends in certain skills desirability
