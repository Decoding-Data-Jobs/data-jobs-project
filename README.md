<p style="text-align: center"> 

# **Decoding the Data Job Market: An Analysis of Data Job Postings**
</p>

---
 
Zacharia Schmitz<br>
Joshua Click<br>
24 October 2023<br>

<p style="text-align: center;"> 

![Alt text](images/output.png)

*Corpus Wordcloud*

--- 


#### Project Plan:

*(Jump To)*

[Project Overview](#overview)

[Data Acquisition](#acquire)

[Preparation](#preparation-data)

[Exploration](#explore)

[Models](#modeling)

[Conclusion](#conclusions)

</p>

---

<br>

## Overview



##### We downloaded the data from Kaggle as a .csv

1. Decide how to clean / prep the data.

2. Text Preprocessing

2. Explore Separately to see the data from different angles.

3. Feature Extraction

4. Model Training

5. Reusable Functioinality in the Form of a Dashboard

---

### Project Goal

The goal is to analyze the Data Analyst/Engineer jobs from various recruiting websites and see if we can find what the most requested skills are and which ones are the most highly compensated skills.

## Plan → Acquire → Prepare → Explore → Model → Deliver

--- 

### Project Description

- NLP: To find commonalities in posts.

- Classification: To predict the location, job type, etc. based on other features

- Regression: To predict pay based on location/skills

- Dashboard: Possibly include drop down boxes for skill/locations/job titles to demonstrate full stack data science capabilities

<br>


---


<br>

### Initial Questions / Hypotheses

1. What skills are most common between job postings?

2. Which skills pay the most?

3. What are the most common locations for work?

4. 

<br>

---


<br>

## Acquire

Data was downloaded from Kaggle as a .csv

Link: https://www.kaggle.com/datasets/lukebarousse/data-analyst-job-postings-google-search/data?select=gsearch_jobs.csv

**Parameters:**

- 

- 

- 

- 

- 


<br>
<br>

---


<br>

### Pre-Modeling Data Dictionary:

<p style="text-align: center;"> 

![Alt text](images/df.png)


### Corpus Information

| Column | Definition |
|--------|-----------|
|`language`|The actual programming language of the repository on GitHub|
|`repo`|The repository location (*h<span>ttps://github.com + **repo** + /blob/master/README.md*)|
|`readme`|The repository prior to the text being processed|
|`preprocessed_readme`|The repository after the text is processed|

 </p>

<br>
<br>

---

## Preparing Data

* 

* 

* 

* 

* 

* 

* 

* 

* 

* 

* 

* 
---
<br>

## Explore

1. 

2. 

3. 

4. 

<br>
<br>

---


<br>

## Modeling

Utilize GridSearch for best parameters for TF-IDF and LogisticRegression

```python
# Split the data using the new random seed
X_train, X_test, y_train, y_test = train_test_split(df['preprocessed_readme'], df['language'], test_size=0.2, random_state=321)

# Create a pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('logreg', LogisticRegression(max_iter=1000))
])

# Define the parameter grid
param_grid = {
    'tfidf__max_features': [500, 1000, 5000],
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'tfidf__min_df': [25, 50, 75],
    'tfidf__max_df': [250, 500, 750],
    'logreg__C': [0.1, 1, 10],
    'logreg__penalty': ['l1', 'l2']
}
```
---

### Best GridSearch:

```python
tfidf = TfidfVectorizer(max_df=250, max_features=500, min_df=25, ngram_range=(1, 2))
logreg = LogisticRegression(C=1, penalty='l2')
```

**Baseline:** 13%

**Best cross-validation score:** 87%

**Train Set:** 84%

**Test Set:** 66%

<br>
<br>

---


<br>

## How to Reproduce:

1. Clone this repo (required CSV is in support_files)

2. Run the notebook.

<i>**Alternatively**</i>

1. The `predict_language()` function is in the last cell of the `final_draft.ipynb`.

2. Within the notebook, this can be used on a random README by leaving it blank
    <i>**OR**</i><br>
   or by using the readme_string key word argument

<u> For example, we used: </u>

```python
readme = requests.get(
    "https://raw.githubusercontent.com/Zacharia-Schmitz/nlp_project/main/README.md"
).text

predict_language(
readme_string=readme,
preprocess_func=preprocess_text,
tfidf_path="support_files/tfidf_vectorizer.pkl",
logreg_path="support_files/logreg_model.pkl",)
```

<u> Which returned: </u>

```python
Predicted: python (Score: 0.44)

Other possible predictions:
c++: 0.12
javascript: 0.10
```


<br>
<br>

---

<br>

## Conclusions

### Recommendations

#### For Modeling:

- **Feature Engineering:** Consider extracting additional features from the README text, such as the number of code snippets, mentions of specific libraries, and the use of certain punctuations typical to a programming language.

- **Ensemble Methods:** Combine multiple models for better prediction. For instance, a combination of logistic regression, random forests, and gradient boosting might yield improved results.

- **Deep Learning:** Explore deep learning techniques like RNNs or Transformers (e.g., BERT) which can capture sequential information in the text and might improve accuracy.

- **Regularization:** If overfitting is observed, consider employing stronger regularization techniques or using models with built-in regularization like Ridge or Lasso for logistic regression.

- **Class Imbalance:** If certain languages are underrepresented in the dataset, consider techniques like SMOTE, undersampling, or oversampling to address class imbalance.

- **External Data Sources:** Incorporate external data sources or pre-trained embeddings like Word2Vec or FastText to enhance the representation of the README text.

#### For Data Collection:

- **Diversify Sources:** To avoid biases, collect READMEs from various sources, not just popular repositories, to ensure a diverse representation of projects and languages.

- **Update Data Periodically:** Languages and their ecosystems evolve. Ensure the dataset is updated periodically to reflect recent trends and libraries.

### Next Steps:

- **Validation:** Set up a validation framework to periodically test the model on new READMEs and ensure its predictions remain accurate over time.

- **User Feedback:** If this model is deployed as a tool or service, incorporate user feedback mechanisms to continuously improve it.

- **Expand Scope:** Beyond predicting programming languages, consider expanding the project's scope to categorize projects based on their purpose (e.g., web development, data analysis, gaming).

- **Multilabel Classification:** Some projects use multiple languages. Explore models that can predict multiple languages for a single README.

- **Interactive Dashboard:** Develop an interactive dashboard where users can paste README text and get predictions, insights about the prediction confidence, and even see which parts of the text most influenced the prediction.

- **Continuous Learning:** Implement a continuous learning mechanism where the model gets retrained as more data becomes available or if its performance drops below a certain threshold.

- **Topic Modeling:** Beyond just predicting the language, perform topic modeling on READMEs to identify common themes or topics within specific language communities.