# Sentiment Analysis on IMDb Reviews

## Overview

This project implements a **Sentiment Analysis** model to classify movie reviews as **positive** or **negative** using the IMDb Reviews dataset. The task involves several key steps including data collection, preprocessing, feature extraction, model training, and evaluation.

### Assignment Objective
The goal of this project is to build a machine learning model that can accurately classify text data (movie reviews) as positive or negative sentiment. The following steps were performed to meet the requirements of the assignment:

## Dataset - important

The dataset used in this project is the **IMDb Reviews Dataset**, which contains 50,000 movie reviews labeled as positive or negative. 

You can download the dataset from the following Kaggle link:
- [IMDb Reviews Dataset](https://www.kaggle.com/competitions/your-competition-name/data)

To use this dataset, you will need to create a Kaggle account (if you don't have one) and authenticate your Kaggle API key.

Once downloaded, place the `IMDB_Dataset.csv` file in the project directory to run the model.

1. **Data Collection**:
   - A publicly available dataset was used for this task, the IMDb Reviews dataset.
   - The dataset contains 50,000 movie reviews, each labeled as **positive** or **negative**.

2. **Data Preprocessing**:
   - **Cleaning the text**: The text data was cleaned by removing special characters, digits, and stop words. It was also converted to lowercase and tokenized into individual words.
   - **Handling missing values**: Any missing data in the reviews column was filled with an empty string.

3. **Feature Extraction**:
   - **TF-IDF (Term Frequency-Inverse Document Frequency)** was used to convert the text into numerical features. TF-IDF is chosen over the **Bag of Words** model because it takes into account the importance of words in a document relative to their frequency across the entire dataset, helping reduce the weight of common words and focusing on less frequent but more informative words.

4. **Model Selection and Training**:
   - The model was trained using the **Naive Bayes** classifier, specifically the **Multinomial Naive Bayes** algorithm, which is effective for text classification problems like this.
   - The data was split into training and testing sets (80% training, 20% testing).

5. **Evaluation**:
   - The performance of the model was evaluated using the following metrics:
     - **Accuracy**
     - **Precision**
     - **Recall**
     - **Confusion Matrix**
   - The model achieved a high accuracy score but had lower precision and recall due to class imbalance.

6. **Improvements**:
   - To improve the model, you could experiment with other algorithms such as **Logistic Regression** or **Support Vector Machines**.
   - Hyperparameter tuning using **GridSearchCV** and handling the class imbalance with techniques like **SMOTE** can improve performance further.

## Files

- `WHF-Part_B.ipynb`: The main Python script containing all the code for data loading, preprocessing, model training, and evaluation.
- `IMDB_Dataset.csv`: The dataset containing the IMDb movie reviews (50,000 labeled samples).
- `requirements.txt`: The list of dependencies required to run this project.
