# SMS Spam Detection

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Preprocessing](#preprocessing)
- [Modeling](#modeling)
- [Evaluation](#evaluation)


## Project Overview
This project focuses on detecting SMS spam messages using machine learning techniques. By analyzing the content of SMS messages, the model is trained to classify them as either spam or ham (non-spam).

## Features
- **Text Preprocessing**: Cleaning and tokenizing SMS messages, removing noise such as punctuation, stopwords, and non-alphanumeric characters.
- **Vectorization**: Transforming the processed text into numerical data using techniques like Bag of Words (BoW) or TF-IDF.
- **Machine Learning**: Applying models like Naive Bayes, Logistic Regression, and XGBoost to predict whether a message is spam.
- **Evaluation Metrics**: Evaluating model performance using accuracy, precision, recall, and F1-score.

## Installation
To set up this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/sms-spam-detection.git
   ```

2. **Navigate to the project directory**:
   ```bash
   cd sms-spam-detection
   ```

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset
The dataset used for this project is a collection of SMS messages labeled as "spam" or "ham". You can download the dataset from the https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

- **Columns**: The dataset consists of two columns: `label` (spam/ham) and `message` (SMS text).
- **Size**: The dataset contains approximately 5,500 SMS messages.

Ensure that the dataset is saved as `spam.csv` in the root directory.

## Usage

1. **Run the training script**:
   ```bash
   python sms_spam_detection.py
  
## Preprocessing
The SMS text is processed in several steps:

1. **Lowercase Conversion**: All text is converted to lowercase.
2. **Tokenization**: SMS messages are tokenized using `nltk`.
3. **Noise Removal**: Special characters and non-alphanumeric tokens are removed.
4. **Stopword Removal**: Common English stopwords are filtered out.
5. **Stemming**: Words are stemmed using the Porter Stemmer to reduce words to their base form.

## Modeling
Several machine learning algorithms were tested, including:

- **Naive Bayes**: A fast and simple probabilistic classifier based on Bayes' theorem.
- **Logistic Regression**: A classification model that uses a logistic function to estimate probabilities.
- **XGBoost**: A more advanced and accurate gradient boosting model.

You can switch between these models in the code depending on your needs.

## Evaluation
The models are evaluated using the following metrics:

- **Accuracy**: The percentage of correctly classified messages.
- **Precision**: How many selected items are relevant (for spam detection).
- **Recall**: How many relevant items are selected.
- **F1 Score**: The harmonic mean of precision and recall.
- **Confusion Matrix**: A matrix showing true/false positives and negatives.





