# Movie Review Sentiment Analysis

This project implements a sentiment analysis model for movie reviews using machine learning techniques.

## Overview

The Jupyter notebook in this repository contains code to:

- Load and preprocess a dataset of movie reviews
- Convert the text reviews to numerical features using TF-IDF vectorization
- Train a Naive Bayes classifier to predict sentiment (positive/negative) 
- Evaluate the model's performance
- Make predictions on new reviews

## Contents

- `movie_review_sentiment_analysis.ipynb`: Main Jupyter notebook with all code
- `IMDB_Dataset_sample.xlsx`: Sample dataset of movie reviews (not included in repo)

## Requirements

- Python 3.x
- Pandas
- Scikit-learn
- NLTK

## Usage

1. Clone this repository
2. Install the required libraries
3. Open and run the Jupyter notebook

The notebook walks through the full process from data loading to model evaluation and testing.

## Model 

The sentiment analysis model uses:

- TF-IDF vectorization to convert text to numerical features
- Multinomial Naive Bayes classifier

## Results

The model achieves the following performance on the test set:

- Accuracy: 81%
- Precision: 0.83 
- Recall: 0.81
- F1-score: 0.80

## Future Work

Potential areas for improvement:
- Try other classification algorithms (e.g. SVM, logistic regression)
- Experiment with different text preprocessing techniques
- Collect more training data
- Implement cross-validation

## License

This project is open source and available under the [MIT License](LICENSE).
