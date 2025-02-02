**Disaster Tweet Classification** 🚀
**Project Overview**
This project aims to classify tweets as disaster-related or non-disaster-related using machine learning models. The goal is to automate disaster detection from social media, which can help emergency responders take timely action.

**Dataset **📊
Source: Kaggle Twitter Disaster Dataset
Key Features:
text: The tweet content
keyword: Specific disaster-related words
location: Tweet location (if available)
target: 1 (Disaster), 0 (Non-Disaster)
Dataset Size: 10,000+ tweets
Class Distribution: ~40% disaster tweets, ~60% non-disaster tweets
Project Workflow 🔄
Data Preprocessing

Tokenization, stopword removal, stemming
Handling missing values (keyword, location)
Removing special characters, URLs, and punctuation
Exploratory Data Analysis (EDA)

WordClouds, bar charts of frequent words
Tweet length analysis
Sentiment analysis
Feature Engineering

TF-IDF Vectorization
Word embeddings (Word2Vec, GloVe)
Additional features like tweet length, presence of URLs, hashtags
Model Training & Tuning

Algorithms used: Logistic Regression, Random Forest, XGBoost, LightGBM
Train-test split (80%-20%)
Hyperparameter tuning with GridSearchCV
Addressing class imbalance using SMOTE & Class Weights
Model Evaluation

Accuracy, Precision, Recall, F1-score, ROC-AUC
Best Model: XGBoost (Accuracy: 88.7%, ROC-AUC: 0.92)
Results & Insights 📈
XGBoost performed best with 88.7% accuracy
Disaster tweets tend to have more negative sentiment
Feature engineering significantly improved model performance
Future scope: Implementing transformer-based models (BERT) for better accuracy
Installation & Usage 💻
To run this project on your local machine:

Clone the repository:
bash
Copy
Edit
git clone https://github.com/yourusername/disaster-tweet-classification.git
cd disaster-tweet-classification
Install dependencies:
bash
Copy
Edit
pip install -r requirements.txt
Run the notebook:
bash
Copy
Edit
jupyter notebook
Train the model:
python
Copy
Edit
python train_model.py
Contributing 🤝
Contributions are welcome! Feel free to open an issue or submit a pull request.
