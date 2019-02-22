import os, math, random, pickle, urllib.parse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

'''
Script for training the selected classifier (Logistic Regression)
Sentiment analysis in query_string_analyzer.py
'''

N = 3       # Ngram value
RANDOM_STATE = 123


def save_pickle(variable, filename):
    save_file = open(filename, 'wb')
    pickle.dump(variable, save_file)
    save_file.close()


def load_file(filename):
    directory = str(os.getcwd())
    filepath = os.path.join(directory, filename)
    result = []
    with open(filepath, 'r') as f:
        for line in f:
            data = str(urllib.parse.unquote(line))  # converting encoded url to simple string
            result.append(data)
    return list(set(result))    # delete duplicate datas


def n_gram_tokenizer(query_string):
    n_grams = []
    if len(query_string) <= N/2:      # too short, it's a token itself
        n_grams.append(query_string)
        return n_grams
    for i in range(0, len(query_string) - N):
        n_grams.append(query_string[i:i + N])
    return n_grams


def confidence_interval(model, x_validation, y_validation):
    # Const values: 1.64 (90%), 1.96 (95%), 2.33 (98%), 2.58 (99%)
    const = 2.58 #want 95% confidence interval
    incorrect_prediction = 0 #todo esce sempre 0, correggere
    y_predicted = model.predict(x_validation)
    for y_pred, y in zip(y_predicted, y_validation):
        if y_pred != y:
            incorrect_prediction += 1
    error = incorrect_prediction/len(y_validation)
    n = len(y_validation)
    value = const * math.sqrt((error * (1 - error)) / n)
    lower_bound = error - value
    upper_bound = error + value
    if lower_bound < 0:
        lower_bound = 0
    if upper_bound > 1:
        upper_bound = 1

    return lower_bound, upper_bound


def validate_model(model, x_validation, y_validation, model_name):
    print(f'\n\n----------------  {model_name}  ----------------')
    # Metrics evaluation
    print(f'Score: \t\t\t\t{model.score(x_validation, y_validation) * 100:10.2f}')
    print(f'Accuracy: \t\t\t{model.score(x_validation, y_validation)*100:10.2f}')
    lower_bound, upper_bound = confidence_interval(model, x_validation, y_validation)
    print(f'Confidence Interval: \t[{lower_bound:.2f}, {upper_bound:.2f}]')


# Creating inputs (X)
#good_queries = load_file('dataset/myDataset/good.txt')
#bad_queries = load_file('dataset/myDataset/bad.txt')


good_queries = load_file('resultTMP/good.txt')
bad_queries = load_file('resultTMP/bad.txt')

good_len = len(good_queries)
bad_len = len(bad_queries)

min_val = min(good_len, bad_len)
if min_val == good_len:
    bad_queries = bad_queries[:good_len]    #less good queries, so balance dataset
else:
    good_queries = good_queries[:bad_len]

all_queries = good_queries + bad_queries   # list of all queries, first the good one

print(f"GOOD QUERIES: {len(good_queries)}")
print(f"BAD QUERIES: {len(bad_queries)}")

# Create supervised output (y), 1 for bad queries, 0 for good queries
good_query_y = [0 for i in range(0, len(good_queries))]
bad_query_y = [1 for i in range(0, len(bad_queries))]
y = good_query_y + bad_query_y      # outputs vector, first for good queries

# Split dataset: train, validation and test (80,10,10)
X_train, X_test, y_train, y_test = train_test_split(all_queries, y, test_size=0.2, random_state=RANDOM_STATE)
X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=RANDOM_STATE)

vectorizer = TfidfVectorizer(tokenizer=n_gram_tokenizer)    # term frequency-inverse document frequency
vectorizer_fitted = vectorizer.fit(X_train)
save_pickle(vectorizer_fitted, "classifiers/TFIDF_Vectorizer")


X_train = vectorizer.transform(X_train)   # convert inputs to vectors
X_validation = vectorizer.transform(X_validation)
X_test = vectorizer.transform(X_test)



# -----------------------------MODEL SELECTION-----------------------------

'''
# LogisticRegression
LR_1 = LogisticRegression(C=0.01, solver='liblinear', penalty='l1', random_state=RANDOM_STATE)
LR_1.fit(X_train, y_train)
LR_2 = LogisticRegression(C=0.02, solver='liblinear', penalty='l1', random_state=RANDOM_STATE)
LR_2.fit(X_train, y_train)
LR_3 = LogisticRegression(C=0.1, solver='liblinear', penalty='l1', random_state=RANDOM_STATE)
LR_3.fit(X_train, y_train)
LR_4 = LogisticRegression(C=0.2, solver='liblinear', penalty='l1', random_state=RANDOM_STATE)
LR_4.fit(X_train, y_train)
LR_5 = LogisticRegression(C=1.0, solver='liblinear', penalty='l1', random_state=RANDOM_STATE)
LR_5.fit(X_train, y_train)
LR_6 = LogisticRegression(C=2.0, solver='liblinear', penalty='l1', random_state=RANDOM_STATE)
LR_6.fit(X_train, y_train)
LR_7 = LogisticRegression(C=10.0, solver='liblinear', penalty='l1', random_state=RANDOM_STATE)
LR_7.fit(X_train, y_train)
LR_8 = LogisticRegression(C=20.0, solver='liblinear', penalty='l1', random_state=RANDOM_STATE)
LR_8.fit(X_train, y_train)

PERFORMANCE:
----------------  Logistic Regression 1  ----------------
Score: 				     98.86
Accuracy: 			     98.86
Confidence Interval: 	[0.01, 0.01]


----------------  Logistic Regression 2  ----------------
Score: 				     99.34
Accuracy: 			     99.34
Confidence Interval: 	[0.01, 0.01]


----------------  Logistic Regression 3  ----------------
Score: 				     99.86
Accuracy: 			     99.86
Confidence Interval: 	[0.00, 0.00]


----------------  Logistic Regression 4  ----------------
Score: 				     99.93
Accuracy: 			     99.93
Confidence Interval: 	[0.00, 0.00]


----------------  Logistic Regression 5  ----------------
Score: 				     99.98
Accuracy: 			     99.98
Confidence Interval: 	[0.00, 0.00]


----------------  Logistic Regression 6  ----------------
Score: 				     99.99
Accuracy: 			     99.99
Confidence Interval: 	[0.00, 0.00]


----------------  Logistic Regression 7  ----------------
Score: 				    100.00
Accuracy: 			    100.00
Confidence Interval: 	[0.00, 0.00]


----------------  Logistic Regression 8  ----------------
Score: 				    100.00
Accuracy: 			    100.00
Confidence Interval: 	[0.00, 0.00]
'''

'''
LR_1 = LogisticRegression(C=0.01, solver='saga', random_state=RANDOM_STATE)
LR_1.fit(X_train, y_train)
LR_2 = LogisticRegression(C=0.02, solver='saga', random_state=RANDOM_STATE)
LR_2.fit(X_train, y_train)
LR_3 = LogisticRegression(C=0.1, solver='saga', random_state=RANDOM_STATE)
LR_3.fit(X_train, y_train)
LR_4 = LogisticRegression(C=0.2, solver='saga', random_state=RANDOM_STATE)
LR_4.fit(X_train, y_train)
LR_5 = LogisticRegression(C=1.0, solver='saga', random_state=RANDOM_STATE)
LR_5.fit(X_train, y_train)
LR_6 = LogisticRegression(C=2.0, solver='saga', random_state=RANDOM_STATE)
LR_6.fit(X_train, y_train)
LR_7 = LogisticRegression(C=10.0, solver='saga', random_state=RANDOM_STATE)
LR_7.fit(X_train, y_train)
LR_8 = LogisticRegression(C=20.0, solver='saga', random_state=RANDOM_STATE)
LR_8.fit(X_train, y_train)

----------------  Logistic Regression 1  ----------------
Score: 				     99.65
Accuracy: 			     99.65
Confidence Interval: 	[0.00, 0.00]


----------------  Logistic Regression 2  ----------------
Score: 				     99.79
Accuracy: 			     99.79
Confidence Interval: 	[0.00, 0.00]


----------------  Logistic Regression 3  ----------------
Score: 				     99.94
Accuracy: 			     99.94
Confidence Interval: 	[0.00, 0.00]


----------------  Logistic Regression 4  ----------------
Score: 				     99.96
Accuracy: 			     99.96
Confidence Interval: 	[0.00, 0.00]


----------------  Logistic Regression 5  ----------------
Score: 				     99.98
Accuracy: 			     99.98
Confidence Interval: 	[0.00, 0.00]


----------------  Logistic Regression 6  ----------------
Score: 				     99.99
Accuracy: 			     99.99
Confidence Interval: 	[0.00, 0.00]


----------------  Logistic Regression 7  ----------------
Score: 				     99.99
Accuracy: 			     99.99
Confidence Interval: 	[0.00, 0.00]


----------------  Logistic Regression 8  ----------------
Score: 				     99.99
Accuracy: 			     99.99
Confidence Interval: 	[0.00, 0.00]
'''

# LogisticRegression
LR_1 = LogisticRegression(C=0.001, solver='lbfgs', random_state=RANDOM_STATE)
LR_1.fit(X_train, y_train)
LR_2 = LogisticRegression(C=0.002, solver='lbfgs', random_state=RANDOM_STATE)
LR_2.fit(X_train, y_train)
LR_3 = LogisticRegression(C=0.01, solver='lbfgs', random_state=RANDOM_STATE)
LR_3.fit(X_train, y_train)
LR_4 = LogisticRegression(C=0.02, solver='saga', random_state=RANDOM_STATE)
LR_4.fit(X_train, y_train)
LR_5 = LogisticRegression(C=0.1, solver='lbfgs', random_state=RANDOM_STATE)
LR_5.fit(X_train, y_train)
LR_6 = LogisticRegression(C=0.2, solver='lbfgs', random_state=RANDOM_STATE)
LR_6.fit(X_train, y_train)
LR_7 = LogisticRegression(C=1.0, solver='lbfgs', random_state=RANDOM_STATE)
LR_7.fit(X_train, y_train)
LR_8 = LogisticRegression(C=2.0, solver='lbfgs', random_state=RANDOM_STATE)
LR_8.fit(X_train, y_train)
LR_9 = LogisticRegression(C=10.0, solver='lbfgs', random_state=RANDOM_STATE)
LR_9.fit(X_train, y_train)
LR_10 = LogisticRegression(C=20.0, solver='lbfgs', random_state=RANDOM_STATE)
LR_10.fit(X_train, y_train)
LR_11 = LogisticRegression(C=100.0, solver='lbfgs', random_state=RANDOM_STATE)
LR_11.fit(X_train, y_train)
LR_12 = LogisticRegression(C=200.0, solver='lbfgs', random_state=RANDOM_STATE)
LR_12.fit(X_train, y_train)



# Metrics of all models (use Validation dataset):
validate_model(LR_1, X_validation, y_validation, 'Logistic Regression 1')
validate_model(LR_2, X_validation, y_validation, 'Logistic Regression 2')
validate_model(LR_3, X_validation, y_validation, 'Logistic Regression 3')
validate_model(LR_4, X_validation, y_validation, 'Logistic Regression 4')
validate_model(LR_5, X_validation, y_validation, 'Logistic Regression 5')
validate_model(LR_6, X_validation, y_validation, 'Logistic Regression 6')
validate_model(LR_7, X_validation, y_validation, 'Logistic Regression 7')
validate_model(LR_8, X_validation, y_validation, 'Logistic Regression 8')
validate_model(LR_9, X_validation, y_validation, 'Logistic Regression 9')
validate_model(LR_10, X_validation, y_validation, 'Logistic Regression 10')
validate_model(LR_11, X_validation, y_validation, 'Logistic Regression 11')
validate_model(LR_12, X_validation, y_validation, 'Logistic Regression 12')

#TESTING
validate_model(LR_5, X_test, y_test, 'BEST CLASSIFIER - LOGISTIC REGRESSION')
#save_pickle(LR_4, 'pickle_tmp/LogisticRegression_Classifier.pickle')
