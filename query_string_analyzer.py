import os, pickle, urllib.parse, time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

'''
Script for training the selected classifier (Logistic Regression), and use
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


def validate_model(model, x_validation, y_validation, model_name):
    print(f'\n\n----------------  {model_name}  ----------------')

    y_predicted = model.predict(x_validation)
    fpr, tpr, _ = metrics.roc_curve(y_validation, (model.predict_proba(x_validation)[:, 1]))
    auc = metrics.auc(fpr, tpr)

    # Metrics evaluation
    print(f'Score: \t\t{model.score(x_validation, y_validation) * 100:10.2f}\n')
    print(f'Accuracy: \t{model.score(x_validation, y_validation)*100:10.2f}')
    print(f'Precision: \t{metrics.precision_score(y_validation, y_predicted)*100:10.2f}')
    print(f'Recall: \t{metrics.recall_score(y_validation, y_predicted)*100:10.2f}')
    print(f'F1-Score: \t{metrics.f1_score(y_validation, y_predicted)*100:10.2f}')
    print(f'AUC: \t\t{auc*100:10.2f}')


def time_stamp(start):
    print(f'Execution time: {start-time.time()}')


start_time = time.time()

# Creating inputs (X)
good_queries = load_file('dataset/myDataset/good.txt')
bad_queries = load_file('dataset/myDataset/bad.txt')
good_queries = good_queries[:len(bad_queries)]     #balance dataset
all_queries = good_queries + bad_queries   # list of all queries, first the good one

# Create supervised output (y), 1 for bad queries, 0 for good queries
good_query_y = [0 for i in range(0, len(good_queries))]
bad_query_y = [1 for i in range(0, len(bad_queries))]
y = good_query_y + bad_query_y      # outputs vector, first for good queries


vectorizer = TfidfVectorizer(tokenizer=n_gram_tokenizer)    # term frequency-inverse document frequency;
X = vectorizer.fit_transform(all_queries)   # convert inputs to vectors

# Split dataset: train, validation and test (80,10,10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=RANDOM_STATE)

#todo remove timestamp
time_stamp(start_time)
print("INITIALISED DATAS, STARTING TRAINING")

# -------------------------------------------------MODEL SELECTION--------------------------------------
# LogisticRegression
LogisticRegression_Classifier = LogisticRegression(solver='lbfgs', random_state=RANDOM_STATE, max_iter=10000) # put solver to silence the warning
LogisticRegression_Classifier.fit(X_train, y_train)   # train the model
save_pickle(LogisticRegression_Classifier, 'pickle_tmp/LogisticRegression_Classifier.pickle')

#todo remove timestamp
time_stamp(start_time)
print("TRAINED Logistic Regression")

# RandomForest
RandomForest_Classifier = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=RANDOM_STATE)
RandomForest_Classifier.fit(X_train, y_train)
save_pickle(RandomForest_Classifier, 'pickle_tmp/RandomForest_Classifier.pickle')

#todo remove timestamp
time_stamp(start_time)
print("TRAINED Random Forest")


# Metrics of all models (use Validation dataset):
validate_model(LogisticRegression_Classifier, X_validation, y_validation, 'Logistic Regression')
#todo remove timestamp
time_stamp(start_time)
print("VALIDATE Logistic Regression")

validate_model(RandomForest_Classifier, X_validation, y_validation, 'RandomForestClassifier')

#todo remove timestamp
time_stamp(start_time)
print("VALIDATE Random Forest")


# Selected model: Logistic Regression
# Metrics of the selected model (use Test dataset):
validate_model(LogisticRegression_Classifier, X_test, y_test, 'BEST CLASSIFIER - LOGISTIC REGRESSION')

#todo remove timestamp
time_stamp(start_time)
print("FINISHED TESTING - END SCRIPT")


#TODO create the sentiment analysis function and update this script into a module




