import os, pickle, urllib.parse, time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, NuSVC

from sklearn import metrics

'''
Script for training the classifiers and do training-validation-testing, to select the best classifier
'''

N = 3       # Ngram value
RANDOM_STATE = 123


def time_stamp(init_time):
    end_time = time.time()
    print(f'\n*********\tExecution time: {end_time-init_time}\t*********\n')

def save_pickle(variable, filename):
    save_file = open(filename, 'wb')
    pickle.dump(variable, save_file)
    save_file.close()


def load_file(name):
    directory = str(os.getcwd())
    filepath = os.path.join(directory, name)
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


start_time = time.time()

# Creating inputs (X)
good_queries = load_file('dataset/kdn_url_queries/goodqueries.txt')
bad_queries = load_file('dataset/kdn_url_queries/badqueries.txt')
good_queries = good_queries[:44532]     # TODO temporary to balance dataset
all_queries = good_queries + bad_queries   # list of all queries, first the good one

# Create supervised output (y), 1 for bad queries, 0 for good queries
good_query_y = [0 for i in range(0, len(good_queries))]
bad_query_y = [1 for i in range(0, len(bad_queries))]
y = good_query_y + bad_query_y      # outputs vector, first for good queries


vectorizer = TfidfVectorizer(tokenizer=n_gram_tokenizer)    # term frequency-inverse document frequency
X = vectorizer.fit_transform(all_queries)   # convert inputs to vectors


# Print sizes
print(f'good_queries: \t\t{len(good_queries)}\t\t({len(good_queries)/len(all_queries)*100:5.2f}%)')
print(f'bad_queries: \t\t{len(bad_queries)}\t\t({len(bad_queries)/len(all_queries)*100:5.2f}%)')
print(f'total_queries: \t\t{len(all_queries)}')

# Split dataset: train, validation and test (80,10,10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=RANDOM_STATE)

time_stamp(start_time)
print("STARTING TRAINING MODELS")


# -------------------------------------------------MODEL SELECTION--------------------------------------
# From: LogisticRegression, NaiveBayes, Random Forest, KNN, SVM with non-linear kernel

# LogisticRegression
LogisticRegression_Classifier = LogisticRegression(solver='lbfgs', random_state=RANDOM_STATE)
LogisticRegression_Classifier.fit(X_train, y_train)   # train the model
save_pickle(LogisticRegression_Classifier, 'pickle_tmp/LogisticRegression_Classifier.pickle')

time_stamp(start_time)
print("FINISHED LOGISTIC REGRESSION")

# Naive Bayes
MultinomialNB_Classifier = MultinomialNB()
MultinomialNB_Classifier.fit(X_train, y_train)
save_pickle(MultinomialNB_Classifier, 'pickle_tmp/MultinomialNB_Classifier.pickle')

time_stamp(start_time)
print("FINISHED MULTINOMIAL NAIVE BAYES")

BernoulliNB_Classifier = BernoulliNB()
BernoulliNB_Classifier.fit(X_train, y_train)
save_pickle(BernoulliNB_Classifier, 'pickle_tmp/BernoulliNB_Classifier.pickle')

time_stamp(start_time)
print("FINISHED BERNOULLI NAIVE BAYES")

ComplementNB_Classifier = ComplementNB()
ComplementNB_Classifier.fit(X_train, y_train)
save_pickle(ComplementNB_Classifier, 'pickle_tmp/ComplementNB_Classifier.pickle')

time_stamp(start_time)
print("FINISHED COMPLEMENT NAIVE BAYES")

# KNN
KNeighbors_Classifier = KNeighborsClassifier(n_neighbors=5)        # slow and not precise
KNeighbors_Classifier.fit(X_train, y_train)
save_pickle(KNeighbors_Classifier, 'pickle_tmp/KNeighbors_Classifier.pickle')

time_stamp(start_time)
print("FINISHED K NEIGHBORS")


# RandomForest
RandomForest_Classifier = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=RANDOM_STATE)
RandomForest_Classifier.fit(X_train, y_train)
save_pickle(RandomForest_Classifier, 'pickle_tmp/RandomForest_Classifier.pickle')

time_stamp(start_time)
print("FINISHED RANDOM FOREST")


# SVM
SVC_Classifier = SVC(C=1.0, gamma='auto', cache_size=500, kernel='sigmoid', random_state=RANDOM_STATE, probability=True)
SVC_Classifier.fit(X_train, y_train)
save_pickle(SVC_Classifier, 'pickle_tmp/SVC_Classifier.pickle')

time_stamp(start_time)
print("FINISHED SVC")

NuSVC_Classifier = NuSVC(nu=0.5, cache_size=500, kernel='sigmoid', random_state=RANDOM_STATE, probability=True)
NuSVC_Classifier.fit(X_train, y_train)
save_pickle(NuSVC_Classifier, 'pickle_tmp/NuSVC_Classifier.pickle')

time_stamp(start_time)
print("FINISHED Nu-SVC")


# Metrics of all models (use Validation dataset):
validate_model(LogisticRegression_Classifier, X_validation, y_validation, 'Logistic Regression')
time_stamp(start_time)
print("FINISHED VALIDATION FOR LOGISTIC REGRESSION")
validate_model(MultinomialNB_Classifier, X_validation, y_validation, 'MultinomialNB')
time_stamp(start_time)
print("FINISHED VALIDATION FOR MULTINOMIAL NAIVE BAYES")
validate_model(BernoulliNB_Classifier, X_validation, y_validation, 'BernoulliNB')
time_stamp(start_time)
print("FINISHED VALIDATION FOR BERNOULLI NAIVE BAYES")
validate_model(ComplementNB_Classifier, X_validation, y_validation, 'ComplementNB')
time_stamp(start_time)
print("FINISHED VALIDATION FOR COMPLEMENT NAIVE BAYES")
validate_model(KNeighbors_Classifier, X_validation, y_validation, 'KNeighborsClassifier')
time_stamp(start_time)
print("FINISHED VALIDATION FOR K NEIGHBORS")
validate_model(RandomForest_Classifier, X_validation, y_validation, 'RandomForestClassifier')
time_stamp(start_time)
print("FINISHED VALIDATION FOR RANDOM FOREST")
validate_model(SVC_Classifier, X_validation, y_validation, 'SVC')
time_stamp(start_time)
print("FINISHED VALIDATION FOR SVC")
validate_model(NuSVC_Classifier, X_validation, y_validation, 'NuSVC')
time_stamp(start_time)
print("FINISHED VALIDATION FOR Nu-SVC")


# Selected model: Logisitc Regression
# Metrics of the selected model (use Test dataset):
validate_model(LogisticRegression_Classifier, X_test, y_test, 'BEST CLASSIFIER - LOGISTIC REGRESSION')

time_stamp(start_time)
print("FINISHED SCRIPT")


