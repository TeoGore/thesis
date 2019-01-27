import os, pickle, urllib.parse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
from sklearn.neighbors import NearestNeighbors, RadiusNeighborsClassifier, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, NuSVC

from sklearn import metrics


N = 3       # Ngram value
RANDOM_STATE = 777

SCORE = []
ACCURACY = []
PRECISION = []
RECALL = []
F1_SCORE = []
AUC = []


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

    score_value = model.score(x_validation, y_validation) * 100
    accuracy_value = model.score(x_validation, y_validation)*100
    precision_value = metrics.precision_score(y_validation, y_predicted)*100
    recall_value = metrics.recall_score(y_validation, y_predicted)*100
    f1_value = metrics.f1_score(y_validation, y_predicted)*100
    auc_value = auc*100

    SCORE.append(score_value)
    ACCURACY.append(accuracy_value)
    PRECISION.append(precision_value)
    RECALL.append(recall_value)
    F1_SCORE.append(f1_value)
    AUC.append(auc_value)

    # Metrics evaluation
    print(f'Score: \t\t{score_value:10.2f}\n')
    print(f'Accuracy: \t{accuracy_value:10.2f}')
    print(f'Precision: \t{precision_value:10.2f}')
    print(f'Recall: \t{recall_value:10.2f}')
    print(f'F1-Score: \t{f1_value:10.2f}')
    print(f'AUC: \t\t{auc_value:10.2f}')


def find_index(valuelist):
    return valuelist.index(max(valuelist)) + 1


def print_best():
    best_score_index = find_index(SCORE,)
    best_accuracy_index = find_index(ACCURACY)
    best_precision_index = find_index(PRECISION)
    best_recall_index = find_index(RECALL)
    best_f1_score_index = find_index(F1_SCORE)
    best_auc_index = find_index(AUC)

    print(f'\n\n----------------  BEST MODELS  ----------------')
    print(f'Score: \t\t{best_score_index}\n')
    print(f'Accuracy: \t{best_accuracy_index}')
    print(f'Precision: \t{best_precision_index}')
    print(f'Recall: \t{best_recall_index}')
    print(f'F1-Score: \t{best_f1_score_index}')
    print(f'AUC: \t\t{best_auc_index}')


# Creating inputs (X)
good_queries = load_file('dataset/kdn_url_queries/goodqueries.txt')
bad_queries = load_file('dataset/kdn_url_queries/badqueries.txt')
good_queries = good_queries[:44532]     # TODO temporary to balance dataset
all_queries = good_queries + bad_queries   # list of all queries, first the good one

# Create supervised output (y), 1 for bad queries, 0 for good queries
good_query_y = [0 for i in range(0, len(good_queries))]
bad_query_y = [1 for j in range(0, len(bad_queries))]
y = good_query_y + bad_query_y      # outputs vector, first for good queries


'''
# n_gram range(A,B) usa tutti gli ngrammi da A a B
# min_df ignora tutti i vocaboli che hanno frequena nei documenti minore del valore dato
vect = TfidfVectorizer(min_df=0.0, analyzer="char", sublinear_tf=True, ngram_range=(1,3)) #converting data to vectors
'''
vectorizer = TfidfVectorizer(tokenizer=n_gram_tokenizer)  # term frequency-inverse document frequency;
X = vectorizer.fit_transform(all_queries)   # convert inputs to vectors


# Print sizes
print(f'good_queries: \t\t{len(good_queries)}\t\t({len(good_queries)/len(all_queries)*100:5.2f}%)')
print(f'bad_queries: \t\t{len(bad_queries)}\t\t({len(bad_queries)/len(all_queries)*100:5.2f}%)')
print(f'total_queries: \t\t{len(all_queries)}')
print(f'X size: \t\t{X.getnnz()}')


# Split dataset: train, validation and test (80,10,10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=RANDOM_STATE)


# -------------------------------------------------MODEL SELECTION--------------------------------------
# From: LogisticRegression, NaiveBayes, Random Forest, KNN, SVM with non-linear kernel
'''
Find best parameters for:

Logistic regression:
KNeighborsClassifier
RadiusNeighborsClassifier
RandomForestClassifier
SVC_Classifier
NuSVC_Classifier

'''

# LogisticRegression

# usare classe liblinear per dataset piccoli, per quelli grandi: sag e saga
# penalita: newton, sag e lbfgs fanno solo L2, liblinear e saga anche L1.

LR_1 = LogisticRegression(C=1.0, tol=0.001, solver='lbfgs', random_state=RANDOM_STATE)
LR_1.fit(X_train, y_train)
LR_2 = LogisticRegression(C=1.0, tol=0.001, solver='liblinear', random_state=RANDOM_STATE)
LR_2.fit(X_train, y_train)
LR_3 = LogisticRegression(C=1.0, tol=0.001, solver='sag', random_state=RANDOM_STATE)
LR_3.fit(X_train, y_train)
LR_4 = LogisticRegression(C=1.0, tol=0.001, solver='saga', random_state=RANDOM_STATE)
LR_4.fit(X_train, y_train)
LR_5 = LogisticRegression(C=1.0, tol=0.001, solver='newton-cg', random_state=RANDOM_STATE)
LR_5.fit(X_train, y_train)
LR_6 = LogisticRegression(C=1.0, tol=0.001, solver='liblinear', penalty='l1', random_state=RANDOM_STATE)
LR_6.fit(X_train, y_train)
LR_7 = LogisticRegression(C=1.0, tol=0.001, solver='saga', penalty='l1', random_state=RANDOM_STATE)
LR_7.fit(X_train, y_train)
LR_8 = LogisticRegression(C=2.0, tol=0.00001, solver='lbfgs', random_state=RANDOM_STATE)
LR_8.fit(X_train, y_train)
LR_9 = LogisticRegression(C=2.0, tol=0.00001, solver='liblinear', random_state=RANDOM_STATE)
LR_9.fit(X_train, y_train)
LR_10 = LogisticRegression(C=2.0, tol=0.00001, solver='sag', random_state=RANDOM_STATE)
LR_10.fit(X_train, y_train)
LR_11 = LogisticRegression(C=2.0, tol=0.00001, solver='saga', random_state=RANDOM_STATE)
LR_11.fit(X_train, y_train)
LR_12 = LogisticRegression(C=2.0, tol=0.00001, solver='newton-cg', random_state=RANDOM_STATE)
LR_12.fit(X_train, y_train)
LR_13 = LogisticRegression(C=2.0, tol=0.00001, solver='liblinear', penalty='l1', random_state=RANDOM_STATE)
LR_13.fit(X_train, y_train)
LR_14 = LogisticRegression(C=2.0, tol=0.00001, solver='saga', penalty='l1', random_state=RANDOM_STATE)
LR_14.fit(X_train, y_train)

'''
#pesi delle classi {nomeclasse:peso; nomeclasse2:peso2, ...}
LogisticRegressionModel = LogisticRegression(class_weight={1: 2 *len(good_queries)/len(bad_queries), 0: 1.0}) # class_weight='balanced')
'''

'''
# KNN
# KNeighborsClassifier_Classifier = KNeighborsClassifier(n_jobs=-1)      usare tutti i core sembra far rallentare
#todo provare algorithm : {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’} se metto auto prende l'algo piu adatto in base ai parametri della fit
KNeighborsClassifier_Classifier = KNeighborsClassifier(n_neighbors=5)        #lento e impreciso
KNeighborsClassifier_Classifier.fit(X_train, y_train)

RadiusNeighborsClassifier_Classifier = RadiusNeighborsClassifier(radius=1.0)
RadiusNeighborsClassifier_Classifier.fit(X_train, y_train)              #todo error in validation

# todo Random forest and svc seems to not have fit method(), no warnings generated, but running the code result in infinite loop in that instruction
# RandomForest
RandomForestClassifier_classifier = RandomForestClassifier(n_estimators=1000, criterion='gini', random_state=RANDOM_STATE)

# SVM
#TODO provare altri kernel (# ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’)e cambiare C e gamma
# usare i parametri gamma=grado per gli altri kernel e degree=grado della polinomiale
SVC_Classifier = SVC(C=1.0, gamma='auto', cache_size=500, kernel='sigmoid', random_state=RANDOM_STATE) #TODO provare altri kernel (# ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’)e cambiare C e gamma
SVC_Classifier.fit(X_train, y_train)
NuSVC_Classifier = NuSVC(nu=0.5, cache_size=500, kernel='sigmoid', random_state=RANDOM_STATE)
NuSVC_Classifier.fit(X_train, y_train)
'''

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
validate_model(LR_13, X_validation, y_validation, 'Logistic Regression 13')
validate_model(LR_14, X_validation, y_validation, 'Logistic Regression 14')


'''
validate_model(KNeighborsClassifier_Classifier, X_validation, y_validation, 'KNeighborsClassifier')
#validate_model(RadiusNeighborsClassifier_Classifier, X_validation, y_validation, 'RadiusNeighborsClassifier')
validate_model(RandomForestClassifier_classifier, X_validation, y_validation, 'RandomForestClassifier')
validate_model(SVC_Classifier, X_validation, y_validation, 'SVC')
validate_model(NuSVC_Classifier, X_validation, y_validation, 'NuSVC')
'''

print_best()
