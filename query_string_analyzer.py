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
RANDOM_STATE = 123


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


# Creating inputs (X)
good_queries = load_file('dataset/kdn_url_queries/goodqueries.txt')
bad_queries = load_file('dataset/kdn_url_queries/badqueries.txt')
good_queries = good_queries[:44532]     # TODO temporary to balance dataset
all_queries = good_queries + bad_queries   # list of all queries, first the good one

# Create supervised output (y), 1 for bad queries, 0 for good queries
good_query_y = [0 for i in range(0, len(good_queries))]
bad_query_y = [1 for i in range(0, len(bad_queries))]
y = good_query_y + bad_query_y      # outputs vector, first for good queries


'''
#n_gram range(A,B) usa tutti gli ngrammi da A a B
#min_df ignora tutti i vocaboli che hanno frequena nei documenti minore del valore dato
vectorizer = TfidfVectorizer(min_df=0.0, analyzer="char", sublinear_tf=True, ngram_range=(1,3)) #converting data to vectors
'''
vectorizer = TfidfVectorizer(tokenizer=n_gram_tokenizer)    # term frequency-inverse document frequency;
X = vectorizer.fit_transform(all_queries)   # convert inputs to vectors


# Print sizes
print(f'good_queries: \t\t{len(good_queries)}\t\t({len(good_queries)/len(all_queries)*100:5.2f}%)')
print(f'bad_queries: \t\t{len(bad_queries)}\t\t({len(bad_queries)/len(all_queries)*100:5.2f}%)')
print(f'total_queries: \t\t{len(all_queries)}')

print(f'X size: \t\t{X.getnnz()}')
# TODO capire e poi togliere la stampa print(X)
print(X)

# Split dataset: train, validation and test (80,10,10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=RANDOM_STATE)

print(f'X_TRAIN (shape[0]): \t\t{X_train.shape[0]}')            # todo: problema le shape[0] sono uguali alla cardinalita di Y, le .getnnz() no
print(f'X_TRAIN (.getnnz()): \t\t{X_train.getnnz()}')
print(f'Y_TRAIN: \t\t{len(y_train)}')
print(f'X_VALIDATION (shape[0]): \t\t{X_validation.shape[0]}')
print(f'X_VALIDATION (.getnnz()): \t\t{X_validation.getnnz()}')
print(f'Y_VALIDATION: \t\t{len(y_validation)}')
print(f'X_TEST (shape[0]): \t\t{X_test.shape[0]}')
print(f'X_TEST (.getnnz()): \t\t{X_test.getnnz()}')
print(f'Y_TEST: \t\t{len(y_test)}')

# problema: poi come uso i dati nuovi (cioè, dovrebbero essere processati come vengono processati quelli
# di training, validation e testing, ovvero: decoding, lower, tokenizing with N-grams, tfidf vectorizing)
# quindi quelle fatte in load_file(), n_gram_tokenizer() e nel tfidf vectorizer.fit_transform


# -------------------------------------------------MODEL SELECTION--------------------------------------
# From: LogisticRegression, NaiveBayes, Random Forest, KNN, SVM with non-linear kernel

# LogisticRegression
# todo provare i solver solver : str, {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, default: ‘liblinear’.
LogisticRegression_Classifier = LogisticRegression(solver='lbfgs', random_state=RANDOM_STATE) # put solver to silence the warning
LogisticRegression_Classifier.fit(X_train, y_train)   # train the model
'''
# pesi delle classi {nomeclasse:peso; nomeclasse2:peso2, ...}
LogisticRegressionModel = LogisticRegression(class_weight={1: 2 *len(good_queries)/len(bad_queries), 0: 1.0}) # class_weight='balanced')
'''

# Naive Bayes
MultinomialNB_Classifier = MultinomialNB()
MultinomialNB_Classifier.fit(X_train, y_train)

BernoulliNB_Classifier = BernoulliNB()       #T ODO probabilmente inutile...i dati non sono binari
BernoulliNB_Classifier.fit(X_train, y_train)

ComplementNB_Classifier = ComplementNB()
ComplementNB_Classifier.fit(X_train, y_train)

# KNN
# KNeighborsClassifier_Classifier = KNeighborsClassifier(n_jobs=-1)      usare tutti i core sembra far rallentare
# todo provare algorithm : {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’} se metto auto prende l'algo piu adatto in base ai parametri della fit
KNeighborsClassifier_Classifier = KNeighborsClassifier(n_neighbors=5)        # lento e impreciso
KNeighborsClassifier_Classifier.fit(X_train, y_train)

RadiusNeighborsClassifier_Classifier = RadiusNeighborsClassifier(radius=1.0)
RadiusNeighborsClassifier_Classifier.fit(X_train, y_train)              # todo error in validation

# todo Random forest and svc seems to not have fit method(), no warnings generated, but running the code result in infinite loop in that instruction

# RandomForest
RandomForestClassifier_classifier = RandomForestClassifier(n_estimators=1000, criterion='gini', random_state=RANDOM_STATE)



# SVM
# TODO provare altri kernel (# ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’)e cambiare C e gamma
# usare i parametri gamma=grado per gli altri kernel e degree=grado della polinomiale
SVC_Classifier = SVC(C=1.0, gamma='auto', cache_size=500, kernel='sigmoid', random_state=RANDOM_STATE)
SVC_Classifier.fit(X_train, y_train)
NuSVC_Classifier = NuSVC(nu=0.5, cache_size=500, kernel='sigmoid', random_state=RANDOM_STATE)
NuSVC_Classifier.fit(X_train, y_train)


# Metrics of all models (use Validation dataset):
# todo si potrebbe cambiare la funzione in modo che printi solo una stringa (riga) con i valori, poi fare ciclo per tutti i classificatori, in modo da avere una tabella di valori
validate_model(LogisticRegression_Classifier, X_validation, y_validation, 'Logistic Regression')
validate_model(MultinomialNB_Classifier, X_validation, y_validation, 'MultinomialNB')
validate_model(BernoulliNB_Classifier, X_validation, y_validation, 'BernoulliNB')
validate_model(ComplementNB_Classifier, X_validation, y_validation, 'ComplementNB')
validate_model(KNeighborsClassifier_Classifier, X_validation, y_validation, 'KNeighborsClassifier')
# validate_model(RadiusNeighborsClassifier_Classifier, X_validation, y_validation, 'RadiusNeighborsClassifier')
validate_model(RandomForestClassifier_classifier, X_validation, y_validation, 'RandomForestClassifier')
validate_model(SVC_Classifier, X_validation, y_validation, 'SVC')
validate_model(NuSVC_Classifier, X_validation, y_validation, 'NuSVC')


# Selected model:
# Metrics of the selected model (use Test dataset):
tmp_Classifier = LogisticRegression_Classifier # TODO remove and replace with selected classifier, also rename print
validate_model(tmp_Classifier, X_test, y_test, 'TMP CLASSIFIER')

# Pickle selected model:
with open('pickle_tmp/queryStringClassifier.pickle', 'wb') as save_classifier_file:
    pickle.dump(tmp_Classifier, save_classifier_file)


