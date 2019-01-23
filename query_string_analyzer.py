import os, math, pickle, urllib.parse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV

#SVM con kernel non lineare (uso SVC perchè è classificazione, non devo fare regressione(SVR))
from sklearn.svm import SVC, NuSVC

#NaiveBayes
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB, BaseNB, BaseDiscreteNB, ComplementNB, BaseEstimator

#Random Forest
from sklearn.ensemble import RandomForestClassifier

#KNN
from sklearn.neighbors import NearestNeighbors, RadiusNeighborsClassifier, KNeighborsClassifier, KDTree

from sklearn import metrics


N = 3       #Ngram value
RANDOM_STATE = 123

def save_pickle(variable, filename):
    save_file = open(filename, 'wb')
    pickle.dump(variable, save_file)
    save_file.close()

def load_file(name):
    directory = str(os.getcwd())
    filepath = os.path.join(directory, name)
    result = []
    with open(filepath,'r') as f:
        for line in f:
            data = str(urllib.parse.unquote(line))  # converting encoded url to simple string
            result.append(data)
    return list(set(result))    #delete duplicate datas


def n_gram_tokenizer(request):
    n_grams = []
    if len(request)<= N/2:      #request too short, it's a token itself
        n_grams.append(request)
        return n_grams
    for i in range(0, len(request) - N):
        n_grams.append(request[i:i + N])
    return n_grams


def validate_model(model, x_validation, y_validation, model_name):
    print(f'\n\n----------------  {model_name}  ----------------')
    print(f'Score: \t\t{model.score(x_validation, y_validation)*100:10.2f}\n')

    y_predicted = model.predict(x_validation)
    fpr, tpr, _ = metrics.roc_curve(y_validation, (model.predict_proba(x_validation)[:, 1]))
    auc = metrics.auc(fpr, tpr)

    #metrics evaluation

    print(f'Accuracy: \t{model.score(x_validation, y_validation)*100:10.2f}')
    print(f'Precision: \t{metrics.precision_score(y_validation, y_predicted)*100:10.2f}')
    print(f'Recall: \t{metrics.recall_score(y_validation, y_predicted)*100:10.2f}')
    print(f'F1-Score: \t{metrics.f1_score(y_validation, y_predicted)*100:10.2f}')
    print(f'AUC: \t\t{auc*100:10.2f}')


#creating inputs
good_queries = load_file('dataset/kdn_url_queries/goodqueries.txt')
bad_queries = load_file('dataset/kdn_url_queries/badqueries.txt')


good_queries = good_queries[:44532]     #TODO temporary to balance dataset


all_queries = good_queries + bad_queries   #list of all queries, first the good one

#create supervised output, 1 for bad queries, 0 for good queries
good_query_y = [0 for i in range(0, len(good_queries))]
bad_query_y = [1 for i in range(0, len(bad_queries))]
y = good_query_y + bad_query_y      #outputs vector, first for good queries


'''
TF (term frequency) = (#times find a given word in a document) / (#words in that document); è la frequenza del termine
IDF (inverse document frequency) =log_e(#total_documents, #documents where the word is found); è l'importanza del termine
In practice: TF calculates term frequency giving same weights to all words, IDF scales TF value so that rare words weigth more than common
TFIDF is the product (TF*IDF), indicates a weigth that is the importance of the word
'''

'''
#n_gram range(A,B) usa tutti gli ngrammi da A a B
#min_df ignora tutti i vocaboli che hanno frequena nei documenti minore del valore dato
vectorizer = TfidfVectorizer(min_df=0.0, analyzer="char", sublinear_tf=True, ngram_range=(1,3)) #converting data to vectors
'''
vectorizer = TfidfVectorizer(tokenizer=n_gram_tokenizer) #term frequency-inverse document frequency;
X = vectorizer.fit_transform(all_queries)   #convert inputs to vectors


#Print sizes
print(f'good_queries: \t\t{len(good_queries)}\t\t({len(good_queries)/len(all_queries)*100:5.2f}%)')
print(f'bad_queries: \t\t{len(bad_queries)}\t\t({len(bad_queries)/len(all_queries)*100:5.2f}%)')
print(f'total_queries: \t\t{len(all_queries)}')

print(f'X size: \t\t{X.getnnz()}')
#TODO capire e poi togliere la stampa print(X)
print(X)

#split dataset: train, validation and test (80,10,10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=RANDOM_STATE)


#-------------------------------------------------MODEL SELECTION--------------------------------------
#model selection (from: LogisticRegression, NaiveBayes, Random Forest, KNN, SVM with non-linear kernel)

#LogisticRegression
LogisticRegression_Classifier = LogisticRegression(random_state=RANDOM_STATE)
LogisticRegression_Classifier.fit(X_train, y_train)   #train the model
'''
#pesi delle classi {nomeclasse:peso; nomeclasse2:peso2, ...}
LogisticRegressionModel = LogisticRegression(class_weight={1: 2 *len(good_queries)/len(bad_queries), 0: 1.0}) # class_weight='balanced')
'''
#SVM
SVC_Classifier = SVC(C=1.0, gamma='auto', cache_size=500, kernel='linear', random_state=RANDOM_STATE) #TODO provare altri kernel e cambiare C e gamma
SVC_Classifier.fit(X_train, y_train)
#NuSVC_Classifier = NuSVC()


#Naive Bayes
#RandomForest
#KNN



#metrics of all models (use Validation dataset):
#todo si potrebbe cambiare la funzione in modo che printi solo una stringa (riga) con i valori, poi fare ciclo per tutti i classificatori, in modo da avere una tabella di valori

validate_model(LogisticRegression_Classifier, X_validation, y_validation, 'Logistic Regression')


validate_model(SVC_Classifier, X_validation, y_validation, 'SVC')
#validate_model(NuSVC_Classifier, X_validation, y_validation, 'NuSVC')



#selected model:
#metrics of selected model (use Test dataset):

#pickle selected model:
tmp_Classifier = 3 #TODO remove and replace with selected classifier
with open('pickle_tmp/queryStringClassifier.pickle', 'wb') as save_classifier_file:
    pickle.dump(tmp_Classifier, save_classifier_file)


