import nltk, os, urllib.parse, time
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from nltk.classify import ClassifierI
from statistics import mode

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

'''
creata per fare i pickle, i pickle vanno salvati in una cartella di backup chiamata pickle_tmp
spostare in pickled_items solo se la percentuale è maggiore di quella precedente

PERCENTUALE PRECEDENTE: 0%
Fatto con tokenizing N-gram con n da 1 a 5, tutto insieme

----------
ho messo il metodo score perche alcuni classificatori non lo avevano piu, il problema è che lavora con vettori quidi il voting classifier va upgradato 
(for elem in input_text: voting_classifier.predict(elem))
'''


def score(classifier, X, y, sample_weight=None):
    return accuracy_score(y, classifier.predict(X), sample_weight=sample_weight)


def load_file(filename):
    directory = str(os.getcwd())
    filepath = os.path.join(directory, filename)
    with open(filepath,'r') as f:
        data = f.readlines()
    data = list(set(data))
    result = []
    for d in data:
        d = str(urllib.parse.unquote(d))   #converting url encoded data to simple string
        result.append(d)
    return result

start_time = time.time()


good_url_paths = load_file('dataset/kdn_url_queries/goodqueries.txt')
bad_url_paths = load_file('dataset/kdn_url_queries/badqueries.txt')

#nel caso ci fossero due url uguali ma uno era encoded il set toglie la ridondanza
good_url_paths = list(set(good_url_paths))
bad_url_paths = list(set(bad_url_paths))
all_url_paths = good_url_paths + bad_url_paths

#creo vettore degli output (le label sono 'good_label' per query buone (1), 'bad_label' per le query malevole (0))
y_good = [1 for i in range(0, len(good_url_paths))]
y_bad = [0 for i in range(0, len(bad_url_paths))]
y = y_good + y_bad


#n_gram range(A,B) usa tutti gli ngrammi da A a B
#min_df ignora tutti i vocaboli che hanno frequena nei documenti minore del valore dato
vectorizer = TfidfVectorizer(min_df=0.0, analyzer="char", sublinear_tf=True, ngram_range=(1, 5)) #converting data to vectors
X = vectorizer.fit_transform(all_url_paths)

#split dataset 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)


#CLASSIFIERS AND PERFORMANCE

#LogisticRegression
#pesi delle classi {nomeclasse:peso; nomeclasse2:peso2, ...}
print('\n---------------------- LogisticRegression ----------------------\n')
LogisticRegression_Classifier = LogisticRegression(class_weight={0: 2 * len(good_url_paths) / len(bad_url_paths), 1: 1.0}) # class_weight='balanced')
LogisticRegression_Classifier.fit(X_train, y_train) #training our model
#performance
predicted_y = LogisticRegression_Classifier.predict(X_test)
print(f'Accuracy: {LogisticRegression_Classifier.score(X_test, y_test)}')
print(f'Precision {metrics.precision_score(y_test, predicted_y)}')
#----pickle----
save_classifier = open('pickle_tmp/LogisticRegression_Classifier.pickle', 'wb')
pickle.dump(LogisticRegression_Classifier, save_classifier)
save_classifier.close()


#altri classificatori e vote classfier
#MultinomialNB
print('\n---------------------- MultinomialNB ----------------------\n')
MNB_Classifier = MultinomialNB()
MNB_Classifier.fit(X_train, y_train)
#performance
predicted_y = MNB_Classifier.predict(X_test)
print(f'Accuracy: {score(MNB_Classifier, X_test, y_test)}')
print(f'Precision {metrics.precision_score(y_test, predicted_y)}')
#----pickle----
save_classifier = open('pickle_tmp/MNB_Classifier.pickle', 'wb')
pickle.dump(MNB_Classifier, save_classifier)
save_classifier.close()


#BernoulliNB
print('\n---------------------- BernoulliNB ----------------------\n')
BernoulliNB_Classifier = BernoulliNB()
BernoulliNB_Classifier.fit(X_train, y_train)
#performance
predicted_y = BernoulliNB.predict(X_test[::])
print(f'Accuracy: {score(BernoulliNB, X_test, y_test)}')
print(f'Precision {metrics.precision_score(y_test, predicted_y)}')
#----pickle----
save_classifier = open('pickle_tmp/BernoulliNB_Classifier.pickle', 'wb')
pickle.dump(BernoulliNB_Classifier, save_classifier)
save_classifier.close()


#GaussianNB
print('\n---------------------- GaussianNB ----------------------\n')
GaussianNB_Classifier = GaussianNB()
GaussianNB_Classifier.fit(X_train, y_train)
#performance
predicted_y = GaussianNB.predict(X_test)
print(f'Accuracy: {LogisticRegression_Classifier.score(X_test, y_test)}')
print(f'Precision {metrics.precision_score(y_test, predicted_y)}')
#----pickle----
save_classifier = open('pickle_tmp/GaussianNB_Classifier.pickle', 'wb')
pickle.dump(GaussianNB_Classifier, save_classifier)
save_classifier.close()


#LinearSVC
print('\n---------------------- LinearSVC ----------------------\n')
LinearSVC_Classifier = LinearSVC()
LinearSVC_Classifier.fit(X_train, y_train)
#performance
predicted_y = LinearSVC.predict(X_test)
print(f'Accuracy: {LinearSVC.score(X_test, y_test)}')
print(f'Precision {metrics.precision_score(y_test, predicted_y)}')
#----pickle----
save_classifier = open('pickle_tmp/LinearSVC_Classifier.pickle', 'wb')
pickle.dump(LinearSVC_Classifier, save_classifier)
save_classifier.close()


class VoteClassifier(ClassifierI):
        def __init__(self, *classifiers):
            self.classifiers = classifiers

        def classify(self, url_path):
            votes = []
            for classifier in self.classifiers:
                vote = classifier.predict(url_path)
                votes.append(vote)
            return mode(votes)  #ritorna chi ha piu voti (il valore piu probabile (quello a probabilita piu alta))

        def confidence(self, url_path):
            votes = []
            for classifier in self.classifiers:
                vote = classifier.predict(url_path)
                votes.append(vote)
            choice_votes = votes.count(mode(votes))
            conf = choice_votes/len(votes)

            return conf

vote_classifier = VoteClassifier(LogisticRegression_Classifier,
                                 MNB_Classifier,
                                 BernoulliNB_Classifier,
                                 LinearSVC_Classifier)



#performance
print('\n---------------------- Voting Classifier ----------------------\n')
predicted_y = []
for x_test in X_test:
    predicted_y.append(vote_classifier.classify(x_test))

#print(f'Voting_Classifier accuracy: {nltk.classify.accuracy(vote_classifier, test_set)*100}%')
#print(f'Accuracy: {score(vote_classifier, X_test, y_test)}%')
print(f'Precision {metrics.precision_score(y_test, predicted_y)}')

end_time = time.time()
print(f'Execution time: {end_time-start_time}')