# Todo list:

* create a flask login app and add it to this repo
* create a module that can be imported from other user, to use our web application firewall
* respect software engineering pattern (adding new filter or modifying existing ones must be easy!)
* find online regex list (ex: xss and SQLInjection)


* dataset balancing(add more badqueries, with sqlmap and xsser, intercept request and save queries)
* sort dataset with "sort -u" for deleting duplicates
* do model selection with cross validation(train, validation, test = (80,10,10) or (70,15,15)), use some models from: LogisticRegression, NaiveBayes, Random Forest, KNN, SVM with non-linear kernel
* when selected a model, do parameter


* compare the project performances with similar projects

More info in info.txt (also machine learning explanation)


-------------------------
ROBA MIA

potrei mettere un metodo di train del classificatore

poteri fare in modo di analizzare in modi diversi e separati le varie parti della richiesta (path, headers, body) e dare un giudizio sull'insieme che sia pesato

potrei creare un dataset piu grande e fare classificazione su piu gruppi in modo da detectare piu possibili attacchi e classificarli correttamente (SQLi, XSS, ecc...)

potrei fare in modo che venga scritto un log per ogni richiesta in modo che sia parsabile e ci si possano fare dei grafici di modo che si possa capire su cosa si è attaccati in modo piu frequente, come e da chi

fare in modo di collegarlo a modsecurity