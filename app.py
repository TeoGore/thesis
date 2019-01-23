from flask import Flask, render_template, request, redirect, url_for
from HTTPRequest import HTTPRequest

app = Flask(__name__)
users = {'Gore':'aaa', 'Gore2':'bbb', 'Gore3':'ccc'}
logged_users = {}


def add_user(user, password):
    #todo check se user o password nulle mandare a pagina errore
    users[user] = password


def delete_user(user):
    del users[user]


def login_user(user, password):
    logged_users[user] = password


def logout_user(user):
    del logged_users[user]


def login(user, password):
    try:
        return users[user] == password
    except(KeyError):
        return False




def is_logged(user):
    return logged_users.__contains__(user)

#TODO aggiungere bottone per cancellarsi
#TODO fare profili utente
#TODO modificare aggiungendo navbar in modo da togliere le pagine di successo, insuccesso e reinderizzamento


@app.route('/', methods=['GET', 'POST'])
def home():
    analize_request(request)
    #if request.form['submit'] == 'LogOut':
        #TODO passare l'utente a tutte le pagine, in modo che possa essere ripreso per il logout, preso user decommentare queste due istruzioni e togliere pass
       #username, password = take_user(request)
       #logout_user(username)
        #pass
    return render_template('login.html')    #tolti file jsp


@app.route('/log_handler/', methods=['POST'])
def log_handler():
    analize_request(request)
    username, password = take_user(request)
    if request.form['submit'] == 'SignUp':
        add_user(username, password)
        login_user(username, password)
        return render_template('success.html')
    if request.form['submit'] == 'Login' and login(username, password):
        login_user(username, password)
        return render_template('loggedIn.html', name=username, key=password)
    else:
        return render_template('logError.html')


#TODO fare in modo di presentare la lista di utenti totali e di utenti loggati

@app.route('/user_list/', methods=['GET'])
def user_list():
    analize_request(request)
    #TODO mettere bottone logout, gia messo in home con valore 'LogOut'
    return render_template('user_list.html', users=users)


if __name__ == '__main__':
    app.run(debug=True, port=5000)

def take_user(request):
    username = request.form['name']
    password = request.form['password']
    return username, password

#TODO fare in modo che venga chiamata a ogni richiesta del server
def analize_request(request):
    r = HTTPRequest(request)
    print(r.raw_request())







