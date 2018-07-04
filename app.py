from flask import Flask, render_template, request

app = Flask(__name__)
users = {'Gore':'aaa', 'Gore2':'bbb', 'Gore3':'ccc'}

def updateUsers(user, password):
    #todo check se user o password nulle mandare a pagina errore
    users[user] = password

def loggedin(user, passkey):
    logged = False
    for name, password in users.items():
        if name == user and password == passkey:
            logged = True
    return logged

@app.route('/', methods=['GET'])
def login():
    return render_template('login')

@app.route('/logged/', methods=['POST'])
def logHandler():
    if request.method == 'POST':
        username = request.form['name']
        password = request.form['password']
        if request.form['submit'] == 'SignUp':
            updateUsers(username, password)
            return render_template('success.html')
        if request.form['submit'] == 'Login' and loggedin(username, password):
            return render_template('loggedIn.html', name=username, key=password)
        else:
            return render_template('logError.html')

@app.route('/home/', methods=['GET'])
def home():
    # todo mettere bottone logout
    return render_template('home', users=users)

if __name__ == '__main__':
    app.run(debug=True)
