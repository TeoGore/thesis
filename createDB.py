import os

'''
script used to create the bad_query dataset with google dorks
the bad queries will be created concatenating vulnerable path found with dorks with actual attack strings

https://github.com/danielmiessler/SecLists/tree/master/Fuzzing

Vulnerabilities are taken from OWASP top 10 2017:
1 - Injections

dorks from: https://gbhackers.com/latest-google-sql-dorks/
https://hackingvision.com/2017/04/14/google-dorks-for-sql-injection/
https://www.techweed.net/wp-content/uploads/2018/04/Fresh-Google-Dorks-List-2018-For-SQLi-Techweed.pdf
https://github.com/Hood3dRob1n/BinGoo/tree/master/dorks

payloads from:
https://github.com/swisskyrepo/PayloadsAllTheThings/tree/master/SQL%20injection

3 - Sensitive Data Exposure
https://howdofree.altervista.org/cc-db-dork-list.html

4 - XXE (XML External Entities)

7 - XSS (Cross-Site Scripting)
dorks:
http://anonganesh.blogspot.com/2014/06/xss-dorks-list.html
http://howtohackstuff.blogspot.com/2017/03/xss-dorks-list.html
'''

def preprocessing():
    pass
'''

2)prende in input n dataset, e da n prende n campioni distinti da dork e payload
70 payload
30 dork

2a) dork*payload e vedere #data

3a)script che campiona e vediamo quanto ci mette a fare training (se >20 min dimezzo dataset)
iniziare con 300K e mettere dataset bad vecchio

1)sostituire i .php con:
#.php, .asp, .aspx, .jsp e anche togliere estensione

per sampling calcolo ogni volta il numero di file per dork e payload

4)presentazione (libre office)

PREWSENTAZIONE:
inzio con logo, mionome, nome massari

outline: elenco ountato che dice di cosa parlero
-introduzione
-altri punti
...
-conclusioni

-introduzione:  modulo per WAF
spiegare cosa è,  perche serve un WAF
-Modsecurity
-basati su regole (facile bypass)

sfruttare conoscenze ML per sviluppare WAF che posso generalizzare su attacchi nuovi

ML si applica bene in:applicazioni reali:CV, NLP, (altrimenti si usa il DL)... (in generale quando costruire modello matematico)

parlare di parte di preprocessing
processing

slide finel di ringraziamento
'''


def load_dork_file(name):
    directory = str(os.getcwd())
    filepath = os.path.join(directory, name)
    result = []
    with open(filepath, 'r') as f:
        for line in f:
            result.append(line[:-1]) #remove double \n at the end of every line
    return list(set(result))    # delete duplicate datas

def load_payload_file(name):
    directory = str(os.getcwd())
    filepath = os.path.join(directory, name)
    result = []
    with open(filepath, 'r') as f:
        for line in f:
            result.append(line[:-1])
    return list(set(result))    # delete duplicate datas


#todo controllare: % spazi ” “ e altri caratteri strani nei file di
#sqli_dork = load_dork_file("sqli_dorks.txt")
#sqli_payload = load_payload_file("sqli_payloads.txt")



sqli_dork = load_dork_file("tmp_dork.txt")
sqli_payload = load_payload_file("tmp_payload.txt")

print(len(sqli_dork)) #3130
print(len(sqli_payload)) #3130

for elem in sqli_dork:
    print(elem)

print('\n\***************\n')

for elem in sqli_payload:
    print(elem)

print('\n\***************\n')

result = []
for dork in sqli_dork:
    for payload in sqli_payload:
        elem  = dork+payload
        result.append(elem)
        print(elem)


with open("bad_query_dataset.txt", 'w') as output_file:
    output_file.write('\n'.join(["a", "b"]))
