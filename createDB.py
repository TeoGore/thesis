import os, random, math

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
https://github.com/swisskyrepo/PayloadsAllTheThings
https://github.com/foospidy/payloads/tree/master/owasp/fuzzing_code_database
https://github.com/danielmiessler/SecLists/tree/master/Fuzzing

https://github.com/foospidy/payloads

3 - Sensitive Data Exposure
https://howdofree.altervista.org/cc-db-dork-list.html

4 - XXE (XML External Entities)

7 - XSS (Cross-Site Scripting)
dorks:
http://anonganesh.blogspot.com/2014/06/xss-dorks-list.html
http://howtohackstuff.blogspot.com/2017/03/xss-dorks-list.html
'''


'''

0)per x_path si puo usare il dataset di sqli
per command injection si usano i path con parametro cmd= o command=
xxe per ora lasciato perdere

1)per sampling calcolo ogni volta il numero di file per dork e payload
dork*payload e vedere #data

2)prende in input n dataset, e da n prende n campioni distinti da dork e payload
70 payload
30 dork

3a)script che campiona e vediamo quanto ci mette a fare training (se >20 min dimezzo dataset)
iniziare con 300K e mettere dataset bad vecchio


4)presentazione (libre office)

'''

DATA_SIZE = 300000

def pre_processing():
    pass


def load_file(name):
    directory = str(os.getcwd())
    filepath = os.path.join(directory, name)
    result = []
    with open(filepath, 'r') as f:
        for line in f:
            result.append(line)
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



tmp_dork = load_file("tmp_dork.txt")
tmp_payload = load_file("tmp_payload.txt")

print(len(tmp_dork)) #3130
print(len(tmp_payload)) #3130

for elem in tmp_dork:
    print(elem)
print('\n\***************\n')

for elem in tmp_payload:
    print(elem)
print('\n\***************\n')

results = []
for dork in tmp_dork:
    for payload in tmp_payload:
        elem = dork[:-1]+payload
        results.append(elem)
        #print(elem)

with open("bad_query_dataset.txt", 'w') as output_file:
    for result in results:
        output_file.write(result)

#****************************************************

def find_dork_payload_size(attack_data_size):
    #for an attack we combine 30% dorks and 70% payloads
    return (math.ceil(attack_data_size * 0.3), math.ceil(attack_data_size * 0.7))


def calculate_data_size(data_size, attack):
    # percentage division: SQLi 30%, CommandInjection5%, LFI 15%, SSI 10%, XPATH 10%, XSS 30%
    attack_percentage = {"SQLi": 0.3, "XSS":0.3, "LFI":0.15, "SSI": 0.1, "X_PATH":0.1, "CI":0.05}
    attack_data_size = math.ceil(data_size * attack_percentage[attack])
    #TODO remove print
    print(f'{attack} - {attack_data_size}')
    return find_dork_payload_size(attack_data_size)


SQLi_dorks, SQLi_payloads = calculate_data_size(DATA_SIZE, "SQLi")
XSS_dorks, XSS_payloads = calculate_data_size(DATA_SIZE, "XSS")
LFI_dorks, LFI_payloads = calculate_data_size(DATA_SIZE, "LFI")
SSI_dorks, SSI_payloads = calculate_data_size(DATA_SIZE, "SSI")
X_PATH_dorks, X_PATH_payloads = calculate_data_size(DATA_SIZE, "X_PATH")
CommandInj_dorks, CommandInj_payloads = calculate_data_size(DATA_SIZE, "CI")

print(f'SQLi: {SQLi_dorks}\t-\t{SQLi_payloads}')
print(f'XSS: {XSS_dorks}\t-\t{XSS_payloads}')
print(f'LFI: {LFI_dorks}\t-\t{LFI_payloads}')
print(f'SSI: {SSI_dorks}\t-\t{SSI_payloads}')
print(f'X_PATH: {X_PATH_dorks}\t-\t{X_PATH_payloads}')
print(f'CommandInj: {CommandInj_dorks}\t-\t{CommandInj_payloads}')

def create_datas(attack, dorks_size, payloads_size):
    #we use same file for SQLi and X_PATH because the attack can be delivered in very similar scenarios
    dork_file = {"SQLi":"SQLi", "XSS":"XSS", "LFI":"LFI", "SSI":"SSI", "X_PATH":"SQLi", "CI":"CI"}
    payload_file = {"SQLi":"SQLi", "XSS":"XSS", "LFI":"LFI", "SSI":"SSI", "X_PATH":"X_PATH", "CI":"CI"}
    directory = str(os.getcwd())
    dork_filepath = os.path.join(directory, f'/data/dorks/{dork_file[attack]}.txt')
    payload_filepath = os.path.join(directory, f'/data/payloads/{payload_file[attack]}.txt')
    result = []
    data_size = dorks_size * payloads_size
    with open(dork_filepath, 'r') as dork_f, open(payload_filepath, 'r') as payload_f:
        dork_list = dork_f.readlines()
        payload_list = payload_f.readlines()
        while len(result)< data_size:
            dork = random.choice(dork_list)
            payload = random.choice(payload_list)
            data = dork + payload
            if data in result:
                continue #data already created
            result.append(data)
    return result

datas = []
datas = datas + create_datas("SQLi", SQLi_dorks, SQLi_payloads)
datas = datas + create_datas("XSS", XSS_dorks, XSS_payloads)
datas = datas + create_datas("LFI", LFI_dorks, LFI_payloads)
datas = datas + create_datas("SSI", SSI_dorks, SSI_payloads)
datas = datas + create_datas("X_PATH", X_PATH_dorks, X_PATH_payloads)
datas = datas + create_datas("CI", CommandInj_dorks, CommandInj_payloads)

with open("out.txt", "w") as output_file:
    for data in datas:
        output_file.write(data)

#TODO controllare le risorse che finiscono con "="
#todo creare file con dork di command injection

#with open("data/dorks/lfi_dorks.txt", "r") as LFI_dork, open("data/dorks/sqli_dorks.txt", "r") as SQLi_dork, open("data/dorks/ssi_dorks.txt", "r") as SSI_dork, open("data/dorks/xss_dorks.txt", "r") as XSS_dork:
#    pass



