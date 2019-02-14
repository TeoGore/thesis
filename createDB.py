import os, random, math

'''
script used to create the bad_query dataset with google dorks
the bad queries will be created concatenating vulnerable path found with dorks with actual attack strings
Vulnerabilities are taken from OWASP top 10 2017:
https://github.com/danielmiessler/SecLists/tree/master/Fuzzing
https://gbhackers.com/latest-google-sql-dorks/
https://hackingvision.com/2017/04/14/google-dorks-for-sql-injection/
https://www.techweed.net/wp-content/uploads/2018/04/Fresh-Google-Dorks-List-2018-For-SQLi-Techweed.pdf
https://github.com/Hood3dRob1n/BinGoo/tree/master/dorks
https://github.com/swisskyrepo/PayloadsAllTheThings/tree/master/SQL%20injection
https://github.com/swisskyrepo/PayloadsAllTheThings
https://github.com/foospidy/payloads/tree/master/owasp/fuzzing_code_database
https://github.com/danielmiessler/SecLists/tree/master/Fuzzing
https://github.com/foospidy/payloads
https://howdofree.altervista.org/cc-db-dork-list.html
http://anonganesh.blogspot.com/2014/06/xss-dorks-list.html
http://howtohackstuff.blogspot.com/2017/03/xss-dorks-list.html

1)per sampling calcolo ogni volta il numero di file per dork e payload
dork*payload e vedere #data
prende in input n dataset, e da n prende n campioni distinti da dork e payload
70 payload
30 dork

2)script che campiona e vediamo quanto ci mette a fare training (se >20 minuti dimezzo dataset)
iniziare con 300K dati e mettendo dataset bad vecchio, quindi nello script principale prendere dalle good query len(badquery) scelte random


3)presentazione tesi (save in PDF)
'''


DATA_SIZE = 300000

def pre_processing():
    #TODO put the main method in here
    pass


def load_file(name):
    directory = str(os.getcwd())
    filepath = os.path.join(directory, name)
    result = []
    with open(filepath, 'r') as f:
        for line in f:
            result.append(line)
    return list(set(result))    # delete duplicate datas

#todo remove method
def load_payload_file(name):
    directory = str(os.getcwd())
    filepath = os.path.join(directory, name)
    result = []
    with open(filepath, 'r') as f:
        for line in f:
            result.append(line[:-1])
    return list(set(result))    # delete duplicate datas



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


def file_len(filename):
    count = 0
    with open(filename, "r") as file:
        for count, line in enumerate(file):
            pass
    return count + 1


def get_dork_payload_filepath(attack):
    # we use same file for SQLi and X_PATH because the attack can be delivered in very similar scenarios
    # for now we don't use xxe since is more used in body not in querystring
    dork_file = {"SQLi":"SQLi", "XSS":"XSS", "LFI":"LFI", "SSI":"SSI", "X_PATH":"SQLi", "CI":"CI"}
    payload_file = {"SQLi":"SQLi", "XSS":"XSS", "LFI":"LFI", "SSI":"SSI", "X_PATH":"X_PATH", "CI":"CI"}
    directory = str(os.getcwd())
    dork_filepath = os.path.join(directory, f'/data/dorks/{dork_file[attack]}.txt')
    payload_filepath = os.path.join(directory, f'/data/payloads/{payload_file[attack]}.txt')
    return  dork_filepath, payload_filepath


def optimise(dorks, payloads, datas_size):
    #function:   (0.3*dorks) * (0.7*payloads) = datas_size
    value = math.ceil(datas_size*0.3*0.7)

    if (dorks * payloads) < datas_size:
        print("Cannot find enough datas")
        exit(-1)

    # try to see if with our dorks we can get enough datas
    needed_payloads = math.ceil(datas_size / (dorks * 0.3 * 0.7))
    if payloads >= needed_payloads:
        return dorks, needed_payloads

    # check with payloads
    needed_dorks = math.ceil(datas_size / (payloads * 0.3 * 0.7))
    if dorks >= needed_dorks:
        return needed_dorks, payloads

    return 0


def find_dork_payload_size(attack_data_size, attack):
    #for an attack we combine 30% dorks and 70% payloads
    dork_filepath, payload_filepath = get_dork_payload_filepath(attack)
    dork_count = file_len(dork_filepath)
    payload_count = file_len(payload_filepath)

    return optimise(dork_count, payload_count, attack_data_size)


def calculate_data_size(attack):
    # percentage division: SQLi 30%, CommandInjection5%, LFI 15%, SSI 10%, XPATH 10%, XSS 30%
    attack_percentage = {"SQLi": 0.3, "XSS":0.3, "LFI":0.15, "SSI": 0.1, "X_PATH":0.1, "CI":0.05}
    attack_data_size = math.ceil(DATA_SIZE * attack_percentage[attack])
    #TODO remove print
    print(f'{attack} - {attack_data_size}')
    return find_dork_payload_size(attack_data_size, attack)


SQLi_dorks, SQLi_payloads = calculate_data_size("SQLi")
XSS_dorks, XSS_payloads = calculate_data_size("XSS")
LFI_dorks, LFI_payloads = calculate_data_size("LFI")
SSI_dorks, SSI_payloads = calculate_data_size("SSI")
X_PATH_dorks, X_PATH_payloads = calculate_data_size("X_PATH")
CommandInj_dorks, CommandInj_payloads = calculate_data_size("CI")

#todo remove prints
print(f'SQLi: {SQLi_dorks}\t-\t{SQLi_payloads}')
print(f'XSS: {XSS_dorks}\t-\t{XSS_payloads}')
print(f'LFI: {LFI_dorks}\t-\t{LFI_payloads}')
print(f'SSI: {SSI_dorks}\t-\t{SSI_payloads}')
print(f'X_PATH: {X_PATH_dorks}\t-\t{X_PATH_payloads}')
print(f'CommandInj: {CommandInj_dorks}\t-\t{CommandInj_payloads}')


def create_datas(attack, dorks_size, payloads_size):
    dork_filepath, payload_filepath = get_dork_payload_filepath(attack)
    result = []
    data_size = dorks_size * payloads_size
    with open(dork_filepath, 'r') as dork_f, open(payload_filepath, 'r') as payload_f:
        dork_list = dork_f.readlines()
        payload_list = payload_f.readlines()
        while len(result) < data_size:
            dork = random.choice(dork_list)
            payload = random.choice(payload_list)
            data = dork[:-1] + payload # remove \n from dork
            if data in result:
                continue        # data already created
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
#then, if out.txt is well-formed, copy and paste in: "bad.txt" in /dataset/myDataset/


#todo risolvere problema quadratico
#todo check file output (if its formatted well)