import os, random, math

'''
script used to create the bad_query dataset with google dorks
the bad queries will be created concatenating vulnerable path found with dorks with actual attack strings
Vulnerabilities are taken from OWASP top 10 2017:
https://gbhackers.com/latest-google-sql-dorks/
https://hackingvision.com/2017/04/14/google-dorks-for-sql-injection/
https://www.techweed.net/wp-content/uploads/2018/04/Fresh-Google-Dorks-List-2018-For-SQLi-Techweed.pdf
https://github.com/danielmiessler/SecLists/tree/master/Fuzzing
https://github.com/Hood3dRob1n/BinGoo/tree/master/dorks
https://github.com/swisskyrepo/PayloadsAllTheThings
https://github.com/foospidy/payloads
https://howdofree.altervista.org/cc-db-dork-list.html
http://anonganesh.blogspot.com/2014/06/xss-dorks-list.html
http://howtohackstuff.blogspot.com/2017/03/xss-dorks-list.html

1)Aumentare payload x_path e SSI data
file sizes: DORK - PAYLOAD
SQLi file sizes: 7369 - 577
XSS file sizes: 146 - 4214
LFI file sizes: 1521 - 11080
SSI file sizes: 14 - 80
X_PATH file sizes: 7369 - 16
CI file sizes: 107 - 8850

2)script che campiona e vediamo quanto ci mette a fare training (se >20 minuti dimezzo dataset)
iniziare con 300K dati e mettendo dataset bad vecchio, quindi nello script principale prendere dalle good query len(badquery) scelte random

3)presentazione tesi (save in PDF)
'''


DATA_SIZE = 600000

def pre_processing():
    #TODO put the main method in here
    pass


def file_len(filename):
    count = 0
    with open(filename, "r") as file:
        for count, line in enumerate(file):
            pass
    return count + 1


def get_dork_payload_filepath(attack):
    # we use same file for SQLi and X_PATH because the attack can be delivered in very similar scenarios
    # for now we don't use xxe since is more used in body not in querystring
    dork_file = {"SQLi":"SQLi", "XSS":"XSS", "LFI":"LFI", "X_PATH":"SQLi", "CI":"CI"}
    payload_file = {"SQLi":"SQLi", "XSS":"XSS", "LFI":"LFI", "X_PATH":"X_PATH", "CI":"CI"}
    directory = str(os.getcwd())
    dork_filepath = os.path.join(directory, f'data/dorks/{dork_file[attack]}.txt')
    payload_filepath = os.path.join(directory, f'data/payloads/{payload_file[attack]}.txt')
    return dork_filepath, payload_filepath


def optimise(dorks, payloads, datas_size):
    '''
    for an attack we want to combine 30% dorks and 70% payloads, so we have the non-linear system:
    A)  dorks*payloads = datas_size
    B)  0.3*(dorks + payloads) = dorks
    C)  0.7*(dorks + payloads) = payloads

    from B and C we get: payloads = (7/3)*dorks; then we combine it with A and we get:
    DORKS:      dorks = sqrt((3/7)*data_size)
    PAYLOADS:   payloads = sqrt((7/3)*data_size)

    then we can ceil to get integers value

    ALSO WE COULD USE AN ITERATIVE METHOD
        needed_dorks = 1
        needed_payloads = 1
        size = needed_dorks * needed_payloads
        while size < datas_size:
            needed_dorks += 1
            needed_payloads = math.ceil((7 / 3) * needed_dorks)
            size = needed_dorks * needed_payloads
    '''

    if (dorks * payloads) < datas_size:
        print(f"Cannot find enough datas - DATA REQUIRED {datas_size}")
        exit(-1)

    needed_dorks = math.ceil(math.sqrt((3 / 7) * datas_size))
    needed_payloads = math.ceil(math.sqrt((7 / 3) * datas_size))

    if needed_dorks > dorks or needed_payloads > payloads:
        #dont have enough dorks or payloads, so get reault by iterative method
        min_val = min(dorks,payloads)
        if min_val == dorks:
            #low dorks
            needed_dorks = dorks
            needed_payloads = math.ceil(datas_size/needed_dorks)
        else:
            #low payloads
            needed_payloads = payloads
            needed_dorks = math.ceil(datas_size/needed_payloads)

    return needed_dorks, needed_payloads


def find_dork_payload_size(attack_data_size, attack):
    dork_filepath, payload_filepath = get_dork_payload_filepath(attack)
    dork_count = file_len(dork_filepath)
    payload_count = file_len(payload_filepath)

    return optimise(dork_count, payload_count, attack_data_size)


def calculate_data_size(attack):
    # percentage division: SQLi 30%, CommandInjection5%, LFI 15%, SSI 10%, XPATH 10%, XSS 30%
    attack_percentage = {"SQLi": 0.3, "XSS":0.3, "LFI":0.15, "X_PATH":0.1, "CI":0.15}
    attack_data_size = math.ceil(DATA_SIZE * attack_percentage[attack])
    print(f'{attack} - {attack_data_size}')
    return find_dork_payload_size(attack_data_size, attack)


def create_datas(attack, dorks_size, payloads_size):
    dork_filepath, payload_filepath = get_dork_payload_filepath(attack)
    result = []
    data_size = dorks_size * payloads_size
    with open(dork_filepath, 'r') as dork_f, open(payload_filepath, 'r') as payload_f:
        dork_list = dork_f.readlines()
        payload_list = payload_f.readlines()

        i=0
        while len(result) < data_size:
            #todo check for infinite loop
            dork = random.choice(dork_list)
            payload = random.choice(payload_list)
            data = dork[:-1] + payload # remove \n from dork

            print(f'[{i}] - generated')
            if data in result:
                continue        # data already created
            result.append(data)
            print(f'[{i}] - ADDED')
            i += 1
    return result


print("*************** ATTACK PARTITIONS ***************")
SQLi_dorks, SQLi_payloads = calculate_data_size("SQLi")
XSS_dorks, XSS_payloads = calculate_data_size("XSS")
LFI_dorks, LFI_payloads = calculate_data_size("LFI")
X_PATH_dorks, X_PATH_payloads = calculate_data_size("X_PATH")
CommandInj_dorks, CommandInj_payloads = calculate_data_size("CI")

print("\n*************** DORKS - PAYLOADS SIZE***************")
print(f'SQLi: {SQLi_dorks}\t-\t{SQLi_payloads}')
print(f'XSS: {XSS_dorks}\t-\t{XSS_payloads}')
print(f'LFI: {LFI_dorks}\t-\t{LFI_payloads}')
print(f'X_PATH: {X_PATH_dorks}\t-\t{X_PATH_payloads}')
print(f'CommandInj: {CommandInj_dorks}\t-\t{CommandInj_payloads}')

#create datas
datas = []
datas = datas + create_datas("SQLi", SQLi_dorks, SQLi_payloads)
datas = datas + create_datas("XSS", XSS_dorks, XSS_payloads)
datas = datas + create_datas("LFI", LFI_dorks, LFI_payloads)
datas = datas + create_datas("X_PATH", X_PATH_dorks, X_PATH_payloads)
datas = datas + create_datas("CI", CommandInj_dorks, CommandInj_payloads)

random.shuffle(datas)
datas = datas[:DATA_SIZE]

print(f"*************** WRITING {DATA_SIZE} QUERYSTRING***************")
i = 0
with open("out.txt", "w") as output_file:
    for data in datas:
        output_file.write(data)
        print(f"[{i}] - {data[:-1]}")
        i +=1

print(f"\nQUERY GENERATED: {len(datas)} ---> CHECK out.txt ")
#then, if out.txt is well-formed, copy and paste in: "bad.txt" in /dataset/myDataset/