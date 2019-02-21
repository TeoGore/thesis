import math, random

DATA_SIZE = 100000


def url_has_questionmark(url):
    if url.find('?') != -1:
        return True
    return False


def strip_payload(url):
    index = url.rfind("=")
    return url[:index+1]


def strip_payloads(filepath):
    results = []
    urls = []
    with open(filepath, "r") as input_file:
        urls = input_file.readlines()

    for url in urls:
        if url_has_questionmark(url):
            results.append(strip_payload(url))
        else:
            results.append(url)
    return results


def dork_urls():
    CI = strip_payloads("data/dorks/CI.txt")
    LFI = strip_payloads("data/dorks/LFI.txt")
    SQLi = strip_payloads("data/dorks/SQLi.txt")
    SSI = strip_payloads("data/dorks/SSI.txt")
    XSS = strip_payloads("data/dorks/XSS.txt")
    results = CI + LFI + SQLi + SSI + XSS
    return results

def dork_urls_dict():
    results = {}
    attacks = ["CI", "LFI", "SQLi", "SSI", "XSS"]
    for attack in attacks:
        attack_results = []
        filepath = f'data/dorks/{attack}.txt'
        with open(filepath, "r") as input_file:
            attack_results = input_file.readlines()
        results[attack] = attack_results
    return results

def get_payload(url):
    index = url.find("?")
    return url[index+1:]

def get_good_payloads(filepath):
    results = []
    urls = []
    with open(filepath, "r") as input_file:
        urls = input_file.readlines()

    for url in urls:
        if url_has_questionmark(url):
            results.append(get_payload(url))
    return results

def get_dork_payloads():
    results = {}
    attacks = ["CI", "LFI", "SQLi", "SSI", "X_PATH", "XSS"]
    for attack in attacks:
        attack_results = []
        filepath = f'data/payloads/{attack}.txt'
        with open(filepath, "r") as input_file:
            attack_results = input_file.readlines()
        results[attack] = attack_results
    return results


def create_good_queries(datasize, urls, payloads):
    results = []

    while len(results) < datasize:
        print(f'GOOD DATA CREATED: [{len(results)}]')
        while len(results) < datasize:
            url = random.choice(urls)
            if url_has_questionmark(url):
                payload = random.choice(payloads)
            else:
                payload = ''
            data = url[:-1] + payload  # remove \n from dork
            results.append(data)
        results = list(set(results))  #delete possible duplicate
    random.shuffle(results)
    return results

def file_len(filename):
    count = 0
    with open(filename, "r") as file:
        for count, line in enumerate(file):
            pass
    return count + 1


def calculate_dork_payload_size(attack_size, attack):
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
    if attack == "X_PATH":
        dork_attack = "SQLi"
    else:
        dork_attack = attack

    dork_size = file_len(f"data/dorks/{dork_attack}.txt")
    payload_size = file_len(f"data/payloads/{attack}.txt")

    if dork_size*payload_size < attack_size:
        print(f"Not enough {attack} data! NEEDED: {attack_size}\nADDED ALL POSSIBLE!")
        return dork_size, payload_size

    needed_dorks = math.ceil(math.sqrt((3 / 7) * attack_size))
    needed_payloads = math.ceil(math.sqrt((7 / 3) * attack_size))

    if needed_dorks > dork_size or needed_payloads > payload_size:
        #dont have enough dorks or payloads, so get reault by iterative method
        min_val = min(dork_size, payload_size)
        if min_val == dork_size:
            #low dorks
            needed_dorks = dork_size
            needed_payloads = math.ceil(attack_size/needed_dorks)
        else:
            #low payloads
            needed_payloads = payload_size
            needed_dorks = math.ceil(attack_size/needed_payloads)

    return needed_dorks, needed_payloads


#todo test eliminating an attack in training but not testing
def find_attacks_size(datasize):
    # percentage division: SQLi 30%, CommandInjection5%, LFI 15%, SSI 10%, XPATH 10%, XSS 30%
    attack_percentage = {"CI":0.05, "LFI":0.15, "SQLi": 0.3, "SSI":0.1, "X_PATH":0.1, "XSS":0.3}
    #attack_percentage = {"CI": 0.05, "LFI": 0.15, "SQLi": 0.3, "SSI": 0.1, "X_PATH": 0.1, "XSS": 0.3}
    results = {}
    for attack_key in attack_percentage.keys():
        attack_size = math.ceil(attack_percentage[attack_key]*datasize)
        dork_size, payload_size = calculate_dork_payload_size(attack_size, attack_key)
        results[attack_key] = (dork_size, payload_size)
    return results

def get_datas(needed_datas):
    results = {}# dizionario: attacco:(lista_dork, lista_payload)

    for attack_key in needed_datas.keys():
        if attack_key == "X_PATH":
            dork_file = "SQLi"
        else:
            dork_file = attack_key
        dork_size, payload_size = needed_datas[attack_key]
        dorks = []
        payloads = []
        with open(f"data/dorks/{dork_file}.txt", "r") as dork_file:
            dorks_lines = dork_file.readlines()
            while len(dorks)< dork_size:
                dorks.append(random.choice(dorks_lines))
        with open(f"data/payloads/{attack_key}.txt", "r") as payload_file:
            payloads_lines = payload_file.readlines()
            while len(payloads) < payload_size:
                payloads.append(random.choice(payloads_lines))
        results[attack_key] = (dorks, payloads)
    return results

def create_SQLi(size, datas):
    results = datas
    with open("data/dorks/SQLi.txt", "r") as dorks, open("data/payloads/SQLi.txt", "r") as payloads:
        dorks = dorks.readlines()
        payloads = payloads.readlines()
    while len(results) < size:
        print(f'MORE SQLi DATA CREATED: [{len(results)}]')
        while len(results) < size:
            dork = random.choice(dorks)
            payload = random.choice(payloads)
            data = dork[:-1] + payload  # remove \n from dork
            results.append(data)
        results = list(set(results))  #delete possible duplicate
    return results

def create_bad_datas(needed_datas):
    results = []
    datas = get_datas(needed_datas)
    for attack_key in datas.keys():
        attack_dorks, attack_payloads = datas[attack_key]
        for dork in attack_dorks:
            for payload in attack_payloads:
                results.append(dork[:-1] + payload)


    if len(results)< DATA_SIZE:
        results = create_SQLi(DATA_SIZE, results)

    random.shuffle(results)
    results = results[:DATA_SIZE]
    return results

def create_bad_queries(datasize, dorks, payloads):
    needed_datas = find_attacks_size(datasize)
    return create_bad_datas(needed_datas)


def create_dataset(datasize):
    good_urls_base = strip_payloads("dataset/kdn_url_queries/goodqueries.txt")
    dork_urls_base = dork_urls()
    url_base = good_urls_base + dork_urls_base

    good_payloads = get_good_payloads("dataset/kdn_url_queries/goodqueries.txt")
    dork_urls_dictionary = dork_urls_dict()
    dork_payloads = get_dork_payloads()

    good_queries = create_good_queries(datasize, url_base, good_payloads)
    bad_queries = create_bad_queries(datasize, dork_urls_dictionary, dork_payloads)

    print(f'\nGOOD CREATED: {len(good_queries)}')
    print(f'BAD CREATED: {len(bad_queries)}')

    with open("resultTMP/good.txt", "w") as output_file:
        for data in good_queries:
            if data.endswith("\n"):
                output_file.write(data)
            else:
                index = data.find("\n")
                output_file.write(data[:index] + "\n")
                output_file.write(data[index+1:])

    with open("resultTMP/bad.txt", "w") as output_file:
        for data in bad_queries:
            output_file.write(data)



if __name__=='__main__':
    create_dataset(DATA_SIZE)