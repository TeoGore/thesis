import random

good_results = []


def create_bad_using_all_payloads_only_once():
    results = []

    with open("data/dorks/CI.txt", "r") as d, open("data/payloads/CI.txt", "r") as p:
        CI_dorks = d.readlines()
        CI_payloads = p.readlines()
        for payload in CI_payloads:
            dork = random.choice(CI_dorks)
            data = dork[:-1] + payload
            results.append(data)

    with open("data/dorks/LFI.txt", "r") as d, open("data/payloads/LFI.txt", "r") as p:
        LFI_dorks = d.readlines()
        LFI_payloads = p.readlines()
        for payload in LFI_payloads:
            dork = random.choice(LFI_dorks)
            data = dork[:-1] + payload
            results.append(data)

    with open("data/dorks/SQLi.txt", "r") as d, open("data/payloads/SQLi.txt", "r") as p:
        SQLi_dorks = d.readlines()
        SQLi_payloads = p.readlines()
        for payload in SQLi_payloads:
            dork = random.choice(SQLi_dorks)
            data = dork[:-1] + payload
            results.append(data)

    with open("data/dorks/SSI.txt", "r") as d, open("data/payloads/SSI.txt", "r") as p:
        SSI_dorks = d.readlines()
        SSI_payloads = p.readlines()
        for payload in SSI_payloads:
            dork = random.choice(SSI_dorks)
            data = dork[:-1] + payload
            results.append(data)

    with open("data/dorks/SQLi.txt", "r") as d, open("data/payloads/X_PATH.txt", "r") as p:
        X_PATH_dorks = d.readlines()
        X_PATH_payloads = p.readlines()
        for payload in X_PATH_payloads:
            dork = random.choice(X_PATH_dorks)
            data = dork[:-1] + payload
            results.append(data)

    with open("data/dorks/XSS.txt", "r") as d, open("data/payloads/XSS.txt", "r") as p:
        XSS_dorks = d.readlines()
        XSS_payloads = p.readlines()
        for payload in XSS_payloads:
            dork = random.choice(XSS_dorks)
            data = dork[:-1] + payload
            results.append(data)

        return results


def ALL_DORKS_AND_PAYLOADS():
    dorks_results = []
    payloads_results = []
    with open("data/dorks/CI.txt", "r") as d, open("data/payloads/CI.txt", "r") as p:
        tmp_d = d.readlines()
        tmp_p = p.readlines()
        dorks_results = dorks_results + tmp_d
        payloads_results = payloads_results + tmp_p


    with open("data/dorks/LFI.txt", "r") as d, open("data/payloads/LFI.txt", "r") as p:
        tmp_d = d.readlines()
        tmp_p = p.readlines()
        dorks_results = dorks_results + tmp_d
        payloads_results = payloads_results + tmp_p

    with open("data/dorks/SQLi.txt", "r") as d, open("data/payloads/SQLi.txt", "r") as p:
        tmp_d = d.readlines()
        tmp_p = p.readlines()
        dorks_results = dorks_results + tmp_d
        payloads_results = payloads_results + tmp_p
    with open("data/dorks/SSI.txt", "r") as d, open("data/payloads/SSI.txt", "r") as p:
        tmp_d = d.readlines()
        tmp_p = p.readlines()
        dorks_results = dorks_results + tmp_d
        payloads_results = payloads_results + tmp_p

    with open("data/dorks/SQLi.txt", "r") as d, open("data/payloads/X_PATH.txt", "r") as p:
        tmp_d = d.readlines()
        tmp_p = p.readlines()
        dorks_results = dorks_results + tmp_d
        payloads_results = payloads_results + tmp_p

    with open("data/dorks/XSS.txt", "r") as d, open("data/payloads/XSS.txt", "r") as p:
        tmp_d = d.readlines()
        tmp_p = p.readlines()
        dorks_results = dorks_results + tmp_d
        payloads_results = payloads_results + tmp_p

    #random.shuffle(dorks_results)
    #random.shuffle(payloads_results)

    return dorks_results, payloads_results

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


def create_good(size):
    results = []
    all_dorks, all_payloads = ALL_DORKS_AND_PAYLOADS()
    base = all_dorks + strip_payloads("dataset/kdn_url_queries/goodqueries.txt")
    all_payloads = get_good_payloads("dataset/kdn_url_queries/goodqueries.txt")

    while len(results) < size:
        while len(results) < size:
            dork = random.choice(base)
            payload = random.choice(all_payloads)
            data = dork[:-1] + payload
            results.append(data)
        results = list(set(results))

    return results


bad_results = create_bad_using_all_payloads_only_once()
good_results = create_good(len(bad_results))


with open("resultTMP/bad.txt", "w") as output_file:
    for r in bad_results:
        output_file.write(r)

with open("resultTMP/good.txt", "w") as output_file:
    for r in good_results:
        output_file.write(r)

