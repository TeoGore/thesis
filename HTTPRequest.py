'''
PARTE NLP

import nlp_analyzer

min_confidence = 75.0
'''

#TODO, finito lo script, importarlo
#from analyzer.nome_script import nome_funzione ES: from analyzer.nlp_word_analyzer import nlp_analyzer

min_confidence = 75.0

#converte a dizionario, vedere se serve
def headers(headers):
    headers_dict = {}
    for header, value in headers:
        headers_dict[header] = value
    return headers_dict

class HTTPRequest():
    def __init__(self, request):

        #todo vedere se serve mettere la richiesta come attributo
        self.method = request.method                                        #string
        self.path = request.path                                            #string
        #TODO mettere urllib.parse per togliere encoding agli url
        self.url = request.url                                              #string
        self.HTTPversion = request.environ['SERVER_PROTOCOL']               #string
        self.headers = headers(request.headers)                                      #dict of strings--> header : value, indicizzabili con self.headers[header_key] o self.headers.get(header_key)
        #di fatto gli header sono una lista di tuple (header, value), li converto con la funzione headers()
        #TODO vedere se la conversione degli headers è utile o meno

        self.body = request.get_data().decode('UTF-8')  # string, a text
        #TODO vedere se si riesce a essere indipendenti dal decode (se non fosse utf-8 come faccio?)
        '''
        Altro modo per prendere il body
        try:
            request_body_size = int(request.environ.get('CONTENT_LENGTH', 0))
        except (ValueError):
            request_body_size = 0

        request_body = request.environ['wsgi.input'].read(request_body_size)
        self.body = request_body.decode('UTF-8')
        '''


    def __str__(self):
        return f'{self.method} request'

    def request_line(self):
        return f'{self.method} {self.path} {self.HTTPversion}'

    def headers_list(self):
        headerList = []
        for header in self.headers.keys():
            value = self.headers.get(header)
            headerList.append(f'{header}: {value}')

        return '\n'.join(headerList)


#TODO se non c'è body stampa \n\n, vedere se fixare
    def raw_request(self):
        return self.request_line() + '\n' + self.headers_list() + '\n\n' +  self.body

    def request_analysis(self):
        (sentiment,confidence) = nlp_analyzer.sentiment(self.request_line())
        if sentiment=='pos' and confidence >= min_confidence:
            return True
        else:
            return False    #bad request or too low confidence
