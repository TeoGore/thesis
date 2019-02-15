from query_string_analyzer import sentiment as query_sentiment


min_confidence = 75.0


# converte a dizionario, vedere se serve
def headers(headers):
    headers_dict = {}
    for header, value in headers:
        headers_dict[header] = value
    return headers_dict


class HTTPRequest():
    def __init__(self, request):

        self.method = request.method                                        # string
        self.path = request.path                                            # string
        self.url = request.url                                              # string
        self.HTTPversion = request.environ['SERVER_PROTOCOL']               # string
        self.headers = headers(request.headers)                                      # dict of strings--> header : value, indicizzabili con self.headers[header_key] o self.headers.get(header_key)
        #di fatto gli header sono una lista di tuple (header, value), li converto con la funzione headers()

        self.body = request.get_data().decode('UTF-8')  # string, a text
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


# TODO se non c'è body stampa \n\n
    def raw_request(self):
        return self.request_line() + '\n' + self.headers_list() + '\n\n' + self.body


    def request_analysis(self):
        (sentiment,confidence) = query_sentiment(self.request_line())
        if sentiment=='pos' and confidence >= min_confidence:
            return True
        else:
            return False    #bad request or too low confidence
