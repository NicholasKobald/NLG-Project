#The code based on baseline provided by the FNC organization,
#under the the Apache License
#https://github.com/FakeNewsChallenge/fnc-1-baseline

from csv import DictReader

from nltk.corpus import stopwords
STOP_WORDS = set(stopwords.words('english'))

class DataSet():

    def __init__(self,
                 bodies_fname="train_bodies.csv",
                 stance_fname="train_stances.csv",
                 path="../data_sets"
                ):

        self.path = path

        bodies = "train_bodies.csv"
        stances = "train_stances.csv"

        self.stances = self.read(stances)
        articles = self.read(bodies)

        for s in self.stances:
            s['Body ID'] = int(s['Body ID'])

        self.articles = dict()
        for article in articles:
            self.articles[int(article['Body ID'])] = article['articleBody']

        self.create_article_headline_stance_triples()

    def read(self, filename):
        rows = []
        with open(self.path + "/" + filename, "r",  encoding='utf-8') as table:
            r = DictReader(table)
            for line in r:
                rows.append(line)

        return rows

    #@return string
    def parse_article(self, article):
        return ' '.join([x.lower() for x in article.split() if x not in STOP_WORDS])

    def print_stances(self, print_limit=10):
        print("First", print_limit, "stances")
        for i in range(print_limit):
            print(self.stances[i])

    def print_articles(self, print_limit=10):
        print("First", print_limit, "articles")
        for i in range(print_limit):
            print(self.articles[self.stances[i]['Body ID']])

    def get_stance_counts(self):
        counts = dict(unrelated=0, discuss=0,
            agree=0, disagree=0
        )
        for s in self.stances:
            counts[s['Stance']] += 1

        return counts

    def create_article_headline_stance_triples(self):
        self.triples = dict(
            stances=[],
            articles=[],
            headlines=[]
        )

        for s in self.stances:
            self.triples['stances'].append(s['Stance'])
            self.triples['articles'].append(self.articles[s['Body ID']])
            self.triples['headlines'].append(s['Headline'])


#overkill?
#could encapsulate article, headline pairs and
#store things like overlapp between them nicely
#with an object like this
class Article():
    def __init__(self, article, headline):
        pass
