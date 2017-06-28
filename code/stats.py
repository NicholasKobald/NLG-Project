#
#TODO update this to python3
#

import sys

from utils import DataSet

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import WordPunctTokenizer


def count_tokens_in_articles(training_set):
    key_to_info = dict()
    tokenizer = WordPunctTokenizer()
    for key, article in training_set.articles.iteritems():
        if key in key_to_info: continue
        sent_count = 0
        word_count = 0
        my_art = unicode(article, 'utf-8')
        for sent in sent_tokenize(my_art):
            sent_count += 1
            for word in tokenizer.tokenize(sent):
                word_count += 1
        key_to_info[key] = (word_count, sent_count)

    max_sent_length = 0
    max_word_length = 0
    min_sent_length = float('inf')
    min_word_length = float('inf')
    avg_words = 0
    avg_num_sent = 0
    count = 0
    for key, data in key_to_info.iteritems():
        max_sent_length = max(max_sent_length, data[1])
        max_word_length = max(max_word_length, data[0])
        min_word_length = min(min_word_length, data[0])
        min_sent_length = min(min_sent_length, data[1])
        avg_words += data[0]
        avg_num_sent += data[1]
        count += 1

    print "Max Words in Article", max_word_length
    print "Max Number Sentences", max_sent_length
    print "Min words", min_word_length
    print "Min sent", min_sent_length
    print "AVERAGE WORDS:", avg_words * 1.0 / count
    print "AVERAGE Sentences", avg_num_sent * 1.0 / count

#probably should look at things again after we tokenize
def report_article_info(training_set):
    max_article_length = 0
    min_article_length = float('inf')
    num_articles = 0
    total_len = 0
    for key, article in training_set.articles.iteritems():
        num_articles += 1
        if len(article) < 10:
            print "SHORT:", article
        max_article_length = max(len(article), max_article_length)
        min_article_length = min(min_article_length, len(article))

    average_article_length = 1.0 * total_len / len(training_set.articles) * 1.0
    print "Max article Length:", max_article_length
    print "Min Article Length:", min_article_length
    print "Average article length:", average_article_length
    print "Num articles two:", num_articles

def double_check_stance_counts(training_set):
    total_count = 0
    for s in training_set.stances:
        total_count += 1
    print "Exactly", total_count, "stances"

def report_stance_counts(training_set):
    print "Total Stances: {}".format(len(training_set.stances))
    print "Stance Counts:\n"
    counts = training_set.get_stance_counts()
    for stance, count in counts.iteritems():
        print stance, "occurs", count, "times."
    print "\nStance percentages:"
    for stance, count in counts.iteritems():
        print "{} compromised {} of stances".format(stance, (1.0 * count / len(training_set.stances) * 1.0))

def report_article_counts(training_set):
    print "{} articles in the training set".format(len(training_set.articles))

def main():
    d_set = DataSet()
    report_stance_counts(d_set)
    report_article_counts(d_set)
    report_article_info(d_set)
    count_tokens_in_articles(d_set)
    #double_check_stance_counts(d_set)

if __name__=="__main__":
    main()
