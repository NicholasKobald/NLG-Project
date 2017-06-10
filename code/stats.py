import sys

from utils import DataSet


#probably should look at things again after we tokenize
def report_article_info(training_set):
    max_article_length = 0
    min_article_length = float('inf')
    total_len = 0
    for key, article in training_set.articles.iteritems():
        max_article_length = max(len(article), max_article_length)
        min_article_length = min(min_article_length, len(article))
        total_len += len(article)

    average_article_length = 1.0 * total_len / len(training_set.articles) * 1.0
    print "Max article Length:", max_article_length
    print "Min Article Length:", min_article_length
    print "Average article length:", average_article_length


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

if __name__=="__main__":
    main()
