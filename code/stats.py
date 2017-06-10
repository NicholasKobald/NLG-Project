import sys

from utils import DataSet


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

if __name__=="__main__":
    main()
