import sys

from utils import DataSet


def assert_article_ids_exist(d):
    print "Test 2."
    for stance in d.stances:
        assert(stance['Body ID'] in d.articles)
    print "Test 2: PASS. Stance IDs are consistent with articles"

def read_dataset():
    print "Test 1: Read Input"
    test_data_set = DataSet()
    #test_data_set.print_stances()
    #test_data_set.print_articles()
    print "Test 1: PASS. Read input without errors."
    return test_data_set

def main():
    d = read_dataset()
    assert_article_ids_exist(d)

if __name__== "__main__":
    main()
