import sys

from utils import DataSet

def check_stances(d):
    print "Test 3. Check Stances"
    valid_stances = ['agree', 'disagree', 'unrelated', 'discuss']
    for stance in d.stances:
        assert(stance['Stance'] in valid_stances)
    print "Test 3: PASS."

def assert_article_ids_exist(d):
    print "Test 2. Check Consistency"
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
    check_stances(d)

if __name__== "__main__":
    main()
