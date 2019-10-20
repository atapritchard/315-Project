###########################################
# Converts a text file into its word vector
# Alton Olson, Eric Li, Alden Pritchard
# 10/20/19
###########################################

import re
import numpy as np

# Get the bucket of words, number of occurrences, sum of squared error terms
def get_word_bag():
    bag_file = 'bag.txt'
    with open(bag_file, 'r') as bag:
        elems = list(bag_file.read().split('\n'))
    assert len(elems) == 5000
    for i in range(elems):
        elems[i] = elems[i].split(' ')
        elems[i][1], elems[i][2] = float(elems[i][1])/12500, float(elems[i][2])/12500
    return elems

# Converts review text file to vector of size 5000 with values set to word counts
# Order of words is same as order in bag.txt
def vectorize(text_file):
    # removes all non alphabet characters
    def filter_chars(s):
        return re.sub('[^a-zA-Z]+', '', s)
 
    # Get bag of words and intialize word vector
    word_vector = np.zeroes(len(word_bag))
    word_bag = get_word_bag()
    bag_words = list(map(lambda x: x[0], word_bag))
    word_counter = dict()
    for word in word_bag:
        word_counter[word] = 0

    # Get character-filtered file text
    with open(text_file, 'r') as file:
        review = filter_chars(file.read()).split(' ')

    # Count occurrences of each word in review
    for i in range(len(review)):
        if review[i] in bag_words:
            word_counter[review[i]] += 1 
    
    # Set vector values equal to normalized count of word in review
    for i in range(len(word_bag)):
        word_vector[i] = (word_counter[word_bag[i][0]] - word_bag[i][1]) / word_bag[i][2]
    
    return word_vector
