###########################################
# Converts a text file into its word vector
# Alton Olson, Eric Li, Alden Pritchard
# 10/20/19
###########################################

import re
import numpy as np

# Get the bucket of words
def get_word_bag():
    bag_file = 'bag.txt'
    with open(bag_file, 'r') as bag:
        words = list(bag_file.read().split('\n'))
    assert len(words) == 5000
    return words

# Parameters:
# text_file: 'review_number.txt' (string)
# word_bucket: {'good', 'bad', ...} (set)
# Returns:
# vector: [0, 3, 5, 2, ...] (int list)
def vectorize(text_file, word_bucket):
    # removes all non alphabet characters
    def filter_chars(s):
        return re.sub('[^a-zA-Z]+', '', s)
 
    # Get bag of words and intialize word vector
    word_vector = np.zeroes(len(word_bag))
    word_bag = get_word_bag()
    word_counter = dict()
    for word in word_bag:
        word_counter[word] = 0

    # Get character-filtered file text
    with open(text_file, 'r') as file:
        review = filter_chars(file.read()).split(' ')

    # Count occurrences of each word in review
    for i in range(len(review)):
        if review[i] in word_bag:
            word_counter[review[i]] += 1 
    
    # Set vector values equal to count of word in review
    for i in range(len(word_bag)):
        word_vector[i] = word_counter[word_bag[i]]
    
    return word_vector

