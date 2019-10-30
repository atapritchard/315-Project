import re
import os
from os.path import isfile, join

NUM_REVIEWS = 12500 # count of each class
TOTAL_REVIEWS = NUM_REVIEWS * 2 # positive + negative
BAG_SIZE = 5000
mypath = "../task1/train"

def filterchars(s): # removes all non alphabet characters
  s = re.sub('<br', '', s)
  return re.sub('[^a-zA-Z]+', '', s).lower()

bag = {}
fcount = {}

def countreviews(path, i):
  with open('stopwords.txt') as f:
    sw = [word for line in f for word in line.split()]

  fullpath = path + "/{0}.txt".format(i + 1)
  f = open(fullpath, "r", encoding = "utf8")
  fcount[fullpath] = {}
  for word in f.read().split():
    word = filterchars(word)
    if len(word) == 0 or word in sw:
      continue
    if word in bag:
      bag[word] = (bag[word][0] + 1, 0, 0)
    else:
      bag[word] = (1, 0, 0)
    if word in fcount[fullpath]:
      fcount[fullpath][word] += 1
    else:
      fcount[fullpath][word] = 1
  f.close()

for i in range(NUM_REVIEWS):
  countreviews(mypath + "/negative", i)
  countreviews(mypath + "/positive", i)

for key in bag.keys(): # compute averages
  bag[key] = (bag[key][0], bag[key][0] / TOTAL_REVIEWS, 0)

def getvariances(path, i):
  fullpath = path + "/{0}.txt".format(i + 1)
  for word in fcount[fullpath].keys():
    bag[word] = (bag[word][0], bag[word][1], \
    bag[word][2] + (fcount[fullpath][word] - bag[word][1]) ** 2)

for i in range(NUM_REVIEWS): # compute variances
  getvariances(mypath + "/negative", i)
  getvariances(mypath + "/positive", i)

for key in bag.keys():
  bag[key] = (bag[key][0], bag[key][1], (bag[key][2] / TOTAL_REVIEWS) ** 0.5)

f = open("bag.txt", "w", encoding = "utf8")
words = bag.items()
def getval(x):
  return -x[1][0]
sortedwords = sorted(words, key = getval)
for word in sortedwords[0:min(BAG_SIZE, len(sortedwords))]: # word is a tuple (word, (occurrences, avg, var))
  f.write("{0} {1:.3f} {2:.3f}\n".format(word[0], word[1][1], word[1][2]))
f.close()