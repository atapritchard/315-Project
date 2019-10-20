import re

NUM_REVIEWS = 12500 # count of each class
TOTAL_REVIEWS = NUM_REVIEWS * 2 # positive + negative
BAG_SIZE = 5000

def filterchars(s): # removes all non alphabet characters
  s = re.sub('<br', '', s)
  return re.sub('[^a-zA-Z]+', '', s).lower()

bag = {}
fcount = {}

def countreviews(path, i):
  fullpath = path + "/{0}.txt".format(i + 1)
  f = open(fullpath, "r")
  fcount[fullpath] = {}
  for word in f.read().split():
    word = filterchars(word)
    if len(word) == 0:
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
  countreviews("task1/train/negative", i)
  countreviews("task1/train/positive", i)

for key in bag.keys(): # compute averages
  bag[key] = (bag[key][0], bag[key][0] / TOTAL_REVIEWS, 0)

def getvariances(path, i):
  fullpath = path + "/{0}.txt".format(i + 1)
  for word in fcount[fullpath].keys():
    bag[word] = (bag[word][0], bag[word][1], \
    bag[word][2] + (fcount[fullpath][word] - bag[word][1]) ** 2)

for i in range(NUM_REVIEWS): # compute variances
  getvariances("task1/train/negative", i)
  getvariances("task1/train/positive", i)

for key in bag.keys():
  bag[key] = (bag[key][0], bag[key][1], (bag[key][2] / TOTAL_REVIEWS) ** 0.5)

f = open("bag.txt", "w")
words = bag.items()
def getval(x):
  return -x[1][0]
sortedwords = sorted(words, key = getval)
for word in sortedwords[0:BAG_SIZE]: # word is a tuple (word, (occurrences, avg, var))
  f.write("{0} {1:.3f} {2:.3f}\n".format(word[0], word[1][1], word[1][2]))
f.close()