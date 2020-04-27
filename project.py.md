
#                                            Problem Statement
Our goal is to look at transcripts of various comedians and note their similarities and differences. Specifically, I'd like to know if Ali Wong's comedy style is different than other comedians

                                            Getting The Data

Luckily, there are wonderful people online that keep track of stand up routine transcripts. Scraps From The Loft makes them available for non-profit and educational purposes.

To decide which comedians to look into, I went on IMDB and looked specifically at comedy specials that were released in the past 5 years. To narrow it down further, I looked only at those with greater than a 7.5/10 rating and more than 2000 votes. If a comedian had multiple specials that fit those requirements, I would pick the most highly rated one. I ended up with a dozen comedy specials.


```python
#web scraping,pickle imports
import requests
from bs4 import BeautifulSoup
import pickle
```


```python
#scrapes transcript data from scrapsfromtheloft.com
def url_to_transcript(url):
    #Returns transcript data specifically from scrapsfrometheloft.com
    page = requests.get(url).text
    soup = BeautifulSoup(page,"lxml")
    text = [p.text for p in soup.find(class_="post-content").find_all('p')]
    print(url)
    return text
```


```python
#url of transcipt in scope
urls = ['http://scrapsfromtheloft.com/2017/05/06/louis-ck-oh-my-god-full-transcript/',
        'http://scrapsfromtheloft.com/2017/04/11/dave-chappelle-age-spin-2017-full-transcript/',
        'http://scrapsfromtheloft.com/2018/03/15/ricky-gervais-humanity-transcript/',
        'http://scrapsfromtheloft.com/2017/08/07/bo-burnham-2013-full-transcript/',
        'http://scrapsfromtheloft.com/2017/05/24/bill-burr-im-sorry-feel-way-2014-full-transcript/',
        'http://scrapsfromtheloft.com/2017/04/21/jim-jefferies-bare-2014-full-transcript/',
        'http://scrapsfromtheloft.com/2017/08/02/john-mulaney-comeback-kid-2015-full-transcript/',
        'http://scrapsfromtheloft.com/2017/10/21/hasan-minhaj-homecoming-king-2017-full-transcript/',
        'http://scrapsfromtheloft.com/2017/09/19/ali-wong-baby-cobra-2016-full-transcript/',
        'http://scrapsfromtheloft.com/2017/08/03/anthony-jeselnik-thoughts-prayers-2015-full-transcript/',
        'http://scrapsfromtheloft.com/2018/03/03/mike-birbiglia-my-girlfriends-boyfriend-2013-full-transcript/',
        'http://scrapsfromtheloft.com/2017/08/19/joe-rogan-triggered-2016-full-transcript/']
#comedian names
comedians = ['louis','dave','ricky','bo','bill','jim','john','hasan','ali','anthony','mike','joe']
```


```python
transcripts = [url_to_transcript(u) for u in urls]

```


```python
import os
os.mkdir("C:\\Users\\Oliver\\Untitled Folder\\transcripts")

```


```python
for i,c in enumerate(comedians):
    with open("transcripts/" + c + ".txt","wb")as file:
        pickle.dump(transcripts[i],file)
```


```python
#load pickled files
data = {}
for i, c in  enumerate(comedians):
 with open("transcripts/" + c + ".txt","rb")as file:
    data[c] = pickle.load(file)
```


```python
#double check to make sure  data has been loaded properly
data.keys()
```


```python
#more checks
data['louis'][:2]
```

#                                      CLEANING THE DATA


```python
#lETS look at the data again
next(iter(data.keys()))
```


```python
#Notice that our dictionary is currently in key"comedian,value:list of text format
next(iter(data.values()))
```


```python
#I am going to change  this to a key:comedian,value:string format
def combine_text(list_of_text):
    #Takes a list of text and combines then into one large chunk of text
    combined_text = ' '.join(list_of_text)
    return combined_text
```


```python
#combine it
data_combined = {key:[combine_text(value)] for (key,value) in data.items()}
data_combined
```


```python
# we can either keep it in dictionary format or put into a pandas dataframe
import pandas as pd
pd.set_option('max_colwidth',150)
data_df = pd.DataFrame.from_dict(data_combined).transpose()
data_df.columns = ['transcript']
data_df = data_df.sort_index()
data_df
```


```python
#lET'S look at the  transcript for ali wong
data_df.transcript.loc['ali']
```


```python
#Apply first round of text  cleaning techniques
#Make text lowercase,remove text in square brackets,remove punctaution and remove words containing numbers
import re
import string
def clean_text_round1(text):
    text = text.lower()
    text = re.sub('\[.*?\]','',text)
    text = re.sub('[%s]'%re.escape(string.punctuation),'',text)
    text = re.sub('\w*\d\w*','',text)
    return text
round1 = lambda x: clean_text_round1(x)
    
```


```python
#Lets look at the updated text
data_clean = pd.DataFrame(data_df.transcript.apply(round1))
data_clean
```


```python
#Applying second round of cleaning
#Geeting rid of some additional punctuation and non-sensical text that was missed the first time around
def clean_text_round2(text):
    text = re.sub('\n','',text)
    return text
round2 = lambda x: clean_text_round2(x)

```


```python
#Lets look at the updated text
data_clean = pd.DataFrame(data_clean.transcript.apply(round2))
data_clean
```

                                               Organizing The Data
                                               


```python
#Lets add comedian fullname as well
full_names = ['Ali Wong', 'Anthony Jeselnik', 'Bill Burr', 'Bo Burnham', 'Dave Chappelle', 'Hasan Minhaj',
              'Jim Jefferies', 'Joe Rogan', 'John Mulaney', 'Louis C.K.', 'Mike Birbiglia', 'Ricky Gervais']
data_df['full_name']= full_names
data_df
```


```python
#Let's pickle it for later use
data_df.to_pickle("corpus.pkl")
```

#                                    Document Term Matrix
                                            
 For many of the techniques we'll be using in future notebooks, the text must be tokenized, meaning broken down into smaller pieces. The most common tokenization technique is to break down text into words. We can do this using scikit-learn's CountVectorizer, where every row will represent a different document and every column will represent a different word.

In addition, with CountVectorizer, we can remove stop words. Stop words are common words that add no additional meaning to text such as 'a', 'the', etc.                                         


```python
#we are going to create a DTM using count vectorizer  and exclude common english stop words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words='english')
data_cv = cv.fit_transform(data_clean.transcript)
data_dtm = pd.DataFrame(data_cv.toarray(),columns=cv.get_feature_names())
data_dtm.index = data_clean.index
data_dtm
```


```python
#lET'S pickle it for later use
data_dtm.to_pickle("dtm.pkl")
```


```python
#Lets also pickled the cleaned data(before we put in DTM)and the countvectorizer object
data_clean.to_pickle('data_clean.pkl')
pickle.dump(cv,open("cv.pkl","wb"))
```

#                             EXPLORATOY DATA ANALYSIS
                                        
After the data cleaning step where we put our data into a few standard formats, the next step is to take a look at the data and see if what we're looking at makes sense. Before applying any fancy algorithms, it's always important to explore the data first.

When working with numerical data, some of the exploratory data analysis (EDA) techniques we can use include finding the average of the data set, the distribution of the data, the most common values, etc. The idea is the same when working with text data. We are going to find some more obvious patterns with EDA before identifying the hidden patterns with machines learning (ML) techniques. We are going to look at the following for each comedian:

Most common words - find these and create word clouds
Size of vocabulary - look number of unique words and also how quickly someone speaks
Amount of profanity - most common terms


```python
#Most cmmon words
#Read in the DTM
import pandas as pd
data = pd.read_pickle('dtm.pkl')
data = data.transpose()
data.head()
```


```python
#Find the top 30 words said by each comedian
top_dict = {}
for c in data.columns:
    top = data[c].sort_values(ascending=False).head(30)
    top_dict[c]=list(zip(top.index,top.values))
top_dict
```


```python
#top 15 words said by each comedian
for comedian, top_words in top_dict.items():
    print(comedian)
    print(','.join([word for word,count in top_words[0:14]]))
    print('---')
```

NOTE: At this point, we could go on and create word clouds. However, by looking at these top words, you can see that some of them have very little meaning and could be added to a stop words list, so let's do just that.


```python
#common words adding to stop words list
#Let's first pull out top 30  words by each comedian
from collections import Counter
words =[]
for comedian in data.columns:
    top = [word for (word,count)in top_dict[comedian]]
    for t in top:
        words.append(t)
words
```


```python
# Let's aggregate this list and identify the most common words along with how many routines they occur in
Counter(words).most_common()
```


```python
#If more than half of the comdians have it as a top word,exclude it from the list
add_stop_words = [word for word,count in Counter(words).most_common() if count > 6]
add_stop_words
```


```python
#Let's update our document-term matrix with the new list of stop words
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer

#Read in cleaned data
data_clean = pd.read_pickle('data_clean.pkl')

#Add new stop words
stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

#Recreate DTM
cv = CountVectorizer(stop_words=stop_words)
data_cv = cv.fit_transform(data_clean.transcript)
data_stop = pd.DataFrame(data_cv.toarray(),columns=cv.get_feature_names())
data_stop.index=data_clean.index

#pickle it for later use
import pickle
pickle.dump(cv,open("cv_stop.pkl","wb"))
data_stop.to_pickle("dtm_stop.pkl")

```


```python
#lets make some world clouds
from wordcloud  import WordCloud
wc = WordCloud(stopwords=stop_words,background_color="white",colormap="Dark2",max_font_size=150,random_state=42)
wc
```


```python
#Reset the output dimensions
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams['figure.figsize'] = [16,6]
full_names = ['Ali Wong', 'Anthony Jeselnik', 'Bill Burr', 'Bo Burnham', 'Dave Chappelle', 'Hasan Minhaj',
              'Jim Jefferies', 'Joe Rogan', 'John Mulaney', 'Louis C.K.', 'Mike Birbiglia', 'Ricky Gervais']
#create subplots for each comedian
for index, comedian in enumerate(data.columns):
    wc.generate(data_clean.transcript[comedian])
    plt.subplot(3,4,index+1)
    plt.imshow(wc,interpolation="bilinear")
    plt.axis("off")
    plt.title(full_names[index])
    
plt.show()
    
```

                                         FINDINGS
  Ali Wong says the s-word a lot and talks about her husband.
  A lot of people use the F-word. Let's dig into that later.

                                         NUMBER OF WORDS                             


```python
# Find the number of unique words that each comedian uses
unique_list =[]
for comedian in data.columns:
    uniques = data[comedian].nonzero()[0].size
    unique_list.append(uniques)
#creating a new dataframe that contains the unique word count
data_words = pd.DataFrame(list(zip(full_names,unique_list)),columns=['comedian','unique_words'])
data_unique_sort = data_words.sort_values(by='unique_words')
data_unique_sort
```


```python
#calculating the words per minute for each comedia
#find the total no of words  that a comedian uses
total_list = []
for comedian in data.columns:
    totals = sum(data[comedian])
    total_list.append(totals)

# comedy special run times from IMDB in minutes
run_times = [60, 59, 80, 60, 67, 73, 77, 63, 62, 58, 76, 79]

#Let's add some column to dataframes
data_words['total_words'] = total_list
data_words['run_times'] = run_times
data_words['words_per_minute'] = data_words['total_words'] / data_words['run_times']

#sort the dataframe by words per minute to see who talks the slowest and fastest
data_wpm_sort = data_words.sort_values(by='words_per_minute')
data_wpm_sort
```


```python
# Let's plot our findings
import numpy as np
y_pos = np.arange(len(data_words))
plt.subplot(1,2,1)
plt.barh(y_pos,data_unique_sort.unique_words,align='center')
plt.yticks(y_pos,data_unique_sort.comedian)
plt.title('Number of unique words',fontsize = 20)

plt.subplot(1,2,2)
plt.barh(y_pos,data_wpm_sort.words_per_minute,align='center')
plt.yticks(y_pos,data_wpm_sort.comedian)
plt.title('Number of words per minute',fontsize=20)

plt.tight_layout()
plt.show()

```

                                                 Findings
Vocabulary

Ricky Gervais (British comedy) and Bill Burr (podcast host) use a lot of words in their comedy

Louis C.K. (self-depricating comedy) and Anthony Jeselnik (dark humor) have a smaller vocabulary
Talking Speed

Joe Rogan (blue comedy) and Bill Burr (podcast host) talk fast

Bo Burnham (musical comedy) and Anthony Jeselnik (dark humor) talk slow

Ali Wong is somewhere in the middle in both cases

                                                Amount of Profanity


```python
#Let's take a look at the most common words again
Counter(words).most_common()
```


```python
#Let's isolate this bad words
data_bad_words = data.transpose()[['fucking', 'fuck', 'shit']]
data_profanity = pd.concat([data_bad_words.fucking + data_bad_words.fuck,data_bad_words.shit],axis =1)
data_profanity.columns = ['f_word','s_word']
data_profanity
```


```python
#let's create a scatter plot 
plt.rcParams['figure.figsize']=[10,8]
for i,comedian in enumerate(data_profanity.index):
    x = data_profanity.f_word.loc[comedian]
    y = data_profanity.s_word.loc[comedian]
    plt.scatter(x,y,color='blue')
    plt.text(x+1.5,y+0.5,full_names[i],fontsize=10)
    plt.xlim(-5,155)
plt.title('Number of bad words used in routine',fontsize=20)
plt.xlabel('Number of F words',fontsize = 15)
plt.ylabel('Number of S words',fontsize = 15)
```

                                              Findings
                                    
Averaging 2 F-Words Per Minute! - I don't like too much swearing, especially the f-word, which is probably why I've never heard of Bill Bur, Joe Rogan and Jim Jefferies.
Clean Humor - It looks like profanity might be a good predictor of the type of comedy I like. Besides Ali Wong, my two other favorite comedians in this group are John Mulaney and Mike Birbiglia.

#                                    Sentiment Analysis
    
    Polarity: How positive or negative a word is. -1 is very negative. +1 is very positive.
    Subjectivity: How subjective, or opinionated a word is. 0 is fact. +1 is very much an opinion.


```python
# We'll start by reading in the corpus, which preserves word order
import pandas as pd
data = pd.read_pickle('corpus.pkl')
data
```


```python
# Create quick lambda functions to find the polarity and subjectivity of each routine
# Terminal / Anaconda Navigator: conda install -c conda-forge textblob

from textblob import TextBlob
pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity

data['polarity'] = data['transcript'].apply(pol)
data['subjectivity'] = data['transcript'].apply(sub)
data
```


```python
#Let's plot the results
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [10, 8]

for index, comedian in enumerate(data.index):
    x = data.polarity.loc[comedian]
    y = data.subjectivity.loc[comedian]
    plt.scatter(x, y, color='blue')
    plt.text(x+.001, y+.001, data['full_name'][index], fontsize=10)
    plt.xlim(-.01, .12) 
    
plt.title('Sentiment Analysis', fontsize=20)
plt.xlabel('<-- Negative -------- Positive -->', fontsize=15)
plt.ylabel('<-- Facts -------- Opinions -->', fontsize=15)

plt.show()


    
```


```python
# Sentiment of Routine Over Time
#Instead of looking at the overall sentiment, 
#let's see if there's anything interesting about the sentiment over time throughout each routine

#split each routine into 10 parts
import numpy as np
import math

def split_text(text, n=10):
#Takes in a string of text and splits into n equal parts, with a default of 10 equal parts.
# Calculate length of text, the size of each chunk of text and the starting points of each chunk of text
    length = len(text)
    size = math.floor(length / n)
    start = np.arange(0, length, size)
    
    #pull out equally sized pieces of text and put it into a list
    split_list = []
    for piece in range(n):
        split_list.append(text[start[piece]:start[piece]+size])
    return split_list
```


```python
# Let's take a look at our data again
data
```


```python
#Let's create  a list to hold all of the pieces of text
#update numpy and conda
list_pieces = []
for t in data.transcript:
    split = split_text(t)
    list_pieces.append(split)
list_pieces
```


```python
# The list has 10 elements, one for each transcript
len(list_pieces)

```


```python
# Each transcript has been split into 10 pieces of text
len(list_pieces[0])
```


```python
# Calculate the polarity for each piece of text
polarity_transcript = []
for lp in list_pieces:
    polarity_piece=[]
    for p in lp:
        polarity_piece.append(TextBlob(p).sentiment.polarity)
    polarity_transcript.append(polarity_piece)
polarity_transcript
```


```python
# Showing the plot for one comedian
plt.plot(polarity_transcript[0])
plt.title(data['full_name'].index[0])
plt.show()
```


```python
# Show the plot for all comedians
plt.rcParams['figure.figsize'] = [16, 12]

for index, comedian in enumerate(data.index):    
    plt.subplot(3, 4, index+1)
    plt.plot(polarity_transcript[index])
    plt.plot(np.arange(0,10), np.zeros(10))
    plt.title(data['full_name'][index])
    plt.ylim(ymin=-.2, ymax=.3)
plt.show()
```

                                             Findings
Ali Wong stays generally positive throughout her routine. Similar comedians are Louis C.K. and Mike Birbiglia.

On the other hand, we have some pretty different patterns here like Bo Burnham who gets happier as time passes and Dave Chappelle who has some pretty down moments in his routine.

#                                      Topic Modelling

Another popular text analysis technique is called topic modeling. The ultimate goal of topic modeling is to find various topics that are present in your corpus. Each document in the corpus will be made up of at least one topic, if not multiple topics.

To use a topic modeling technique, you need to provide (1) a document-term matrix and (2) the number of topics you would like the algorithm to pick up using LDA(Latent Dirichlet Allocation)


```python
#Topic Modelling -Attempt 1(All text)
# Let's read in our document-term matrix
import pandas as pd
import pickle

data = pd.read_pickle('dtm_stop.pkl')
data
```


```python
# Import the necessary modules for LDA with gensim
# Terminal / Anaconda Navigator: conda install -c conda-forge gensim

from gensim import matutils,models
import scipy.sparse

#import logging
#ogging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
```


```python
# One of the required inputs is a term-document matrix
tdm = data.transpose()
tdm.head()
```


```python
# We're going to put the term-document matrix into a new gensim format, from df --> sparse matrix --> gensim corpus
sparse_counts = scipy.sparse.csr_matrix(tdm)
corpus = matutils.Sparse2Corpus(sparse_counts)
```


```python
# Gensim also requires dictionary of the all terms and their respective location in the term-document matrix
cv = pickle.load(open("cv_stop.pkl","rb"))
id2word = dict((v,k)for k,v in cv.vocabulary_.items())
```

Now that we have the corpus (term-document matrix) and id2word (dictionary of location: term), we need to specify two other parameters - the number of topics and the number of passes. Let's start the number of topics at 2, see if the results make sense, and increase the number from there.


```python
# Now that we have the corpus (term-document matrix) and id2word (dictionary of location: term),
# we need to specify two other parameters as well - the number of topics and the number of passes

lda = models.LdaModel(corpus = corpus, id2word=id2word,num_topics=2,passes=10)
lda.print_topics()
```


```python
# LDA for num_topics = 3
lda = models.LdaModel(corpus=corpus,id2word=id2word,num_topics=3,passes=10)
lda.print_topics()
```


```python
# LDA for num_topics = 4
lda = models.LdaModel(corpus=corpus,id2word=id2word,num_topics=4,passes=10)
lda.print_topics()
```


```python
#These topics aren't looking too great. We've tried modifying our parameters. Let's try modifying our terms list as well.
```

                                        Topic Modeling - Attempt #2 (Nouns Only)


```python
# Let's create a function to pull out nouns from a string of text
from nltk import word_tokenize, pos_tag

def nouns(text):
    '''Given a string of text, tokenize the text and pull out only the nouns.'''
    is_noun = lambda pos: pos[:2] == 'NN'
    tokenized = word_tokenize(text)
    all_nouns = [word for (word, pos) in pos_tag(tokenized) if is_noun(pos)] 
    return ' '.join(all_nouns)
```


```python
# Read in the cleaned data, before the CountVectorizer step
data_clean = pd.read_pickle('data_clean.pkl')
data_clean
```


```python
# Apply the nouns function to the transcripts to filter only on nouns
data_nouns = pd.DataFrame(data_clean.transcript.apply(nouns))
data_nouns
```


```python
# Create a new document-term matrix using only nouns
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer

# Re-add the additional stop words since we are recreating the document-term matrix
add_stop_words = ['like', 'im', 'know', 'just', 'dont', 'thats', 'right', 'people',
                  'youre', 'got', 'gonna', 'time', 'think', 'yeah', 'said']
stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

# Recreate a document-term matrix with only nouns
cvn = CountVectorizer(stop_words=stop_words)
data_cvn = cvn.fit_transform(data_nouns.transcript)
data_dtmn = pd.DataFrame(data_cvn.toarray(), columns=cvn.get_feature_names())
data_dtmn.index = data_nouns.index
data_dtmn
```


```python
# Create the gensim corpus
corpusn = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(data_dtmn.transpose()))

# Create the vocabulary dictionary
id2wordn = dict((v, k) for k, v in cvn.vocabulary_.items())
```


```python
# Let's start with 2 topics
ldan = models.LdaModel(corpus=corpusn, num_topics=2, id2word=id2wordn, passes=10)
ldan.print_topics()
```


```python
# Let's try topics = 3
ldan = models.LdaModel(corpus=corpusn, num_topics=3, id2word=id2wordn, passes=10)
ldan.print_topics()
```


```python
# Let's try 4 topics
ldan = models.LdaModel(corpus=corpusn, num_topics=4, id2word=id2wordn, passes=10)
ldan.print_topics()
```

                        Topic Modeling - Attempt #3 (Nouns and Adjectives)


```python
# Let's create a function to pull out nouns from a string of text
def nouns_adj(text):
    '''Given a string of text, tokenize the text and pull out only the nouns and adjectives.'''
    is_noun_adj = lambda pos: pos[:2] == 'NN' or pos[:2] == 'JJ'
    tokenized = word_tokenize(text)
    nouns_adj = [word for (word, pos) in pos_tag(tokenized) if is_noun_adj(pos)] 
    return ' '.join(nouns_adj)
```


```python
# Apply the nouns function to the transcripts to filter only on nouns
data_nouns_adj = pd.DataFrame(data_clean.transcript.apply(nouns_adj))
data_nouns_adj
```


```python
# Create a new document-term matrix using only nouns and adjectives, also remove common words with max_df
cvna = CountVectorizer(stop_words=stop_words, max_df=.8)
data_cvna = cvna.fit_transform(data_nouns_adj.transcript)
data_dtmna = pd.DataFrame(data_cvna.toarray(), columns=cvna.get_feature_names())
data_dtmna.index = data_nouns_adj.index
data_dtmna
```


```python
# Create the gensim corpus
corpusna = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(data_dtmna.transpose()))

# Create the vocabulary dictionary
id2wordna = dict((v, k) for k, v in cvna.vocabulary_.items())
```


```python
# Let's start with 2 topics
ldana = models.LdaModel(corpus=corpusna, num_topics=2, id2word=id2wordna, passes=10)
ldana.print_topics()
```


```python
# Let's try 3 topics
ldana = models.LdaModel(corpus=corpusna, num_topics=3, id2word=id2wordna, passes=10)
ldana.print_topics()
```


```python
# Let's try 4 topics
ldana = models.LdaModel(corpus=corpusna, num_topics=4, id2word=id2wordna, passes=10)
ldana.print_topics()
```

                                           FINDINGS
Identify Topics in Each Document
Out of the 9 topic models we looked at, the nouns and adjectives, 4 topic one made the most sense. So let's pull that down here and run it through some more iterations to get more fine-tuned topics.


```python
# Our final LDA model (for now)
ldana = models.LdaModel(corpus=corpusna, num_topics=4, id2word=id2wordna, passes=80)
ldana.print_topics()
```

                                         FINDINGS

These four topics look pretty decent. Let's settle on these for now.

Topic 0: mom, parents
Topic 1: husband, wife
Topic 2: guns
Topic 3: profanity


```python
# Let's take a look at which topics each transcript contains
corpus_transformed = ldana[corpusna]
list(zip([a for [(a,b)] in corpus_transformed], data_dtmna.index))
```

FINDINGS

For a first pass of LDA, these kind of make sense to me, so we'll call it a day for now.

Topic 0: mom, parents [Ali, Dave, Louis]
Topic 1: husband, wife [Bo, Hasan, Jim]
Topic 2: guns [Anthony,ricky]
Topic 3: profanity [Bill,Joe,John]

#                                                Text Generation

Markov chains can be used for very basic text generation. Think about every word in a corpus as a state. We can make a simple assumption that the next word is only dependent on the previous word - which is the basic assumption of a Markov chain.

Markov chains don't generate text as well as deep learning, but it's a good (and fun!) start.


```python
# Read in the corpus, including punctuation!
import pandas as pd

data = pd.read_pickle('corpus.pkl')
data
```


```python
# Extract only Ali Wong's text
ali_text = data.transcript.loc['ali']
ali_text[:200]
```

                                        Build a Markov Chain Function

We are going to build a simple Markov chain function that creates a dictionary:

The keys should be all of the words in the corpus
The values should be a list of the words that follow the keys


```python
from collections import defaultdict

def markov_chain(text):
    '''The input is a string of text and the output will be a dictionary with each word as
       a key and each value as the list of words that come after the key in the text.'''
    
    # Tokenize the text by word, though including punctuation
    words = text.split(' ')
    
    # Initialize a default dictionary to hold all of the words and next words
    m_dict = defaultdict(list)
    
    # Create a zipped list of all of the word pairs and put them in word: list of next words format
    for current_word, next_word in zip(words[0:-1], words[1:]):
        m_dict[current_word].append(next_word)

    # Convert the default dict back into a dictionary
    m_dict = dict(m_dict)
    return m_dict
```


```python
# Create the dictionary for Ali's routine, take a look at it
ali_dict = markov_chain(ali_text)
ali_dict
```

                                    Create a Text Generator

We're going to create a function that generates sentences. It will take two things as inputs:

The dictionary you just created
The number of words you want generated


```python
import random

def generate_sentence(chain, count=15):
    '''Input a dictionary in the format of key = current word, value = list of next words
       along with the number of words you would like to see in your generated sentence.'''

    # Capitalize the first word
    word1 = random.choice(list(chain.keys()))
    sentence = word1.capitalize()

    # Generate the second word from the value list. Set the new word as the first word. Repeat.
    for i in range(count-1):
        word2 = random.choice(chain[word1])
        word1 = word2
        sentence += ' ' + word2

    # End it with a period
    sentence += '.'
    return(sentence)
```


```python
generate_sentence(ali_dict)
```


```python
 
```
