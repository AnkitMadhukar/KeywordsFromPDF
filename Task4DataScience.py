# importing required modules
import PyPDF2
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
import string
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer



#setting the stopwords
stop_words = set(stopwords.words('english'))
 
# creating a pdf file object
pdfFileObj = open("Name of your File / path", 'rb')
 
# creating a pdf reader object
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
 
#saving the no of pages
noOfPages = pdfReader.numPages

aList=[]
 
# creating a page object and adding content of 
# all the pages 
for x in range(0,noOfPages):
    pageObj = pdfReader.getPage(x)
    a = pageObj.extractText()
    aList.append( a );
 

#making a single string text
complete_text=" ".join(aList)

#Tokenizing 
word_tokens = word_tokenize(complete_text)
word_tokens=[word.lower() for word in word_tokens if word.isalpha()]

processed_sentence = [w for w in word_tokens if not w in stop_words]
processed_sentence = []
 
for w in word_tokens:
    if w not in stop_words:
       processed_sentence.append(w)
 
Counter = Counter(processed_sentence)


print("The most common ten words in the dataset are \n") 
print( Counter.most_common(10))

print("Keywords using TF-IDF Algorithm")

tfidf_vectorizer = TfidfVectorizer(analyzer='word',max_features=10,max_df =5,min_df=2)
tfidf = tfidf_vectorizer.fit_transform(aList)

print (tfidf_vectorizer.vocabulary_)

# closing the pdf file object
pdfFileObj.close()