from bs4 import BeautifulSoup
from bs4.element import Comment
import urllib.request
import nltk
nltk.download('averaged_perceptron_tagger')



def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)
    return u" ".join(t.strip() for t in visible_texts)

html = urllib.request.urlopen('https://en.wikipedia.org/wiki/Google').read()
#print(text_from_html(html))
#print(text)
#
text = str(text_from_html(html).encode("utf-8"))
text_file = open("input.txt", 'w')
text_file.write(text)
text_file.close()


# Question 3

wtokens = nltk.word_tokenize(text)
for t in wtokens:
    print(t)

print(nltk.pos_tag(wtokens))

#from nltk.stem import PorterStemmer

#pStemmer = PorterStemmer()
#print(pStemmer.stem(text))
import inflect

inflect = inflect.engine()

singular = []
plurals = []

for w in wtokens:
    if inflect.singular_noun(w) is False:
        singular.append(w)
    else:
        plurals.append(w)

#print('\n All plurals', plurals)
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

singles = [stemmer.stem(plural) for plural in plurals]

print('\nThe single Stemmers are :', singles)

from nltk.util import ngrams
trigram = ngrams(wtokens,3)
for t in trigram:
    print(t)

from nltk import wordpunct_tokenize, pos_tag, ne_chunk
print(ne_chunk(pos_tag(wordpunct_tokenize(text))))
for sent in nltk.sent_tokenize(text):
    for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.wordpunct_tokenize(sent))):
        if hasattr(chunk, 'label'):
            print(chunk.label(), ' '.join(c[0] for c in chunk))
#from nltk import wordpunct_tokenize,ne_chunk,pos_tag
#print("Named Entity Recognition")
#for w in wtokens:
 #   ner = ne_chunk(pos_tag(wordpunct_tokenize(w)))
  #  print(str(ner))