###WIKIPEDIA TEXT PREPROCESSING AND VISUALIZATION ####
# This is a text processing and visualization project.
# Some Wikıpedia texts are used in this project.

#### WIKIPEDIA Text Preprocessing ####

from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from textblob import Word, TextBlob
from wordcloud import WordCloud


filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# ##### LOADINDING ########

def load():
    data = pd.read_csv("/Users/cemiloksuz/PycharmProjects/EuroTechMiullDataScience/week_13/wiki-221126-161428/wiki_data.csv")
    data = data.drop('Unnamed: 0', axis=1)
    return data

df = load()
df_copy = df.copy
df.head()

##################################################
# 1. Text Preprocessing
##################################################

# Clean_text
def clean_text(text):
    text = text.str.lower().\
        replace('[^\w\s]', '', regex=True).\
        replace('\n', '', regex=True).\
        replace('\d', '', regex=True).\
        replace("â", "", regex=True)
    return text

df["text"] = clean_text(df["text"])


# Stopwords
def stop_words(text, lang="english"):
    sw = stopwords.words(lang)
    text = text.apply(lambda x: " ".join(x for x in x.split() if x not in sw))
    return text

df["text"] = stop_words(df["text"])

# Rare_words
def temp(text):
    temp = pd.Series(" ".join(text).split()).value_counts()
    drops = temp[temp <= 1750]
    text = text.apply(lambda x: " ".join(x for x in x.split() if x not in drops))
    return text

df["text"] = temp(df["text"])

# Tokenization
def token(text):
    text = text.apply(lambda x: TextBlob(x).words)
    return text

df["text"] = token(df["text"])

# Lemmatization
def lemma(text):
    text = text.apply(lambda x: [Word(word).lemmatize() for word in x])
    text = text.apply(lambda x: " ".join(x))
    return text

df["text"] = lemma(df["text"])

##################################################
# 2. Text Visualization
##################################################

# Barplot
def tf_visual (text):
    tf = text.apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
    tf.columns = ["words", "tf"]
    tf = tf.sort_values("tf", ascending=False)
    tf[tf["tf"] > 2000].plot.bar(x="words", y="tf")
    plt.show()

tf_visual(df["text"])

# Wordcloud
def word_cloud(text):
    text = " ".join(i for i in text)
    wordcloud = WordCloud(max_font_size=50,
                          max_words=100,
                          background_color="green").generate(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

word_cloud(df["text"])

##################################################
# 3. Funktion
##################################################

df = pd.read_csv("/Users/cemiloksuz/PycharmProjects/EuroTechMiullDataScience/week_13/wiki-221126-161428/wiki_data.csv", index_col=0)
def wiki_preprocess(text, Barplot=False, Wordcloud=False):
    """
    Textler üzerinde ön işleme işlemleri yapar.

    :param text: DataFrame'deki textlerin olduğu değişken
    :param Barplot: Barplot görselleştirme
    :param Wordcloud: Wordcloud görselleştirme
    :return: text


    Example:
            wiki_preprocess(dataframe[col_name])

    """
    # Normalizing Case Folding
    text = text.str.lower()
    # Punctuations
    text = text.str.replace('[^\w\s]', '')
    text = text.str.replace("\n", '')
    # Numbers
    text = text.str.replace('\d', '')
    # Stopwords
    sw = stopwords.words('English')
    text = text.apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))
    # Rarewords / Custom Words
    sil = pd.Series(' '.join(text).split()).value_counts()[-1000:]
    text = text.apply(lambda x: " ".join(x for x in x.split() if x not in sil))


    if Barplot:
        # Terim Frekanslarının Hesaplanması
        tf = text.apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
        # Sütunların isimlendirilmesi
        tf.columns = ["words", "tf"]
        # 5000'den fazla geçen kelimelerin görselleştirilmesi
        tf[tf["tf"] > 2000].plot.bar(x="words", y="tf")
        plt.show(block=True)

    if Wordcloud:
        # Kelimeleri birleştirdik
        text = " ".join(i for i in text)
        # wordcloud görselleştirmenin özelliklerini belirliyoruz
        wordcloud = WordCloud(max_font_size=50,
                              max_words=100,
                              background_color="white").generate(text)
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show(block=True)

    return text

wiki_preprocess(df["text"])

wiki_preprocess(df["text"], True, True)



