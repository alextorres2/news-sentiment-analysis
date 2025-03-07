import requests
from bs4 import BeautifulSoup
import nltk
from textblob import TextBlob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import spacy
import datetime
from wordcloud import WordCloud

nlp = spacy.load('en_core_web_sm')

def get_fox_articles():
    url = "https://www.foxnews.com"
    response = requests.get(url)
    response.encoding = 'utf-8'
     
    html = response.text
    soup = BeautifulSoup(html, features="lxml")
    text = soup.get_text()

    clean_text = text.replace('\n',' ')
    clean_text = clean_text.replace('/', ' ')
    clean_text = ''.join([c for c in clean_text if c != "\'"])

    # Sentiment Analysis
    sentence=[]
    tokens = nlp(clean_text)
    for sent in tokens.sents:
        sentence.append((sent.text.strip()))

    textblob_sentiment = []
    pattern_sentiment = []
    for s in sentence:
        # TextBlob Sentiment
        text = TextBlob(s)
        a = text.sentiment.polarity
        b = text.sentiment.subjectivity
        textblob_sentiment.append([s,a,b])

    df_textblob = pd.DataFrame(textblob_sentiment, columns =['Sentence', 'Polarity', 'Subjectivity'])

    sns.displot(df_textblob["Polarity"], height= 5, aspect=1.8)
    plt.xlabel("Sentence Polarity (Textblob)")
    # plt.show()

    sns.displot(df_textblob["Subjectivity"], height= 5, aspect=1.8)
    plt.xlabel("Sentence Subjectivity (Textblob)")
    # plt.show()

    # Word Cloud
    tokenizer = nltk.tokenize.RegexpTokenizer('\w+')    # Creating the tokenizer
    tokens = tokenizer.tokenize(clean_text)
    words = []  # Make them all LowerCase
    for word in tokens:
        words.append(word.lower())
    try:
        stopwords = nltk.corpus.stopwords.words('english')
    except Exception:
        nltk.download('stopwords')
        stopwords = nltk.corpus.stopwords.words('english')
    stopwords.append('fox')

    words_new = []
    for word in words:  # Appending to words_new all words that are in words but not in stopwords
        if word not in stopwords:
            words_new.append(word)
    
    freq_dist = nltk.FreqDist(words_new)    # Word Frequency Distribution
    df_freq_dist = pd.DataFrame(freq_dist.items(), columns=['word', 'frequency'])
    plt.subplots(figsize=(16,10))
    freq_dist.plot(20)
    # plt.show()

    res=' '.join([i for i in words_new if not i.isdigit()])

    plt.subplots(figsize=(16,10))
    wordcloud = WordCloud(
        background_color='black',
        max_words=100,
        width=1400,
        height=1200
    ).generate(res)

    plt.imshow(wordcloud)
    plt.title('NEWS ARTICLE (100 words)')
    plt.axis('off')
    plt.show()

    # Write data to CSV
    now = str(datetime.datetime.now())
    now = now.replace(" ", "_")
    df_textblob.to_csv(f"data/processed/fox/TextBlob/textblob_{now}.csv")
    df_freq_dist.to_csv(f"data/processed/fox/WordCloud/freq_dist_{now}.csv")

    print()

    # TODO list:
    #   - Save data to csv file
    #       - date, average polarity, average subjectivity
    #   - Will be able to show plot over time of sentiment changes
    #   - Keep track of keywords
    #       - maybe also save this to csv file
    #       - can create a word cloude
    #       - could have scroller to show word cloud changes over time
    #       - tokenize words -- count each word occurrence

    # soup = BeautifulSoup(response.content, 'html.parser')

    # # Find the news articles on the page
    # articles = soup.find_all('a', class_='gs-c-promo-heading')
    # data = []

    # for article in articles:
    #     title = article.get_text()
    #     link = article['href']
    #     if link.startswith('/'):
    #         link = "https://www.bbc.com" + link  # Handle relative URLs
    #     data.append({"title": title, "link": link})

    # return data

# Test the scraper
articles = get_fox_articles()
# for article in articles:
#     print(article['title'], article['link'])
