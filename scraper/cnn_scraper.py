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

def get_articles_content(url: str) -> list:
    """Given a URL to CNN news homepage, get the text of all articles in the homepage
        returns list of lists [[article 1 paragraph 1, article 1 paragraph 2, ...], [article 2 paragraph 1, article 2 paragraph 2, ...], ...]
    """
    response = requests.get(url)
     
    soup = BeautifulSoup(response.content, 'html.parser')
    articles = soup.find_all(
        'a', 
        class_="container__link container__link--type-article container_lead-package__link container_lead-package__left container_lead-package__light"
    )

    sub_urls = set()
    for article in articles:
        sub_urls.add("https://www.cnn.com" + article['href'])

    contents = []
    for url in sub_urls:
        sub_response = requests.get(url)
        sub_soup = BeautifulSoup(sub_response.content, 'html.parser')
        paragraphs = sub_soup.find_all('p', class_="paragraph inline-placeholder vossi-paragraph")
        paragraphs = [p.get_text().replace('\n', '').strip() for p in paragraphs]
        contents.append(paragraphs)
    return contents

def cnn_homepage_articles_analysis(show: bool = False):
    """Perform and save sentiment analysis on articles on CNN homepage website.
        Creates plots for Articles sentiment (Polarity and Subjectivity) and a WordCloud of the words in the articles
        Saves data to data/processed/cnn/TextBlob/ and data/processed/cnn/WordCloud/
    """
    url = "https://www.cnn.com"
    contents = get_articles_content(url)

    # Sentiment Analysis
    textblob_sentiment = []
    for paragraphs in contents:
        for paragraph in paragraphs:
            # TextBlob Sentiment
            text = TextBlob(paragraph)
            a = text.sentiment.polarity
            b = text.sentiment.subjectivity
            textblob_sentiment.append([text,a,b])

    df_textblob = pd.DataFrame(textblob_sentiment, columns =['Sentence', 'Polarity', 'Subjectivity'])

    sns.displot(df_textblob["Polarity"], height= 5, aspect=1.8)
    plt.xlabel("Article Polarity (Textblob)")

    sns.displot(df_textblob["Subjectivity"], height= 5, aspect=1.8)
    plt.xlabel("Article Subjectivity (Textblob)")

    # Word Cloud
    words = []
    for text in contents:
        text = '\n'.join(text)
        tokenizer = nltk.tokenize.RegexpTokenizer('\w+')    # Creating the tokenizer
        tokens = tokenizer.tokenize(text)
        # Make them all LowerCase
        for word in tokens:
            words.append(word.lower())
    try:
        stopwords = nltk.corpus.stopwords.words('english')
    except Exception:
        nltk.download('stopwords')
        stopwords = nltk.corpus.stopwords.words('english')
    extra_wordstoignore = ['cnn', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'said']
    for word in extra_wordstoignore:
        stopwords.append(word)
    words_new = []
    for word in words:  # Appending to words_new all words that are in words but not in stopwords
        if word not in stopwords:
            words_new.append(word)

    freq_dist = nltk.FreqDist(words_new)    # Word Frequency Distribution
    df_freq_dist = pd.DataFrame(freq_dist.items(), columns=['word', 'frequency'])
    plt.subplots(figsize=(8,5))
    freq_dist.plot(20)

    res = ' '.join([i for i in words_new if not i.isdigit()])

    plt.subplots(figsize=(10,7))
    wordcloud = WordCloud(
        background_color='black',
        max_words=100,
        width=1400,
        height=1200
    ).generate(res)
    plt.imshow(wordcloud)
    plt.title('CNN Website WordCloud')
    plt.axis('off')
    if show:
        plt.show()

    # Write data to CSV
    now = str(datetime.datetime.now())
    now = now.replace(" ", "_")
    df_textblob.to_csv(f"data/processed/cnn/TextBlob/textblob_{now}.csv")
    df_freq_dist.to_csv(f"data/processed/cnn/WordCloud/freq_dist_{now}.csv")

if __name__ == "__main__":
    cnn_homepage_articles_analysis(show=True)

    # TODO list:
    #   - Save data to csv file
    #       - date, average polarity, average subjectivity
    #   - Will be able to show plot over time of sentiment changes
    #   - Keep track of keywords
    #       - maybe also save this to csv file
    #       - can create a word cloude
    #       - could have scroller to show word cloud changes over time
    #       - tokenize words -- count each word occurrence
    #   - Analysis of group of words, not just individual keywords
    #   - Look through html, find links, go to links, read those articles and do analysis on those

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