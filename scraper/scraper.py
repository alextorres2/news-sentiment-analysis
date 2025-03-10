from cnn_scraper import cnn_homepage_articles_analysis
from bbc_scraper import bbc_homepage_articles_analysis
from fox_scraper import fox_homepage_articles_analysis

def main(show):
    cnn_homepage_articles_analysis(show)
    bbc_homepage_articles_analysis(show)
    fox_homepage_articles_analysis(show)

if __name__ == "__main__":
    show = False
    main(show)