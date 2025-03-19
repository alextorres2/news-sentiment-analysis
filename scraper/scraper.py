from cnn_scraper import cnn_homepage_articles_analysis
from bbc_scraper import bbc_homepage_articles_analysis
from fox_scraper import fox_homepage_articles_analysis
from cbs_scraper import cbs_homepage_articles_analysis
from msnbc_scraper import msnbc_homepage_articles_analysis

def runner(show, save):
    cnn_homepage_articles_analysis(show, save)
    bbc_homepage_articles_analysis(show, save)
    fox_homepage_articles_analysis(show, save)
    cbs_homepage_articles_analysis(show, save)
    msnbc_homepage_articles_analysis(show, save)

if __name__ == "__main__":
    show = False
    save=False
    runner(show, save)