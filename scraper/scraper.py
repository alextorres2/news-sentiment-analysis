from cnn_scraper import cnn_homepage_articles_analysis
from bbc_scraper import bbc_homepage_articles_analysis
from fox_scraper import fox_homepage_articles_analysis
from cbs_scraper import cbs_homepage_articles_analysis
from msnbc_scraper import msnbc_homepage_articles_analysis

def runner(show, save):
    """
    Main runner for analysis on 5 different News Main Webpages: CNN, BBC, FOXNEWS, CBS, and MSNBC

    Args:
        show (bool): Flag when True creates and displays the plots for each news org
        save (bool): Flag when True saves the data to CSV
    
    Returns:
        int: 
    """
    cnn_homepage_articles_analysis(show, save)
    bbc_homepage_articles_analysis(show, save)
    fox_homepage_articles_analysis(show, save)
    cbs_homepage_articles_analysis(show, save)
    msnbc_homepage_articles_analysis(show, save)

if __name__ == "__main__":
    show = True
    save=True
    runner(show, save)