from cnn_scraper import get_cnn_articles
from bbc_scraper import get_bbc_articles
from fox_scraper import get_fox_articles

def main():
    get_cnn_articles()
    get_bbc_articles()
    get_fox_articles()

if __name__ == "__main__":
    main()