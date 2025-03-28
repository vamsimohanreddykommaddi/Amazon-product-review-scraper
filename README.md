# Amazon Product(Laptop) Reviews Scraper & Sentiment Analysis with WordClouds

## 📌 Overview
This project extracts laptop product listings and customer reviews from Amazon using **Selenium** and **BeautifulSoup**, then performs **sentiment analysis** and visualizes key insights using **WordClouds**.

## 🚀 Features
- **Web Scraping**: Extracts laptop product data (titles, images, ratings, and prices) from multiple pages on Amazon.
- **Customer Review Analysis**: Fetches customer reviews for a specific laptop model.
- **Sentiment Analysis**:
  - WordClouds for **all reviews**, **positive words**, and **negative words**.
  - Bigram frequency analysis to identify commonly used word pairs.
- **TF-IDF Processing**: Uses `TfidfVectorizer` to analyze review words.

## 🛠️ Tech Stack
- **Python**
- **Selenium** (for web scraping)
- **BeautifulSoup** (for HTML parsing)
- **Pandas** (for data handling)
- **Matplotlib** (for visualization)
- **WordCloud** (for sentiment visualization)
- **NLTK & Scikit-learn** (for text processing)

## 📊 Data Collection
1. **Product Information**: Extracted from Amazon search results.
2. **Customer Reviews**: Collected from product review pages.
3. **Stopwords Removal**: Custom stopword list applied.
4. **Sentiment Analysis**:
   - Positive and negative word classification.
   - Bigrams extraction for phrase analysis.

## 🔍 How to Run
1. **Install Dependencies:**
   ```bash
   pip install selenium beautifulsoup4 pandas matplotlib wordcloud tqdm nltk scikit-learn
2. **Run the Script:**
       python web-scraping-amazon.py
3. **View Results:**
       Extracted data is saved in products_laptop_amazon.csv.
       WordClouds are displayed for sentiment analysis.
## ⚠️ Disclaimer
    This project is for educational purposes only.
    Scraping Amazon may violate their Terms of Service—use responsibly.
