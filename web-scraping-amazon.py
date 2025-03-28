
import pandas as pd

from selenium import webdriver

from bs4 import BeautifulSoup as bs

import time

from tqdm import tqdm

# opening chrome driver

driver = webdriver.Chrome()

# providing url to driver

driver.get('https://amazon.in')

driver.get('https://www.amazon.in/s?k=laptops&crid=3E6KN3JQB44Y3&sprefix=laptops%2Caps%2C471&ref=nb_sb_noss_1')

#getting html data from driver to extract data using beautifulsoup

html_data = bs(driver.page_source, 'html.parser')

html_data

# getting number of pages

no_of_pages = html_data.find('span',{'class' : 's-pagination-item s-pagination-disabled'}).text

no_of_pages = int(no_of_pages)

no_of_pages

'''we have to repeat our loop for these 20 pages to extract laptops data from amazon'''

products = html_data.find_all('div',{'data-component-type' : 's-search-result'})

#creating lists for product data

titles = []
images = []
ratings = []
prices = []

for i in tqdm(range(1,no_of_pages+1)):
    
    url = 'https://www.amazon.in/s?k=laptops&crid=3E6KN3JQB44Y3&sprefix=laptops%2Caps%2C471&ref=nb_sb_noss_1&page='+str(i)
    
    driver.get(url)
    
    time.sleep(3)
    
    html_data = bs(driver.page_source,'html.parser')
    
    products = html_data.find_all('div',{'data-component-type' : 's-search-result'})

    for product in products:
        # extracting product titles
        title = product.find('h2',{'class' : 'a-size-medium a-spacing-none a-color-base a-text-normal'}).text
        titles.append(title)
        
        #extracting image of product
        img = product.find('img')['src']
        images.append(img)
        
        #extracting rating of the product
        rating = product.find('span',{'class' : 'a-icon-alt'})
        if rating == None:
            rating = 'No Rating'
        else:
            rating = product.find('span',{'class' : 'a-icon-alt'}).text
        ratings.append(rating)
        
        #extracting price of the product
        
        price = product.find('span',{'class' : 'a-price-whole'})
        if price == None:
            price = 'Not Available'
        else :
            price = product.find('span',{'class' : 'a-price-whole'}).text
        prices.append(price)
        
# creating a dataframe with products data

product_df = pd.DataFrame({'title':titles, 'image':images, 'rating':ratings, 'price':prices})

product_df.head()

# storing the products data into csv file

product_df.to_csv('products_laptop_amazon.csv')


'''Extracting product reviews for sentiment analysis'''

driver.get('https://www.amazon.in/product-reviews/B07WFPMQPC/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews&')

html_data_reviews = bs(driver.page_source, 'html.parser')

reviews = html_data_reviews.find_all('span',{'data-hook' : 'review-body'})

no_of_review_pages = 10 # got from website

iqoo_reviews = []

for i in tqdm(range(1,no_of_review_pages+1)):
    
    url = 'https://www.amazon.in/product-reviews/B07WFPMQPC/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews&pageNumber='+str(i)
    
    driver.get(url)
    
    time.sleep(3)
    
    html_data = bs(driver.page_source, 'html.parser')
    
    reviews = html_data.find_all('span',{'data-hook' : 'review-body'})
    
    for review in reviews:
        
        #extracting review text
        iqoo_reviews.append(review.text)
        
# performing sentiment analysis --> wordcloud

import re

from wordcloud import WordCloud

import matplotlib.pyplot as plt

#writing reviews of iqoo 13 into a text file

with open('iqoo13_reviews.text','w',encoding='utf-8') as f:
    f.write(str(iqoo_reviews))
    
#joining all reviews into a single paragraph

review_string = ''.join(iqoo_reviews)

#removing unwanted symbols incase they exists

review_string = re.sub('[^A-Za-z" "]+'," ",review_string).lower()

#finding words that are in the reviews

review_words = review_string.split(" ")

review_words = review_words[1:] #ignoring space

#performing TFIDF

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(ngram_range = (1,1), use_idf = True)

dtm = vectorizer.fit_transform(review_words)

with open(r'D:\practice-360\Data-Science(hands-on)\3.f.Text Mining\stopwords-en.txt','r',encoding = 'utf-8') as sw:
    stop_words = sw.read()
    
stop_words = stop_words.split('\n')

# adding extra stop words based on reviews

stop_words.extend(['iqoo','mobile','phone','phones','android','device','product'])

# extracting words in reviews without stopwords

review_words_new = [w for w in review_words if w not in stop_words]


# Generating wordcloud --> wordcloud is performed on string inputs

review_string_new = ' '.join(review_words_new)

word_cloud_iqoo = WordCloud(background_color='white',
                            width=1800,
                            height=1400).generate(review_string_new)

# Display the word cloud
plt.figure(figsize=[10, 10])
plt.imshow(word_cloud_iqoo, interpolation="bilinear")
plt.axis("off")
plt.show()

# Extracting positive words

with open(r"D:\practice-360\Data-Science(hands-on)\3.f.Text Mining\positive-words.txt", 'r', encoding = 'utf-8') as f:
    positive_words = f.read().split('\n')
    
#extracting positive words from reviews

review_words_positive = [w for w in review_words_new if w in positive_words]

review_string_positive = ' '.join(review_words_positive)

# performing positive word cloud

wordcloud_positive = WordCloud(background_color='white',
                               width = 1800,
                               height=1400).generate(review_string_positive)

# Display the word cloud
plt.figure(figsize=[10, 10])
plt.imshow(wordcloud_positive, interpolation="bilinear")
plt.axis("off")
plt.show()

#Extracting negative words

with open(r"D:\practice-360\Data-Science(hands-on)\3.f.Text Mining\negative-words.txt",'r',encoding='utf-8') as f:
    
    negative_words = f.read().split('\n')
    
# extracting negative words from reviews

review_words_negative = [w for w in review_words_new if w in negative_words]

review_string_negative = ' '.join(review_words_negative)

# performing negative word cloud

wordcloud_negative = WordCloud(background_color='white',
                               width = 1800,
                               height=1400).generate(review_string_negative)

# Display the word cloud
plt.figure(figsize=[10, 10])
plt.imshow(wordcloud_negative, interpolation="bilinear")
plt.axis("off")
plt.show()

''' Now performing wordcloud with bigrams '''

import nltk

nltk.download('punkt')

from wordcloud import WordCloud, STOPWORDS

wnl = nltk.WordNetLemmatizer()

#removing single quotes from reviews to avoid problems during tokenization

review_string = review_string.replace("'"," ")

#Tokenization using nltk

tokens = nltk.word_tokenize(review_string)

text = nltk.Text(tokens)

# Removing extra characters as well as stopwords

review_content = [' '.join(re.split("[ .,;:!?''""@#$%^&*()<>{}~\n\t\\\-]",word)) for word in text]

stopwords_wc = set(STOPWORDS)

stop_words_new = stopwords_wc.union({'price','great','13','iqoo','phones','mobile'})

review_content = [word for word in review_content if word not in stop_words_new]

# take only non-empty entries i.e., len>1

review_content = [s for s in review_content if len(s)>1]

#getting lemmas for each word to reduce number of similar words

review_content = [wnl.lemmatize(s) for s in review_content]

# Extracting bi-grams from review content

bigrams_list = list(nltk.bigrams(review_content))

review_dictionary = [' '.join(tup) for tup in bigrams_list]

# using count vectorizer to analyze frequency of bigrams

from sklearn.feature_extraction.text import CountVectorizer

vectorizer_bigram = CountVectorizer(ngram_range=(2,2))

bag_of_words = vectorizer_bigram.fit_transform(review_dictionary)

vectorizer_bigram.vocabulary_

sum_words = bag_of_words.sum(axis=0)


word_freq = [(word,sum_words[0,idx]) for word,idx in vectorizer_bigram.vocabulary_.items()]

word_freq = sorted(word_freq, key = lambda x : x[1], reverse = True)

print(word_freq[:100])

# Generating wordcloud for bigrams

words_dict = dict(word_freq)

word_cloud_bigram = WordCloud(background_color='white',
                              width = 1800,
                              height = 1400,
                              stopwords = stop_words_new).generate_from_frequencies(words_dict)

# Display the word cloud
plt.figure(figsize=[10, 10])
plt.title('Most frequently occurring bigrams connected by same color and font size')
plt.imshow(word_cloud_bigram, interpolation="bilinear")
plt.axis("off")
plt.show()