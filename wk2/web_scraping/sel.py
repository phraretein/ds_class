from selenium import webdriver


chrome = webdriver.Chrome('./chromedriver')

chrome.get('https://rod.kaidee.com/product-351076017')

chrome.close()