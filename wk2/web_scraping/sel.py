from selenium import webdriver

# chrome = webdriver.Chrome('./chromedriver')
# chrome.get('https://www.bloomberg.com/asia')

# selector = 'a.single-story-module__headline-link'


# # Find element
# ## find_element(s)_by_css_selector
# element = chrome.find_element_by_css_selector(selector)

# # Select text from element
# print(element.text)

# # select attribute
# img_selector = 'img.single-story-module__image-img'
# img_element = chrome.find_element_by_css_selector(img_selector)
# # get attribute value
# img_url = element.get_attribute('src')

# print(img_element)
# print(img_url)

# Scrolling
chrome = webdriver.Chrome('./chromedriver')
chrome.get('https://twitter.com/realdonaldtrump?lang=th')
while True:
    scrolling_script = "window.scrollTo(0, document.body.scrollHeight);"
    chrome.execute_script(scrolling_script)
chrome.close()