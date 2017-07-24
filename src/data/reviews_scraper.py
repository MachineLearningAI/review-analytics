from selenium import webdriver
from selenium.webdriver.common.keys import Keys

class Review(object):
    pass

def scrape_data_from_page(driver):
    review_els = driver.find_elements_by_class_name("BVRRReviewDisplayStyle3Main")

    out = []
    for el in review_els:
        review = Review()
        try:
            ratings_el = el.find_element_by_class_name("BVRRReviewRatingsContainer")
            review.rating = ratings_el.find_elements_by_class_name("BVRRRatingRangeNumber")[0].text
        except:
            review.rating = None
        try:
            title_el = el.find_element_by_class_name("BVRRReviewTitleContainer")
            review.title = title_el.find_elements_by_class_name("BVRRReviewTitle")[0].text
        except:
            review.title = ""
        try:
            proscons_el = el.find_element_by_class_name("BVRRReviewProsConsContainer")
            review.pros = [x.text for x in proscons_el.find_elements_by_class_name("BVRRReviewPros")]
        except:
            review.pros = []
        try:
            proscons_el = el.find_element_by_class_name("BVRRReviewProsConsContainer")
            review.cons = [x.text for x in proscons_el.find_elements_by_class_name("BVRRReviewCons")]
        except:
            review.cons = []
        try:
            text_el = el.find_element_by_class_name("BVRRReviewDisplayStyle3Content")
            review.text = text_el.find_elements_by_class_name("BVRRReviewText")[0].text
        except:
            review.text = []
        out.append(review)
    return out

def scrape_data():
    driver = webdriver.Chrome("../../selenium/chromedriver")
    driver.get("https://turbotax.intuit.com/reviews/?product=turbotax-online-federal-free-edition")
    data_exists = True
    out = []
    i = 0
    while data_exists and i < 2:
        try:
            out.extend(scrape_data_from_page(driver))
            button = driver.find_elements_by_class_name("BVRRNextPage")[0]
            driver.implicitly_wait(2)
            button.click()
            driver.implicitly_wait(2)
        except:
            data_exists = False
        i += 1
    driver.close()
    return out


def test_scrape_data():
    reviews = scrape_data()
    for review in reviews:
        print(review.rating, review.title, review.pros, review.cons, review.text)
        print("-----")

test_scrape_data()
