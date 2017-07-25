from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By

import json

def scrape_data_from_page(driver):
    review_els = driver.find_elements_by_class_name("BVRRReviewDisplayStyle3Main")

    out = []
    for el in review_els:
        review = {}
        try:
            ratings_el = el.find_element_by_class_name("BVRRReviewRatingsContainer")
            review["rating"] = ratings_el.find_elements_by_class_name("BVRRRatingNumber")[0].get_attribute("textContent")
        except:
            review["rating"] = None
        try:
            title_el = el.find_element_by_class_name("BVRRReviewTitleContainer")
            review["title"] = title_el.find_elements_by_class_name("BVRRReviewTitle")[0].text
        except:
            review["title"] = ""
        try:
            proscons_el = el.find_element_by_class_name("BVRRReviewProsConsContainer")
            review["pros"] = [x.text for x in proscons_el.find_elements_by_class_name("BVRRReviewPros")]
        except:
            review["pros"] = []
        try:
            proscons_el = el.find_element_by_class_name("BVRRReviewProsConsContainer")
            review["cons"] = [x.text for x in proscons_el.find_elements_by_class_name("BVRRReviewCons")]
        except:
            review["cons"] = []
        try:
            text_el = el.find_element_by_class_name("BVRRReviewDisplayStyle3Content")
            review["text"] = text_el.find_elements_by_class_name("BVRRReviewText")[0].text
        except:
            # print("no review text")
            review["text"] = ""
        out.append(review)
    return out

def scrape_data():
    driver = webdriver.Chrome("../../selenium/chromedriver")
    mouse = webdriver.ActionChains(driver)
    driver.get("https://turbotax.intuit.com/reviews/?product=turbotax-online-federal-free-edition")
    data_exists = True
    out = []
    i = 0
    while data_exists and i < 1500:
        try:
            data = scrape_data_from_page(driver)
            out.extend(data)
            button = driver.find_elements_by_name("BV_TrackingTag_Review_Display_NextPage")[0]
            # button = driver.find_elements_by_class_name("BVRRNextPage")[0]
            # # mouse.move_to_element(button).click().perform()
            # # driver.implicitly_wait(5)
            # driver.execute_script("arguments[0].click();", button)
            button.click()
            while out[-1]["text"] == data[-1]["text"]:
                data = scrape_data_from_page(driver)
            # except:
            #     print("hi")
            #     data_exists = False
            i += 1
            print("---Finished Review " + str(i) + "---")
        except:
            print("process failed")
            data_exists = False
    driver.close()
    return out


def test_scrape_data():
    reviews = scrape_data()
    for review in reviews:
        print(review["rating"], review["title"], review["pros"], review["cons"], review["text"])
        print("-----")


def write_data_to_file():
    reviews = scrape_data()
    with open("reviews.json", "w") as f:
        json.dump(reviews, f)

if __name__ == "__main__":
    write_data_to_file()
    # test_scrape_data()
