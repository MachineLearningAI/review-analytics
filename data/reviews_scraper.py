from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By

import json

def scrape_data_from_page(driver):
    review_els = driver.find_elements_by_class_name("BVRRReviewDisplayStyle3Main")
    user_els = driver.find_elements_by_class_name("BVRRReviewDisplayStyle3Summary")

    out = []
    for (el, user_el) in zip(review_els, user_els):
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
            # # print("no review text")
            review["text"] = ""
        # try:
        #     location_el = user_els.find_element_by_class_name("BVRRUserLocationContainer")
        #     print(location_el)
        #     review["location"] = location_el.find_elements_by_class_name("BVRRUserLocation")[0].text
        # except:
        #     review["location"] = ""
        # try:
        #     date_el = user_els.find_element_by_class_name("BVRRReviewDateContainer")
        #     review["date"] = date_el.find_elements_by_class_name("BVRRReviewDate")[0].text
        # except:
        #     review["date"] = "DATE"
        # try:
        #     contextdata_el = user_el.find_element_by_class_name("BVRRContextDataContainer")
        #     review["married"] = contextdata_el.find_elements_by_class_name("BVRRContextDataValueMarried")[0].text
        #     # print(review["married"])
        #     review["home"] = contextdata_el.find_elements_by_class_name("BVRRContextDataValueHome")[0].text
        #     # print(review["home"])
        #     review["kids"] = contextdata_el.find_elements_by_class_name("BVRRContextDataValueKids")[0].text
        #     # print(review["kids"])
        #     review["business"] = contextdata_el.find_elements_by_class_name("BVRRContextDataValueBusiness")[0].text
        #     # print(review["business"])
        #     review["school"] = contextdata_el.find_elements_by_class_name("BVRRContextDataValueSchool")[0].text
        #     # print(review["school"])
        #     review["language"] = contextdata_el.find_elements_by_class_name("BVRRContextDataValueLanguage")[0].text
        #     # print(review["language"])
        # except:
        #     review["married"] = ""
        #     review["home"] = ""
        #     review["kids"] = ""
        #     review["business"] = ""
        #     review["school"] = ""
        #     review["language"] = ""
        out.append(review)
    return out

def scrape_data():
    driver = webdriver.Chrome("../../selenium/chromedriver")
    mouse = webdriver.ActionChains(driver)
    driver.get("https://turbotax.intuit.com/reviews/?product=turbotax-online-premier")
    data_exists = True
    out = []
    i = 0
    # number of pages
    while data_exists and i < 480:
        try:
            data = scrape_data_from_page(driver)
            out.extend(data)
            button = driver.find_elements_by_name("BV_TrackingTag_Review_Display_NextPage")[0]
            # button = driver.find_elements_by_class_name("BVRRNextPage")[0]
            # # mouse.move_to_element(button).click().perform()
            # # driver.implicitly_wait(5)
            # driver.execute_script("arguments[0].click();", button)
            button.click()
            while out[-1]["text"] == data[-1]["text"] and out[-1]["title"] == data[-1]["title"] and out[-2]["text"] == data[-2]["text"] and out[-2]["title"] == data[-2]["title"] and out[-1]["pros"] == data[-1]["pros"] and out[-2]["rating"] == data[-2]["rating"]:
                data = scrape_data_from_page(driver)
                driver.implicitly_wait(1)
            # except:
            #     # print("hi")
            #     data_exists = False
            i += 1
            print("---Finished Review " + str(i) + "---")
        except:
            # print("process failed")
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
    with open("reviews_online_premier.json", "w") as f:
        json.dump(reviews, f)

if __name__ == "__main__":
    # write_data_to_file()
    test_scrape_data()
