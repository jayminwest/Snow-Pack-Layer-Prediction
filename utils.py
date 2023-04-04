"""
Jaymin West
Spring, 2023

This file contains utility functions for the Snowpack Analysis project.
"""
import xmltodict
from datetime import datetime
import pandas as pd
import bs4 as bs
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By

def snowpilot_xml_to_dict(fname):
    """
    Parses the snowpilot xml file and returns a dictionary of the data.
    """
    with open(fname, 'r', encoding='utf-8') as file:
        sp_xml = file.read()

    sp_xml = xmltodict.parse(sp_xml)

    return sp_xml['Pit_Observation']

def scrape_avalanche_data(url, location):
    """
    Scrapes the avalanche data from the avalanche.org website and returns a
    dataframe with the data.
    """
    # Using Selenium to get the avalanche forecast data:
    date_risks = []

    browser = webdriver.Chrome()
    url = 'https://nwac.us/avalanche-forecast/#/archive'
    browser.get(url)
    # Finding by XPATH:
    select_element = Select(browser.find_element(By.XPATH,'//*[@id="afp-forecast-widget"]/div/div/div[1]/div[1]/div/div[1]/div[2]/div[2]/select'))
    # Selecting Snoqualmie Pass from dropdown menu:
    select_element.select_by_visible_text(location)

    response = browser.page_source

    soup = bs.BeautifulSoup(response, 'html.parser')

    prediction_table = soup.find_all('tr', {'class': 'VueTables__row'})
    for row in prediction_table:
        date = row.find_all('td')[0].text
        org_date = datetime.strptime(date, '%b %d, %Y')
        new_date = datetime.strftime(org_date, '%Y-%m-%d')
        new_date = datetime.strptime(new_date, '%Y-%m-%d')
        date_risks.append([new_date, row.find_all('td')[5].text])

    select_element = browser.find_element(By.XPATH,'//*[@id="afp-forecast-widget"]/div/div/div[1]/div[2]/div[2]/nav/ul/li[4]/a')
    select_element.click()

    response = browser.page_source

    soup = bs.BeautifulSoup(response, 'html.parser')

    prediction_table = soup.find_all('tr', {'class': 'VueTables__row'})
    for row in prediction_table:
        date = row.find_all('td')[0].text
        org_date = datetime.strptime(date, '%b %d, %Y')
        new_date = datetime.strftime(org_date, '%Y-%m-%d')
        new_date = datetime.strptime(new_date, '%Y-%m-%d')
        date_risks.append([new_date, row.find_all('td')[5].text])

    browser.quit()

    date_risks = pd.DataFrame(date_risks, columns=['time', 'risk'])
    return date_risks
