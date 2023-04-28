"""
Jaymin West
Spring, 2023

This file represents the webscraper for the Snowpack Analysis project.
"""
import xmltodict, time, re
from datetime import datetime
import pandas as pd
import bs4 as bs
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import psycopg2, csv

class Webscraper():
    def __init__(self, location, season="Current Season", browser=webdriver.Chrome(), url='https://nwac.us/avalanche-forecast/#/archive'):
        self.location = location
        self.season = season
        self.browser = browser
        self.url = url
        self.browser.get(self.url)

    def open_archive_page(self):
        # Selecting only avalanche reports: 
        select_element = Select(self.browser.find_element(By.XPATH,'/html/body/div[1]/div[2]/div[2]/div/div/div[1]/div[1]/div/div[1]/div[2]/div[1]/select'))
        select_element.select_by_visible_text('Avalanche Forecast')

        # Finding location selection by XPATH:
        select_element = Select(self.browser.find_element(By.XPATH,'//*[@id="afp-forecast-widget"]/div/div/div[1]/div[1]/div/div[1]/div[2]/div[2]/select'))
        select_element.select_by_visible_text(self.location)

        # Finding season selection by XPATH:
        select_element = Select(self.browser.find_element(By.XPATH,'//*[@id="afp-forecast-widget"]/div/div/div[1]/div[1]/div/div[1]/div[1]/div[1]/select'))
        select_element.select_by_visible_text(self.season)
    
    def scrape_report_page(self):
        """
        Scrapes the individual report pages for all of the text within the report.
        """
        # loaded = False
        # while loaded is False:
        #     try:
        #         above_treeline_risk = self.browser.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[2]/div/div/div[2]/div/div[4]/div[1]/div/div[1]/div[1]/div[1]/div/div[1]/span[2]")
        #         loaded = True
        #     except:
        #         time.sleep(1)

        try:
            elem = WebDriverWait(self.browser, 10).until(
                EC.presence_of_element_located((By.XPATH, "/html/body/div[1]/div[2]/div[2]/div/div/div[2]/div/div[4]/div[1]/div/div[1]/div[1]/div[1]/div/div[1]/span[2]")) #This is a dummy element
            )
        except:
            return [None, None, None, None, None, None]
            
        # Scraping the risk data:
        above_treeline_risk = self.browser.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[2]/div/div/div[2]/div/div[4]/div[1]/div/div[1]/div[1]/div[1]/div/div[1]/span[2]")
        above_treeline_risk = above_treeline_risk.text

        near_treeline_risk = self.browser.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[2]/div/div/div[2]/div/div[4]/div[1]/div/div[1]/div[1]/div[1]/div/div[2]/span[2]")
        near_treeline_risk = near_treeline_risk.text

        below_treeline_risk = self.browser.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[2]/div/div/div[2]/div/div[4]/div[1]/div/div[1]/div[1]/div[1]/div/div[3]/span[2]")
        below_treeline_risk = below_treeline_risk.text

        # Scraping the report text:
        bottom_line = self.browser.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[2]/div/div/div[2]/div/div[3]")
        bottom_line_text = bottom_line.text

        # Gettting the avalanche problems text:
        problem_type_headers = self.browser.find_elements(By.CLASS_NAME, 'nac-problem')
        problem_type_text = ""
        for header in problem_type_headers:
            av_problem = header.find_element(By.CLASS_NAME, "nac-tinymce")
            narrow_av_problem = av_problem.find_element(By.CLASS_NAME, "nac-html-p")
            problem_type_text += narrow_av_problem.text
        # clean_problem_type_text = re.sub(r'PROBLEM TYPE.*?SIZE', '', problem_type_text, flags=re.DOTALL)

        forecast_discussion = self.browser.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[2]/div/div/div[2]/div/div[4]/div[1]/div/div[3]")
        forecast_discussion_text = forecast_discussion.text

        data = [above_treeline_risk, near_treeline_risk, below_treeline_risk, bottom_line_text, problem_type_text, forecast_discussion_text]
        for i in range(len(data)):
            data[i] = re.sub(r'PROBLEM TYPE.*?SIZE', '', data[i], flags=re.DOTALL)
            data[i] = re.sub(r'THE BOTTOM LINE\n', '', data[i])
            data[i] = re.sub(r"\d+\s-\s.*", str(data[i].split(" - ")[0]), data[i])
            data[i] = data[i].replace("\n", "")
    
        return data


    def scrape_daily_reports(self):
        """
        Scrapes the daily reports from the avalanche.org archive and returns
        the text within each report.
        """  
        reports_by_date = []      
        while True:
            try:
                elem = WebDriverWait(self.browser, 10).until(
                    EC.presence_of_element_located((By.XPATH, "/html/body/div[1]/div[2]/div[2]/div/div/div[1]"))
                )
            finally:
                pass
            # Getting the table:
            table = self.browser.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[2]/div/div/div[1]")
            rows = table.find_elements(By.TAG_NAME, "tr")
            
            for i in range(1, len(rows)):
                # Getting the date, area, and overall risk:
                overall_risk = rows[i].find_element(By.CSS_SELECTOR, "td:nth-child(6) > span:nth-child(1) > span:nth-child(1)").text
                zone = rows[i].find_element(By.CSS_SELECTOR, "td:nth-child(4)").text
                # Converting date to YYYY-MM-DD format:
                link = rows[i].find_element(By.TAG_NAME, "a")
                date = link.text
                org_date = datetime.strptime(date, '%b %d, %Y')
                new_date = datetime.strftime(org_date, '%Y-%m-%d')
                new_date = datetime.strptime(new_date, '%Y-%m-%d')
                
                # Getting the report data:
                link.click()

                report_data = self.scrape_report_page()
                report_data.insert(0, new_date)
                report_data.insert(1, zone)
                report_data.insert(2, overall_risk)

                reports_by_date.append(report_data)

                self.browser.back()
                self.to_csv(pd.DataFrame(reports_by_date))
            
            # Finding the "next page" button:
            button = self.browser.find_element(By.CLASS_NAME, "VuePagination__pagination li.VuePagination__pagination-item.nac-html-li.nac-page-item.VuePagination__pagination-item-next-page")  # add 1 to exclude "previous" button
            # Checking to see if the "next page" button is disabled:
            if button.find_element(By.TAG_NAME, "a").get_attribute("disabled"):
                print("Scraping complete.")
                break
            else:
                button.click()
                time.sleep(3)

        return pd.DataFrame(reports_by_date)
        

    def to_csv(self, data):
        data.to_csv('output_data/%s_%s_reports_data.csv'%(self.location.replace(" ", "_"), self.season.replace(" ", "_")), index=False)

    def to_postgres(self, data):
        pass

if __name__ == '__main__':
    # ws = Webscraper("All Zones")
    ws = Webscraper("Snoqualmie Pass")
    ws.open_archive_page()
    reports_data = ws.scrape_daily_reports()
    ws.to_csv(reports_data)
    print("Uncomment Code to Run")


