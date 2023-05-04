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
import psycopg2, csv
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import gensim
from gensim import corpora
from gensim.utils import simple_preprocess
import keys
from meteostat import Stations, Daily, Point, Hourly

def snowpilot_xml_to_dict(fname):
    """
    Parses the snowpilot xml file and returns a dictionary of the data.
    """
    with open(fname, 'r', encoding='utf-8') as file:
        sp_xml = file.read()

    sp_xml = xmltodict.parse(sp_xml)

    return sp_xml['Pit_Observation']

def combine_csv(f1, f2, f3):
    """
    Combines f1 and f2 files into a new csv file, f3
    """
    with open(f1, 'r') as f:
        reader = csv.reader(f)
        data1 = list(reader)

    with open(f2, 'r') as f:
        reader = csv.reader(f)
        data2 = list(reader)

    data = data1 + data2

    with open(f3, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(data)


def preprocess_text_column(text):
    stop_words = set(stopwords.words('english'))
    # adding days of the week to stop words
    stop_words.update(['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'])
    # adding months to stop words
    stop_words.update(['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
                       'september', 'october', 'november', 'december'])
    lemmatizer = WordNetLemmatizer()
    
    words = word_tokenize(text.lower())
    words = [w for w in words if not w in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    words = simple_preprocess(str(words), deacc=True)
    
    return words

def prepare_text_column(column):
    """
    Prepares a text column for LDA analysis.
    """
    column = [str(item) for item in column]
    processed = [preprocess(doc) for doc in column]

    # Create a dictionary of terms and their frequency
    dictionary = corpora.Dictionary(processed)

    # Create a document-term matrix
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in processed]

    return doc_term_matrix, dictionary


def clean_raw_webscraper_data(fname):
    """
    Takes in a file of raw avalanche reports data from the webscraper, and returns a dataframe of the cleaned
    """
    # Reading in the CSV file:
    all_data = pd.read_csv('output_data/incomplete_All_Zones_Current_Season_reports_data.csv')
    # Adding Column Names:
    all_data.columns = columns=['date', 'zone', 'overall_risk', 'above_treeline_risk', 'near_treeline_risk', 'below_treeline_risk', 'bottom_line_text', 'problem_type_text', 'forecast_discussion_text']
    # Adding a column for the combined text of all 3 text columns:
    all_data['combined_text'] = all_data['bottom_line_text'] + all_data['problem_type_text'] + all_data['forecast_discussion_text']

def collect_weather_data(station, start_date, end_date):
    """
    Uses Meteostat to collect weather data of a given station between two dates.
    """
    return None

    

def dataframe_to_postgres(df, table_name):
    """
    Converts the csv files to a postgres database.
    """
    return None
    conn = psycopg2.connect("dbname=%s user=%s host=%s password=%s"%(keys.POSTGRESQL_ADDON_DB, keys.POSTGRESQL_ADDON_USER, keys.POSTGRESQL_ADDON_HOST, keys.POSTGRESQL_ADDON_PASSWORD))
    cursor = conn.cursor()

    sql0 = """DROP TABLE IF EXISTS weather_data;"""

    cursor.execute(sql0)

    sql = """CREATE TABLE weather_data (
        time varchar(20),
        temp float,
        dwpt float,
        rhum float,
        prcp float,
        wdir float,
        wspd float,
        pres float,
        coco float,
        risk integer
        );
        """
    
    cursor.execute(sql)

    with open(file, 'r') as f:
        next(f)
        for line in f:
            list = line.split(',')
            print(list[1:])
            # line = str(list[1:])
            line = ','.join(list[1:])
            sql2 = """INSERT INTO weather_data VALUES (%s);"""%(line)
            # sql2 = """INSERT INTO weather_data VALUES ;"""%(list)
            cursor.execute(sql2)
    
    conn.commit()
    conn.close()
