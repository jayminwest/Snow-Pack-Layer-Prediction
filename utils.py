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
from sqlalchemy import create_engine

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
    
    try: 
        words = word_tokenize(text.lower())
        words = [w for w in words if not w in stop_words]
        words = [lemmatizer.lemmatize(w) for w in words]
        words = simple_preprocess(str(words), deacc=True)

        return ' '.join(words)
    except:
        return text

def prepare_text_column(column):
    """
    Prepares a text column for LDA analysis.
    """
    column = [str(item) for item in column]
    processed = [preprocess_text_column(doc) for doc in column]

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
    all_data = pd.read_csv(fname)
    # Adding Column Names:
    all_data.columns = ['date', 'zone', 'overall_risk', 'above_treeline_risk', 'near_treeline_risk', 'below_treeline_risk', 'bottom_line_text', 'problem_type_text', 'forecast_discussion_text']
    # Adding a column for the combined text of all 3 text columns:
    all_data['combined_text'] = all_data['bottom_line_text'] + all_data['problem_type_text'] + all_data['forecast_discussion_text']
    # Converting date column to datetime:
    all_data['date'] = pd.to_datetime(all_data['date'])
    
    # Processing all text columns:
    text_coloumns = ['bottom_line_text', 'problem_type_text', 'forecast_discussion_text', 'combined_text']
    
    for column in text_coloumns:
        all_data[column] = all_data[column].apply(preprocess_text_column)

    all_data['zone'] = all_data['zone'].str.lower()
    all_data['overall_risk'] = all_data['overall_risk'].str.lower()

    # Mapping risk ratings to numbers:
    rating_mapping = {
        "extreme": 5.0,
        "high": 4.0,
        "considerable": 3.0,
        "moderate": 2.0,
        "low": 1.0,
        "no rating": 0.0
    }

    all_data['overall_risk'] = all_data['overall_risk'].map(rating_mapping)

    return all_data

def get_weather_from_location(location, start, end=None):
    """
    Gathers weather data from the Meteostat API for a given location and time period.
    """
    stations = Stations()
    station = stations.nearby(location[0], location[1], 120000)
    station = station.fetch()

    # If no end date is given, use start date to just get single date data
    if end is None:
        end = start

    weather_data = Daily(station, start=start, end=end)
    weather_data = weather_data.normalize()
    weather_data = weather_data.aggregate('1D', spatial=True) # Aggregating data over time and spatialy (averaging all stations' data)
    weather_data = weather_data.fetch()
    
    # Removing empty columns:
    weather_data = weather_data[['tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wdir', 'pres', 'tsun']]
    weather_data = weather_data.reset_index()
    
    return weather_data

def add_weather_to_reports(reports_df):
    """
    Creates a new dataframe with weather data added to the reports dataframe.
    """
    centers_dict = {
        'olympics': (47.7795, -123.39750000000001),
        'west slopes north': (48.735552, -121.560974),
        'east slopes north': (48.410495499999996, -120.53237949999999),
        'west slopes central': (48.142828, -121.613846),
        'east slopes central': (47.92624, -121.04461699999999),
        'stevens pass': (47.721148, -121.1483),
        'snoqualmie pass': (47.3921515, -121.4380645),
        'west slopes south': (46.9136095, -121.898804),
        'east slopes south': (46.8361025, -121.324768),
        'mt hood': (45.3659035, -121.7367555)
    }

    col_names = ['tavg', 'tmin', 'tmax', 'prcp', 'wdir', 'pres', 'tsun'] # temporary fix
    for item in col_names: reports_df[item ]= None

    for index, row in reports_df.iterrows():
        weather_data = get_weather_from_location(centers_dict[row['zone']], row['date'])
        for item in col_names:
            reports_df.loc[index, item] = weather_data.loc[0, item]

    return reports_df

def get_weather_from_location(location, start, end=None):
    """
    Gathers weather data from the Meteostat API for a given location and time period.
    """
    stations = Stations()
    station = stations.nearby(location[0], location[1], 120000)
    station = station.fetch()

    # If no end date is given, use start date to just get single date data
    if end is None:
        end = start

    weather_data = Daily(station, start=start, end=end)
    weather_data = weather_data.normalize()
    weather_data = weather_data.aggregate('1D', spatial=True) # Aggregating data over time and spatialy (averaging all stations' data)
    weather_data = weather_data.fetch()
    
    # Removing empty columns:
    weather_data = weather_data[['tavg', 'tmin', 'tmax', 'prcp', 'wdir', 'pres', 'tsun']]
    weather_data = weather_data.reset_index()
    
    return weather_data

def dataframe_to_postgres(df, table_name=None):
    """
    Converts the csv files to a postgres database.
    """
    # Create the SQLAlchemy engine
    engine = create_engine(f'postgresql://{keys.POSTGRESQL_ADDON_USER}:{keys.POSTGRESQL_ADDON_PASSWORD}@{keys.POSTGRESQL_ADDON_HOST}:{keys.POSTGRESQL_ADDON_PORT}/{keys.POSTGRESQL_DB}')

    # Insert DataFrame into PostgreSQL table
    df.to_sql(table_name, engine, if_exists='replace', index=False)

    # Close the database connection
    engine.dispose()
    
    # return None
    # conn = psycopg2.connect("dbname=%s user=%s host=%s password=%s"%(keys.POSTGRESQL_ADDON_DB, keys.POSTGRESQL_ADDON_USER, keys.POSTGRESQL_ADDON_HOST, keys.POSTGRESQL_ADDON_PASSWORD))
    # cursor = conn.cursor()

    # sql0 = """DROP TABLE IF EXISTS current_season;"""

    # cursor.execute(sql0)

    # sql = """CREATE TABLE current_season (
    #     date varchar(20),
    #     zone SHORTTEXT,
    #     overall_risk float,
    #     above_treeline_risk float,
    #     near_treeline_risk float,
    #     below_treeline_risk float,
    #     bottom_line_text TEXT,
    #     problem_type_text TEXT,
    #     forecast_discussion_text TEXT,
    #     combined_text TEXT,
    #     tavg float,
    #     tmin float,
    #     tmax float,
    #     prcp float,
    #     wdir float,
    #     pres float,
    #     tsun float
    #     );
    #     """
    
    # cursor.execute(sql)

    # with open(file, 'r') as f:
    #     next(f)
    #     for line in f:
    #         list = line.split(',')
    #         print(list[1:])
    #         # line = str(list[1:])
    #         line = ','.join(list[1:])
    #         sql2 = """INSERT INTO weather_data VALUES (%s);"""%(line)
    #         # sql2 = """INSERT INTO weather_data VALUES ;"""%(list)
    #         cursor.execute(sql2)
    
    # conn.commit()
    # conn.close()
