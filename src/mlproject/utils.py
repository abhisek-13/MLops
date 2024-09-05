import os
import sys
import mysql.connector
import pandas as pd
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from dataclasses import dataclass
from dotenv import load_dotenv
import pymysql

#load_dotenv()

#host = os.getenv('host')
#user = os.getenv('user')
#password = os.getenv('password')
#db = os.getenv('db')

def read_sql_data():
  logging.info("Reading SQL database started.")
  try:
    mydb = pymysql.connect(
      host='localhost',
      user='root',
      password='trader life 9861',
      db='college_data')
    logging.info("Reading of data completed.",mydb)
    
    df = pd.read_sql_query('Select * from students',mydb)
    print(df.head())
    
    return df
    
  except Exception as ex:
    raise CustomException(ex,sys)
  