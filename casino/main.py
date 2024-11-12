from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import pandas as pd
import tensorflow as tf 
from tensorflow import keras
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from time import sleep
import statsmodels as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import sklearn as sk
import os
from collections import Counter
from datetime import date
import datetime
import calendar
import time
import string
from string import digits
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input, Embedding, TimeDistributed, Flatten, Dropout
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def mape(actual,pred):
    actual,pred = np.array(actual),np.array(pred)
    return np.mean(np.abs((actual-pred)/actual)) * 100


counter = 0
df = pd.DataFrame(columns = ['amount players','round result', 'MO','bank','timee','wkday','coeff'])
driver = webdriver.Chrome()
driver.get("https://hellstore.me/ru/crash")
sleep(2)
while True:
    counter +=1

    try:
        element = WebDriverWait(driver,9000).until(
            EC.presence_of_element_located((By.CLASS_NAME, "game-timer-crashed"))
        )
    except TimeoutException as ex:
        print("Exceptinon has been thrown" + str(ex))
    html = driver.page_source


    #amount players pd ------------------------------------------------------------------------------------
    amount_of_players = float(driver.find_element(By.CLASS_NAME,"game-stat-value").text)


    #round result pd --------------------------------------------------------------------------------------
    betted_amount = list()
    all_betted = driver.find_elements(By.CLASS_NAME,"player-betted-amount")
    for bet in all_betted:
        betted_amount.append(float(bet.text[1:]))
    

    real_wins = list()
    all_wins = driver.find_elements(By.CLASS_NAME, 'player-winned-amount')
    for win in all_wins:
        if win.text != '':
            real_wins.append(float(win.text[1:]))


    round_result = round(sum(betted_amount) - sum(real_wins),2)


    #MO pd -------------------------------------------------------------------------------------------------
    f = open(r"F:\folders\casino\mo.txt","r").readlines()
    a = []
    for x in f:
        a.append(round(float(x[0:-1]),1))
    thre = 3
    for i in range(len(a)):
        if a[i]>thre:
            a[i] = thre
    d = {}
    for i in a:
        if i in d:
            d[i] +=1
        else:
            d[i] = 1
    sorted_tuple = sorted(d.items(), key=lambda x: x[0])
    d = dict(sorted_tuple)
    s = 0
    num = 0        
    for i in d.values():
        num +=i
    for i in d.keys():
        tmp = d.get(i)
        s+=(i*tmp/num)
    mo = round(s,3)
    #bank pd ----------------------------------------------------------------------------------------------
    #amount_betted


    #timee pd ----------------------------------------------------------------------------------------------
    timee = round(time.time(),2)


    #weekday pd ----------------------------------------------------------------------------------------------
    wkday = (datetime.datetime.today().weekday())+1


    #coeff pd ----------------------------------------------------------------------------------------------
    f = open("hist.txt","a")
    e1 = driver.find_element(By.CLASS_NAME, "game-timer-crashed")
    f.write(e1.text[0:-1])
    f.write("\n")
    f.close()
    coeff = driver.find_element(By.CLASS_NAME, "game-timer-crashed")
    coeff = float(coeff.text[0:-1])
    if coeff > 3.0:
        coeff = 3.0
    f = open(r'F:\folders\casino\histtt.txt','a')
    f.write(f"{amount_of_players};{round_result};{mo};{round(sum(betted_amount),2)};{timee};{wkday};{coeff}")
    f.write("\n")
    f.close()
    ff = open(r'F:\folders\casino\mo.txt','a')
    ff.write(f"{coeff}")
    ff.write("\n")
    ff.close()
    sleep(10)
    


    


