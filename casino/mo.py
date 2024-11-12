from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from time import sleep
import datetime
import time
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from random import randint


def mape(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

counter = 0
df = pd.read_csv(r"F:\folders\casino\histtt.txt", sep=";")
driver = webdriver.Chrome()
driver.get("https://hellstore.me/ru/crash")
sleep(2)

while True:
    counter += 1

    try:
        element = WebDriverWait(driver, 600).until(
            EC.presence_of_element_located((By.CLASS_NAME, "game-timer-crashed"))
        )
    except TimeoutException as ex:
        print("Exception has been thrown" + str(ex))
    html = driver.page_source

    amount_of_players = float(driver.find_element(By.CLASS_NAME, "game-stat-value").text)

    betted_amount = list()
    all_betted = driver.find_elements(By.CLASS_NAME, "player-betted-amount")
    for bet in all_betted:
        betted_amount.append(float(bet.text[1:]))

    real_wins = list()
    all_wins = driver.find_elements(By.CLASS_NAME, 'player-winned-amount')
    for win in all_wins:
        if win.text != '':
            real_wins.append(float(win.text[1:]))

    round_result = round(sum(betted_amount) - sum(real_wins), 2)

    f = open(r"F:\folders\casino\mo.txt", "r").readlines()
    a = []
    for x in f:
        a.append(round(float(x[0:-1]), 1))
    thre = 3
    for i in range(len(a)):
        if a[i] > thre:
            a[i] = thre
    d = {}
    for i in a:
        if i in d:
            d[i] += 1
        else:
            d[i] = 1
    sorted_tuple = sorted(d.items(), key=lambda x: x[0])
    d = dict(sorted_tuple)
    s = 0
    num = 0        
    for i in d.values():
        num += i
    for i in d.keys():
        tmp = d.get(i)
        s += (i * tmp / num)
    mo = round(s, 3)
    bank = round(sum(betted_amount), 2)
    timee = round(time.time(), 2)
    wkday = (datetime.datetime.today().weekday()) + 1
    f = open("hist.txt", "a")
    e1 = driver.find_element(By.CLASS_NAME, "game-timer-crashed")
    f.write(e1.text[0:-1])
    f.write("\n")
    f.close()
    coeff = driver.find_element(By.CLASS_NAME, "game-timer-crashed")
    coeff = float(coeff.text[0:-1])
    ff = open(r"F:\folders\casino\mo.txt", "a")
    ff.write(f'{coeff}')
    ff.write('\n')
    ff.close()
    df.dropna(inplace=True)

    file = open(r'F:\folders\casino\histtt.txt', 'a')
    file.write(f'{amount_of_players};{round_result};{mo};{bank};{timee};{wkday};{coeff}')
    file.write('\n')
    file.close()

    df['amount players'] = df['amount players'].astype(float)
    df['round result'] = df['round result'].astype(float)
    df['MO'] = df['MO'].astype(float)
    df['bank'] = df['bank'].astype(float)
    df['timee'] = pd.to_datetime(df['timee'], unit='s')
    df['hour'] = df['timee'].dt.hour
    df['day_of_week'] = df['timee'].dt.dayofweek
    df['day_of_month'] = df['timee'].dt.day
    df['month'] = df['timee'].dt.month
    df['year'] = df['timee'].dt.year
    df['wkday'] = df['wkday'].astype(int)
    df['coeff'] = df['coeff'].astype(float)

    df['target'] = (df['coeff'] > 1.2).astype(int)

    X = df[['amount players', 'round result', 'MO', 'bank','hour', 'day_of_week', 'day_of_month', 'month', 'year', 'wkday']].values
    y = df['target'].values

    if counter > 5:
        scaler_X = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)

        def create_windows(data, target, window_size):
            Xs, ys = [], []
            for i in range(len(data) - window_size):
                Xs.append(data[i:i + window_size])
                ys.append(target[i + window_size])
            return np.array(Xs), np.array(ys)

        window_size = 11
        X_windows, y_windows = create_windows(X_scaled, y, window_size)
        X_train, X_test, y_train, y_test = train_test_split(X_windows, y_windows, test_size=0.2, random_state=42)

        model = Sequential()
        model.add(LSTM(64, input_shape=(window_size, X_train.shape[2]), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(32))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

        loss, accuracy = model.evaluate(X_test, y_test)

        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)
        print(f'Accuracy: {accuracy}',end=' ')
        print(f'Loss: {loss}',end=' ')
        print(f'MAPE: {mape(y_test, y_pred_prob)}',end=' ')
        print(y_pred, y_test)

    sleep(10)