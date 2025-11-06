"""
a.	Create a new Python file (e.g., lab1_ex2.py) and write a script that uses the adafruit_dht module to read temperature and humidity from the DHT-11 sensor. 

b.	Every 2 seconds, print the T/H values using the following format:
year-month-day hour:minute:second.microseconds - mac_address:temperature = T
year-month-day hour:minute:second.microseconds - mac_address:humidity = H

where mac_address is the MAC address of your network card, 
T is the temperature value in degree Celsius, 
and H is the humidity value in %RH.

Example:
2022-10-01 19:21:51.699254 - 0xf0b61e0bfe09:temperature = 21
2022-10-01 19:21:51.699254 - 0xf0b61e0bfe09:humidity = 60
2022-10-01 19:21:53.701326 - 0xf0b61e0bfe09:temperature = 22
2022-10-01 19:21:53.701326 - 0xf0b61e0bfe09:humidity = 59
"""


import adafruit_dht
import uuid
import time
from datetime import datetime
from board import D4

mac_address = hex(uuid.getnode())
dht_device = adafruit_dht.DHT11(D4)

while True:
    timestamp = time.time()
    formatted_datetime = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')
    try:
        temperature = dht_device.temperature
        humidity = dht_device.humidity

        print(f'{formatted_datetime} - {mac_address}:temperature = {temperature}')
        print(f'{formatted_datetime} - {mac_address}:humidity = {humidity}')
    except:
        print(f'{formatted_datetime} - sensor failure')
        dht_device.exit()
        dht_device = adafruit_dht.DHT11(D4)

    
    time.sleep(1)