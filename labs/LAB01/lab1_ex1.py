import adafruit_dht
import uuid
import time
from datetime import datetime
from board import D4 as D4

mac_address = hex(uuid.getnode())
dhtDevice = adafruit_dht.DHT11(D4)

while True:
    timestamp = time.time()
    formatted_datetime = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')
    try:
        temperature = dhtDevice.temperature
        humidity = dhtDevice.humidity
        
        print("%s - %s:temperature = %.3f"%(formatted_datetime, mac_address, temperature))
        print("%s - %s:humidity = %.3f\n"%(formatted_datetime, mac_address, humidity))
    except:
        print('%s - sensor failure\n'%(formatted_datetime))
        dhtDevice.exit()
        dhtDevice = adafruit_dht.DHT11(D4)

    time.sleep(1)