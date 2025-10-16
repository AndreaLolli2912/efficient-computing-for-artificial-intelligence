import adafruit_dht
import uuid
import time
from datetime import datetime
from board import D4 as D4

mac_address = hex(uuid.getnode())
dhtDevice = adafruit_dht.DHT11(D4)

REDIS_HOST = 'redis-15750.c135.eu-central-1-1.ec2.redns.redis-cloud.com'
REDIS_PORT = 15750
REDIS_USERNAME = 'default'
REDIS_PASSWORD = 'r35F7Gez05k66A86KA9JcSfqdZL9ekrG'

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    username=REDIS_USERNAME,
    password=REDIS_PASSWORD
)

assert redis_client.ping(), 'Could not connect to Redis'

try:
    redis_client.ts().create('mytimeseries')
except redis.ResponseError:
    pass


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