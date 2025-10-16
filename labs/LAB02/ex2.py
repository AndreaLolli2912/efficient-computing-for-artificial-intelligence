import adafruit_dht
import uuid
import time
from datetime import datetime
from board import D4 as D4

mac_address = hex(uuid.getnode())
dhtDevice = adafruit_dht.DHT11(D4)

TEMPERATURE_TS = 'TemperatureTS'
HUMIDITY_TS = 'HumidityTS'

redis_client = redis.Redis(
    host='redis-15750.c135.eu-central-1-1.ec2.redns.redis-cloud.com',
    port=15750,
    username='default',
    password='r35F7Gez05k66A86KA9JcSfqdZL9ekrG'
)

assert redis_client.ping(), 'Could not connect to Redis'

try:
    redis_client.ts().create(TEMPERATURE_TS)
except redis.ResponseError:
    pass

try:
    redis_client.ts().create(HUMIDITY_TS)
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
        
        redis_client.ts().add(
            key=TEMPERATURE_TS,
            timestamp=timestamp, 
            value=temperature,
            labels={'time':formatted_datetime, 'mac':mac_address}
        )
        
        redis_client.ts().add(
            key=HUMIDITY_TS,
            timestamp=timestamp,  
            value=humidity,
            labels={'time':formatted_datetime, 'mac':mac_address}
        )

        
    except:
        print('%s - sensor failure\n'%(formatted_datetime))
        dhtDevice.exit()
        dhtDevice = adafruit_dht.DHT11(D4)

    time.sleep(2)