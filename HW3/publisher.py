import logging
import uuid

import adafruit_dht
from board import D4

from time import sleep, time
from json import dumps

import paho.mqtt.client as mqtt

if __name__ == "__main__":
    BROKER_ADDRESS:str = 'broker.emqx.io'
    BROKER_PORT:int  = 1883
    BROKER_TOPIC:str = 's344860'
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )

    logger = logging.getLogger(__name__)

    # Initialize the DHT-11 sensor to collect temperature and humidity data.
    mac_address = hex(uuid.getnode())
    logger.info("Initializing DHT-11 sensor...")
    dht_device = adafruit_dht.DHT11(D4)
    
    # Initialize MQTT client
    client = mqtt.Client()
    try:
        assert client.connect(BROKER_ADDRESS, BROKER_PORT) == 0, "Failed to connect to MQTT broker"
        logger.info("Connected to MQTT broker at %s:%d", BROKER_ADDRESS, BROKER_PORT)
    except Exception as e:
        logger.error("Could not connect to MQTT broker: %s", e)
        exit(1)
    
    while True:
        try:
            timestamp_ms = int(time() * 1000) # Convert Unix time in milliseconds and cast it to integer

            temperature = float(dht_device.temperature)
            humidity    = float(dht_device.humidity)
            
            logger.info(f"Reading: Temperature: {temperature} | Humidity: {humidity}")
            
            client.publish(BROKER_TOPIC, dumps({
                "mac_address": mac_address,
                "timestamp": timestamp_ms,
                "data":[
                    {"name": "temperature", "value": temperature},
                    {"name": "humidity", "value": humidity}
                ]
                }))

            logger.info("Uploaded via Mqtt timestamp=%d", timestamp_ms)
            
            sleep(5)
            
        except Exception as e:
            logger.warning("Sensor read failure")
            dht_device.exit()
            dht_device = adafruit_dht.DHT11(D4)
            continue