import redis
import time
import matplotlib.pyplot as plt
from datetime import datetime

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

is_connected = redis_client.ping()
print('Redis Connected:', is_connected)
assert is_connected, "Failed to connect to Redis"


def plot_redis_timeseries(redis_client):
    plt.ion()  # Turn on interactive mode
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Real-Time Redis Time Series")

    while True:
        # Fetch last 50 data points
        humidity_data = redis_client.ts().range('HumidityTS', '-', '+')[-50:]
        temperature_data = redis_client.ts().range('TemperatureTS', '-', '+')[-50:]

        # Convert timestamps to seconds and format
        humidity_times = [datetime.fromtimestamp(int(x[0]) // 1000).strftime("%H:%M:%S") for x in humidity_data]
        humidity_values = [x[1] for x in humidity_data]

        temperature_times = [datetime.fromtimestamp(int(x[0]) // 1000).strftime("%H:%M:%S") for x in temperature_data]
        temperature_values = [x[1] for x in temperature_data]

        # Clear previous plots
        ax1.clear()
        ax2.clear()

        # Plot humidity
        ax1.plot(humidity_times, humidity_values, marker='o', color='blue')
        ax1.set_title("Humidity")
        ax1.set_xlabel("Time (HH:MM:SS)")
        ax1.set_ylabel("Value")
        ax1.tick_params(axis='x', rotation=45)

        # Plot temperature
        ax2.plot(temperature_times, temperature_values, marker='o', color='red')
        ax2.set_title("Temperature")
        ax2.set_xlabel("Time (HH:MM:SS)")
        ax2.set_ylabel("Value")
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.draw()
        plt.pause(10)  # Refresh every 10 seconds

plot_redis_timeseries(redis_client)
