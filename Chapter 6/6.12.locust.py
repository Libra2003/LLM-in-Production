# First start a streaming server
# python chapters/chapter_6/listing_6.3.streaming.py

# Run the locust test
# locust -f chapters/chapter_6/listing_6.12.locust.py

import time
from locust import HttpUser, task, events

stat_file = open("start.csv", "w")
stat_file.write("Latency, TTFT, TPS\n")

class streamUser(HttpUser):
    @task

    def generate(self):

        token_count = 0
        start = time.time()

        #Make Request
        with self.client.post(
            "/generate",
            data = '{"prompt": "Salt late city is a}',
            catch_response=True,
            stream=True,
        ) as response:
            first_response = time.time()
            for line in response.iter_lines(decode_unicode=True):
                token_count += 1

        end = time.time()
        latency = end - start
        ttft = first_response - start
        tps = token_count / (end - first_response)

        stat_file.write(f"{latency}, {ttft},{tps}\n")

    
@events.quitting.add_listener
def close_stats_file(enviroment):
    stat_file.close()