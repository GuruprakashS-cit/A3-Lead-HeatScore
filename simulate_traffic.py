import requests
import json
import time

API_URL = "http://127.0.0.1:8000/recommend/"
NUM_REQUESTS = 10

sample_lead = {
  "lead_id": 1,
  "source": "website",
  "recency_days": 5,
  "region": "NA",
  "role": "Data Scientist",
  "campaign": "AI/ML",
  "page_views": 10,
  "last_touch": "Website Visit",
  "prior_course_interest": 1
}

print(f"Sending {NUM_REQUESTS} requests to the API...")
for i in range(NUM_REQUESTS):
    response = requests.post(API_URL, json=sample_lead)
    print(f"Request {i+1}: Status {response.status_code}")
    time.sleep(1) # Wait for 1 second between requests

print("Finished simulating traffic. Check the log file for data.")
