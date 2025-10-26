import requests

url = "http://localhost:9697/predict"
client = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}

response = requests.post(url, json=client)
result = response.json()

print(f"Probability: {result['convert_probability']}")
print(f"Full response: {result}")