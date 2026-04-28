import requests

url = "https://sensor-backend-521504670907.asia-southeast1.run.app/analyze"
file_path = "download.mp4"

with open(file_path, "rb") as f:
    files = {"video": (file_path, f, "video/mp4")}
    response = requests.post(url, files=files)

print("Status Code:", response.status_code)
print("Response:", response.json())
