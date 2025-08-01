import base64
import requests
import os

# Path to local image to send
image_path = "sample_ecg1.jpg"  # Change this if needed

# Convert image to base64
with open(image_path, "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode("utf-8")

# Send request to API
response = requests.post(
    "http://127.0.0.1:5000/process_ecg",
    json={"image": base64_image}
)

# Handle response
if response.status_code == 200:
    data = response.json()

    # Save returned ECG paper image
    if "ecg_image_base64" in data:
        result_img_data = base64.b64decode(data["ecg_image_base64"])
        result_path = "output/result_ecg_from_api.png"
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, "wb") as f:
            f.write(result_img_data)
        print(f"[✅] ECG paper saved to: {result_path}")
    else:
        print("[⚠️] ECG paper missing from response.")

    # Print analysis
    if "description" in data:
        print("[✅] ECG Analysis:")
        print(data["description"])
    else:
        print("[⚠️] Analysis missing from response.")

else:
    print("[❌] Request failed:")
    print(response.status_code, response.text)
