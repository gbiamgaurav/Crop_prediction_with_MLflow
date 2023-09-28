
import os
import requests
from bs4 import BeautifulSoup

# Define a dictionary to map crop names to image search queries
crop_image_queries = {
    "rice": "rice plant",
    "maize": "maize plant",
    "jute": "jute plant",
    "cotton": "cotton plant",
    "coconut": "coconut tree",
    "papaya": "papaya plant",
    "orange": "orange tree",
    "apple": "apple tree",
    "muskmelon": "muskmelon plant",
    "watermelon": "watermelon plant",
    "grapes": "grape vine",
    "mango": "mango tree",
    "banana": "banana plant",
    "pomegranate": "pomegranate plant",
    "lentil": "lentil plant",
    "blackgram": "black gram plant",
    "mungbean": "mung bean plant",
    "mothbeans": "moth bean plant",
    "pigeonpeas": "pigeon pea plant",
    "kidneybean": "kidney bean plant",
    "chickpea": "chickpea plant",
    "coffee": "coffee plant",
}

# Create a directory to save the downloaded images
output_directory = "downloaded_images"
os.makedirs(output_directory, exist_ok=True)

# Loop through the crop names and download images
for crop, query in crop_image_queries.items():
    # Define the URL for image search (you can use other search engines as well)
    search_url = f"https://www.google.com/imghp?hl=en"

    # Set user-agent header to avoid being blocked
    headers = {"User-Agent": "Mozilla/5.0"}

    # Send an HTTP GET request to the search URL
    response = requests.get(search_url, headers=headers)

    # Parse the HTML content of the response to extract image URLs
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        img_tags = soup.find_all("img", attrs={"data-src": True})

        # Download the first image (you can modify this to download more images)
        if img_tags:
            img_url = img_tags[0]["data-src"]
            img_response = requests.get(img_url, headers=headers)

            # Save the image to the output directory
            if img_response.status_code == 200:
                image_filename = f"{crop}.jpg"
                image_path = os.path.join(output_directory, image_filename)

                with open(image_path, "wb") as img_file:
                    img_file.write(img_response.content)
                print(f"Downloaded: {image_path}")
            else:
                print(f"Failed to download {crop} image.")
        else:
            print(f"No images found for {crop}.")
    else:
        print(f"Failed to retrieve search results for {crop}.")

print("Image download complete.")
