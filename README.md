# building-detection
🏗️ Building Segmentation API
📂 Folder: building_detection
 📄 Main File: buildingdetectionapi.py

✅ What This Project Does
This project takes a .tif satellite image as input and returns a .geojson file that shows the building footprints detected in the image.
Everything is packed into a FastAPI app – just run it, upload your image, and get the building map back!

🚶‍♂️Step-by-Step Overview
User uploads a .tif file through the API.


The image is split into smaller tiles so it's easier for the model to process.


Each tile is passed through a YOLOv8 model trained to find buildings.


The model gives building masks (white where buildings are).


All tiles are merged back into one big mask.


We remove small noisy areas that aren’t real buildings.


The final mask is converted to vector shapes (GeoJSON) using geolocation.


The user gets a .geojson file as the result showing all detected buildings.





🧠 What's Inside
buildingdetectionapi.py contains all the logic:


Loading and running the model.


Splitting and merging image tiles.


Removing small blobs.


Converting to GeoJSON.


Creating the FastAPI endpoint.



📦 How to Use It
Keep the file inside a folder called building_detection.


Run the FastAPI app.


Hit the /process_tiff/ endpoint and upload a .tif satellite image.


Get a .geojson file in return that shows all detected buildings.
 
