from pymongo import MongoClient
from dotenv import load_dotenv, find_dotenv
import gridfs
import os
import cv2

load_dotenv(find_dotenv())

client = MongoClient(os.environ.get('CLIENT'))
db = client['Faces']
celebrities = db['Celebrities']
people = db['People']

fs = gridfs.GridFS(db)

for file in os.listdir('./images'):
    image = cv2.imread(f'./images/{file}')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    imageString = image.tobytes()

    imageID = fs.put(imageString, encoding='utf-8')

    filename, _ = os.path.splitext(file)

    meta = {
        'name': filename,
        'meta': {
            'imageID': imageID,
            'shape': image.shape,
            'dtype': str(image.dtype)
        }
    }

    people.insert_one(meta)
