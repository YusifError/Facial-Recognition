from dotenv import load_dotenv, find_dotenv
from pymongo import MongoClient
from datetime import datetime, timedelta
import gridfs
import cv2
import os
import numpy as np


load_dotenv(find_dotenv())

client = MongoClient(os.environ.get('CLIENT'))
db = client['Faces']
celebrities = db['Celebrities']
people = db['People']
detected = db['Detected']

fs = gridfs.GridFS(db)


async def load_images():
    for img in people.find():
        meta = img['meta']
        gOut = fs.get(meta['imageID'])

        image = np.frombuffer(gOut.read(), dtype=np.uint8)
        image = np.reshape(image, meta['shape'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'./images/{img["name"]}.jpg', image)


async def insert(name: str, image) -> None:
    now = datetime.utcnow()

    image_string = image.tobytes()
    image_id = fs.put(image_string, encoding='utf-8')

    insert_data = {
        'name': name,
        'date': now,
        'photo': {
            'imageID': image_id,
            'shape': image.shape,
            'dtype': str(image.dtype)
        }
    }

    detected.insert_one(insert_data)


async def check(name: str, image) -> None:
    now = datetime.utcnow()
    human = detected.find_one({'name': name})
    image_string = image.tobytes()
    image_id = fs.put(image_string, encoding='utf-8')

    if human is not None:
        detection_date = human['date']
        if now - timedelta(minutes=2) >= detection_date:
            detected.update_one({'name': name}, {'$set': {'date': now}})
            detected.update_one({'name': name},
                                {'$set': {'photo': {
                                    'imageID': image_id,
                                    'shape': image.shape,
                                    'dtype': str(image.dtype)
                                }}})
        else:
            pass
    else:
        await insert(name, image)
