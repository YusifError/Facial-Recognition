import gridfs
import cv2
import os
import numpy as np
from dotenv import load_dotenv, find_dotenv
from pymongo import MongoClient
from datetime import datetime, timedelta


load_dotenv(find_dotenv())

client = MongoClient(os.environ.get('CLIENT'))
db = client['Faces']
celebrities = db['Celebrities']
people = db['People']
detected = db['Detected']

fs = gridfs.GridFS(db)


def load_images():
    for img in people.find():
        meta = img['meta']
        gOut = fs.get(meta['imageID'])

        image = np.frombuffer(gOut.read(), dtype=np.uint8)
        image = np.reshape(image, meta['shape'])
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'./images/{img["name"]}.jpg', image)


async def insert(name: str):
    now = datetime.utcnow()

    insert_data = {
        'name': name,
        'date': now
    }

    detected.insert_one(insert_data)


async def check(name: str):
    now = datetime.utcnow()
    human = detected.find_one({'name': name})

    if human is not None:
        detection_date = human['date']
        if now - timedelta(minutes=2) >= detection_date:
            detected.update_one({'name': name}, {'$set': {'date': now}})
        else:
            pass
    else:
        await insert(name)