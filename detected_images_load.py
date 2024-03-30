import asyncio
import os

import gridfs
import cv2
import numpy as np
from dotenv import load_dotenv, find_dotenv
from pymongo import MongoClient

load_dotenv(find_dotenv())

client = MongoClient(os.environ.get('CLIENT'))
db = client['Faces']
detected = db['Detected']

fs = gridfs.GridFS(db)


async def load_detected_images():
    for img in detected.find():
        photo = img['photo']
        gOut = fs.get(photo['imageID'])

        image = np.frombuffer(gOut.read(), dtype=np.uint8)
        image = np.reshape(image, photo['shape'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'./detected_images/{img["name"]}.jpg', image)


if __name__ == '__main__':
    asyncio.run(load_detected_images())
