import asyncio
import cv2
from simple_facerec import SimpleFacerec
from datetime import datetime, timedelta

# Распознование базы наших лиц
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# Загрузка камеры
cap = cv2.VideoCapture(0)


async def main():
    while True:
        ret, frame = cap.read()

        # Распознание персон (лиц)
        face_locations, face_names = await sfr.detect_known_faces(frame)
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

if __name__ == '__main__':
    asyncio.run(main())
