from simple_facerec import SimpleFacerec
from data import load_images, check
import asyncio
import cv2

# Распознование базы наших лиц
sfr = SimpleFacerec()

# Загрузка камеры
cap = cv2.VideoCapture(0)


async def main():
    while 1:
        ret, frame = cap.read()

        # Распознание персон (лиц)
        face_locations, face_names = await sfr.detect_known_faces(frame)
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            await check(name, bgr_frame)

        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break


async def run():
    print('Выберите пункт меню:\n'
          '1: Начать работу\n'
          '2: Обновить папку с картинками')
    point = input()

    if point == '1':
        await sfr.load_encoding_images('./images')
        await main()
    elif point == '2':
        await load_images()

if __name__ == '__main__':
    asyncio.run(run())
