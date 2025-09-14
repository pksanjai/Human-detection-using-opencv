import cv2
import imutils
import numpy as np
import argparse

def detect(frame):
    bounding_box_cordinates, weights = HOGCV.detectMultiScale(
        frame, winStride=(4, 4), padding=(8, 8), scale=1.03
    )

    person = 1
    for x, y, w, h in bounding_box_cordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'person {person}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        person += 1

    cv2.putText(frame, 'Status : Detecting', (40, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, f'Total Persons : {person - 1}', (40, 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow('output', frame)

    return frame

def detectByPathVideo(path, writer):
    video = cv2.VideoCapture(path)
    check, frame = video.read()
    if not check:
        print('Video Not Found. Please Enter a Valid Path.')
        return

    print('Detecting people...')
    while video.isOpened():
        check, frame = video.read()
        if not check:
            break

        frame = imutils.resize(frame, width=min(800, frame.shape[1]))
        frame = detect(frame)

        if writer is not None:
            writer.write(frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

def detectByCamera(writer):
    video = cv2.VideoCapture(0)
    print('Detecting people...')

    while True:
        check, frame = video.read()
        if not check:
            break

        frame = imutils.resize(frame, width=min(800, frame.shape[1]))
        frame = detect(frame)

        if writer is not None:
            writer.write(frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

def detectByPathImage(path, output_path):
    image = cv2.imread(path)
    if image is None:
        print('Image not found. Please check the path.')
        return

    image = imutils.resize(image, width=min(800, image.shape[1]))
    result_image = detect(image)

    if output_path is not None:
        cv2.imwrite(output_path, result_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def humanDetector(args):
    image_path = args["image"]
    video_path = args["video"]
    camera = args["camera"]
    output_path = args["output"]

    writer = None

    if output_path is not None and image_path is None:
        # Get width and height from a dummy frame for consistency
        frame_width = 800
        frame_height = 600
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), 10, (frame_width, frame_height))

    if camera:
        print('[INFO] Opening Web Cam.')
        detectByCamera(writer)
    elif video_path is not None:
        print('[INFO] Opening Video from path.')
        detectByPathVideo(video_path, writer)
    elif image_path is not None:
        print('[INFO] Opening Image from path.')
        detectByPathImage(image_path, output_path)
    else:
        print("No input source provided. Use --image, --video, or --camera True")

def argsParser():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-v", "--video", default=None, help="Path to video file.")
    arg_parse.add_argument("-i", "--image", default=None, help="Path to image file.")
    arg_parse.add_argument("-c", "--camera", action='store_true', help="Use camera for live detection.")
    arg_parse.add_argument("-o", "--output", type=str, help="Path to optional output video file.")
    return vars(arg_parse.parse_args())

if __name__ == "__main__":
    HOGCV = cv2.HOGDescriptor()
    HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    args = argsParser()
    humanDetector(args)
