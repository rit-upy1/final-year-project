import cv2
from imutils.video import WebcamVideoStream


class VideoCamera(object):
    def __init__(self):
        while(True):
            self.stream = WebcamVideoStream(src=0).start()

    def __del__(self):
        self.stream.stop()


    def gen(camera):
        while True:
            data = camera.get_frame()

            frame = data[0]
            yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


    def get_frame(self):
        image = self.stream.read()

        ret, jpeg = cv2.imencode('.jpg', image)
        data = []
        data.append(jpeg.tobytes())
        data.append(name)
        return data
