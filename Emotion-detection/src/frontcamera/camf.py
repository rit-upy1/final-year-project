from webcam import VideoCamera
import base64,cv2

from flask import Flask, render_template, request, jsonify, Response
import requests


app=Flask(__name__)

@app.route('/')
def home_page():
    return render_template("camera.html")

def gen(camera):
    while True:
        data= camera.get_frame()

        frame=data[0]
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')






if __name__=="__main__" :
    app.run(debug=True,port=1999)
    





