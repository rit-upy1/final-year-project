from flask import Flask,render_template,request,url_for
import sys 
import emotions
import os
from webcamera import webcam

#sys.path.append('/home/samyak/Downloads/final year project/Emotion-detection/src')
sys.path.append('/home/ritvik/Final Year Project/fyp')
import face_mood_analyser

app=Flask(__name__)


@app.route("/")
@app.route("/hom",methods=["post",'get'])
def hom():
    if request.method =='POST':
        user=request.form['sam']
        print("usER IS ",user)
    return render_template('home.html')


@app.route("/opencamera",methods=["post",'get'])
def opencamera():
    # return {'msg':"ok"}
    # return  emotions.display()
    return redirect(url_for("/recommendations"))

    
    
def get_mood():
    webcam.capture_pictures(10)
    return face_mood_analyser.get_mood('/home/ritvik/Final Year Project/Emotion-detection/src/webcamera/image')

    
if __name__=="__main__" :
    app.run(debug=True,port=7000)
