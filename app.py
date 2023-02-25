import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import load_model


# LOADING THE MODEL

model = load_model("detection_model.h5")


from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)
camera = cv2.VideoCapture(0)

def face_extraction(frame):

    ''' Detect faces in a frame and extract them '''

    faces = cascade_model.detectMultiScale(frame, 1.1, 5)

    for x, y, w, h in faces:
        frame = frame[y:y+h, x:x+w]

    return frame





def image_processing(frame):

    ''' Preprocessing of the image for predictions '''
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (48, 48))
    frame = image.img_to_array(frame)
    frame = frame/255
    frame = np.expand_dims(frame, axis=0)

    return frame




def detect_expressions(frame, detection_model):

    ''' Detect final expressions and return the predictions
        done by the detection_model '''

    cropped_frame = face_extraction(frame)

    test_frame = image_processing(cropped_frame)

    prediction = np.argmax(model.predict(test_frame), axis=-1)

    return prediction




# LOADING HAAR CASCADE CLASSIFIER

cascade_model = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")




def generate_frames():
    while True:
            
        ## read the camera frame
        success, frame = camera.read()
        if not success:
            break
        else:
            faces = cascade_model.detectMultiScale(frame, 1.1, 5)
        
            for x, y, w, h in faces:

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)    
                
                prediction = detect_expressions(frame, model)
                
                font = cv2.FONT_ITALIC

                if prediction == [0]:
                    cv2.putText(frame, "Angry", (x, y), font, 1, (0, 0, 255), 2)

                elif prediction == [1]:
                    cv2.putText(frame, "Happy", (x, y), font, 1, (0, 0, 255), 2)

                elif prediction == [2]:
                    cv2.putText(frame, "Sad", (x, y), font, 1, (0, 0, 255), 2)

                else:
                    cv2.putText(frame, "Surprised", (x, y), font, 1, (0, 0, 255), 2)
                    
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)
    
