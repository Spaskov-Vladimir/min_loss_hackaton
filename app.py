from flask import Flask, render_template, make_response, request
import os
from keras.models import load_model
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

UPLOAD_FOLDER = "/root/hackaton/flask/static/image_data/"

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def check_details():

    model = load_model('model.h5')
    img_path = "/root/hackaton/flask/static/image_data/data.jpg"
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)

    class_names = ['CS120.01.413', 'CS120.07.442', 'CS150.01.427-01', 'SU160.00.404', 'SU80.01.426', 'SU80.10.409A', 'ЗВТ86.103К-02', 'СВМ.37.060', 'СВМ.37.060А', 'СВП-120.00.060', 'СВП120.42.020', 'СВП120.42.030', 'СК20.01.01.01.406', 'СК20.01.01.02.402', 'СК30.01.01.02.402', 'СК30.01.01.03.403', 'СК50.01.01.404', 'СК50.02.01.411', 'СПО250.14.190']

    return class_names[np.argmax(preds)]

@app.route('/mobile', methods=['POST'])
def mobile_interface():
    if request.method == 'POST':
        if 'file' not in request.files:
            return make_response("Ошибка, файл не загружен", 404)
        file = request.files['file']
        if file.name == '':
            return make_response("Ошибка, файл не загружен", 404)
        if file and allowed_file(file.filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], "data.jpg"))
            text_data = check_details()
            return text_data
        return make_response("Ошибка, файл не загружен", 404)
    else:
        return make_response("BAD REQUEST", 400)

@app.route('/', methods=['GET', 'POST'])
def web_interface():
    if request.method == 'POST':
        if 'file' not in request.files:
            return make_response("Ошибка, файл не загружен 1", 404)
        file = request.files['file']
        if file.name == '':
            return make_response("Ошибка, файл не загружен 2", 404)
        if file and allowed_file(file.filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], "data.png"))
            return render_template('index_with_image.html', model_data=check_details())
        return make_response('Ошибка, файл не загружен', 404)
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=80, host='0.0.0.0')

