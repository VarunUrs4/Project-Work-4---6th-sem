from flask import Flask, render_template, request
from keras.models import load_model
from keras_preprocessing import image
import info
import numpy as np

app = Flask(__name__)

model = load_model('model1.h5')
class_names = info.class_names
features = info.features


@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = './images/' + imagefile.filename
    imagefile.save(image_path)
    img = image.load_img(image_path, target_size=(224, 224, 3))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    pred = model.predict(images, batch_size=32)
    return render_template('results.html', type=class_names[np.argmax(pred)],
                           prediction=features[class_names[np.argmax(pred)]])


if __name__ == '__main__':
    app.run(port=3000, debug=True)
