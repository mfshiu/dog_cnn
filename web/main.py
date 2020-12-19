from flask import Flask
from flask import render_template
from flask import request
from config import DevConfig
import os
from verifydog import verify_dogs
import random


app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = "./static/upload"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config.from_object(DevConfig)

global left_dog
left_dog = "0001"

global right_dog
right_dog = "right_dog.PNG"


@app.route("/", methods=['GET', 'POST'])
@app.route("/demo1", methods=['GET', 'POST'])
def demo1():
    global left_dog
    global right_dog

    if request.method == 'POST':
        if 'right_dog' in request.files:
            file = request.files['right_dog']
            if file:
                right_dog = "right_{}.png".format(random.randint(1, 999999))
                path = os.path.join(app.config['UPLOAD_FOLDER'], right_dog)
                file.save(path)

    dog = request.args.get('dog')
    if not dog:
        dog = left_dog

    left_dog = dog
    left = "dogs/" + left_dog + "/0.PNG"
    right = "upload/" + right_dog
    return render_template("demo1.html", dog_left=left, dog_right=right)


@app.route("/demo2")
def demo2():
    return render_template("demo2.html")


@app.route("/demo3")
def demo3():
    return render_template("demo3.html")


@app.route("/verify_dog")
def verify_dog():
    is_equal, similarity = verify_dogs(left_dog, "upload/" + right_dog)
    return {'is_equal': is_equal, 'similarity': similarity}


@app.route("/home")
def home():
    return render_template("home.html")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
