from flask import Flask
from flask import render_template
from config import DevConfig
from verifydog import verify_dogs


app = Flask(__name__, static_url_path='/static')
app.config.from_object(DevConfig)


@app.route("/")
def index():
    return render_template("demo1.html")
    # return render_template("index-dog.html")


@app.route("/demo1")
def demo1():
    return render_template("demo1.html")


@app.route("/demo2")
def demo2():
    return render_template("demo2.html")


@app.route("/demo3")
def demo3():
    return render_template("demo3.html")


@app.route("/verify_dog")
def verify_dog():
    is_equal = verify_dogs()
    return {'is_equal': is_equal}


@app.route("/home")
def home():
    return render_template("home.html")


if __name__ == "__main__":
    app.run()
