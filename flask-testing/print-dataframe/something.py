from flask import Flask,redirect,render_template,url_for

app = Flask(__name__)

@app.route('/')
def index():
    render_template('home.html')
    for _ in range(100000000):
        continue
    
    return redirect(url_for('another_page'))

@app.route("/another_page")
def another_page():
    
    return '<h1> Hello there! <h1>'

app.run(debug=True)
