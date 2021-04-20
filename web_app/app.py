from flask import Flask, render_template, request
import webbrowser, threading, os

app = Flask(__name__)  

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')   
# about page
@app.route('/about')
def about():
    return render_template('about.html')

# contact page
@app.route('/contact')
def contact():
    return render_template('contact.html')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)