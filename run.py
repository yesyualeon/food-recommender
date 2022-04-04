from flask import Flask

app = Flask(__name__)

from app.module.controller import *
from distutils.log import debug
from app import app
app.run(debug=True, host='127.0.0.1', port=5000)