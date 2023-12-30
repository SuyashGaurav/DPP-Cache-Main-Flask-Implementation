from flask import Flask, render_template, url_for, request, redirect, send_file
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import random, time, re
import pandas as pd
import os


app = Flask(__name__)

@app.route('/download/<string:content>')
def download(content):
    file_path = f'AllFiles/{content}.txt'
    return send_file(file_path, as_attachment=True)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
