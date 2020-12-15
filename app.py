# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, redirect, url_for, send_file
from summarize import summarize
import os

app = Flask(__name__)
port = 5000

def get_results(request):
    file = request.files['video-file']
    filename = file.filename
    print("Received file: " + filename)
    file.save(filename)
    summary_filename = 'summary.mp4'
    summarize(filename, summary_filename)
    print("Summarized into: " + summary_filename)
    os.remove(filename)
    return send_file(summary_filename, mimetype='video/mp4', as_attachment=True)

@app.route('/', methods=['GET', 'POST'])
def main_page():
    try:
        if request.method == 'POST':
            return get_results(request)
        return render_template('index.html')
    except Exception as e:
        return " Service error: " + str(e)
    
    
if __name__ == '__main__':
    
    app.run(host='localhost', port=port, debug=True)
