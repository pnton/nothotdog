import os
import glob
import imghdr
import random
from flask import Flask
from flask import jsonify
from flask import request, url_for, render_template
import predict
import boto3
import botocore

BUCKET_NAME = 'nothotdog-trained-model'
KEY = 'checkpoint.pth'

tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')

app = Flask(__name__, template_folder=tmp_dir)
app.config['UPLOAD_FOLDER'] = 'static/img'

valid_mimetypes = ['image/jpeg', 'image/png']

# Code for downloading model from S3 only needs to be executed once when hosted on Paperspace
# Comment this section out after code has been run once

s3 = boto3.resource('s3')

try:
    s3.Bucket(BUCKET_NAME).download_file(KEY, '/storage/checkpoint.pth')
except botocore.exceptions.ClientError as e:
    if e.response['Error']['Code'] == "404":
        print("The object does not exist.")
    else:
        raise


"""
Index view
"""
@app.route('/')
def index():
    # Sort files by upload date
    recent_files = sorted(
        glob.glob("%s/*" % app.config['UPLOAD_FOLDER']), 
        key=os.path.getctime, reverse=True
    )
    # Pick the most recent two or less for the index view
    slice_index = 2 if len(recent_files) > 1 else len(recent_files)
    recents = recent_files[:slice_index]
    return render_template('index.html', recents=recents)
    
"""
Endpoint for hot dog checking
"""
@app.route('/is-hot-dog', methods=['POST'])
def is_hot_dog():
    if request.method == 'POST':
        if not 'file' in request.files:
            return jsonify({'error': 'no file'}), 400
        # Image info
        img_file = request.files.get('file')
        img_name = img_file.filename
        mimetype = img_file.content_type
        # Return an error if not a valid mimetype
        if not mimetype in valid_mimetypes:
            return jsonify({'error': 'bad-type'})
        # Write image to static directory and do the hot dog check
        img_file.save(os.path.join(app.config['UPLOAD_FOLDER'], img_name))
        probs, classes, confidence = predict.main(img_name)
        # Delete image when done with analysis
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], img_name))
        is_hot_dog = 'false' if classes[0] == 'not hotdog' else 'true'
        return_packet = {
            'is_hot_dog': is_hot_dog,
            'confidence': confidence
        }
        return jsonify(return_packet)
        
if __name__ == "__main__":
	app.run(host='0.0.0.0')
