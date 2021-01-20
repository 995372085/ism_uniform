
from flask import Flask, Response, request, render_template
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# 设置图片保存文件夹
UPLOAD_FOLDER = 'photo'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 设置允许上传的文件格式
ALLOW_EXTENSIONS = ['png', 'jpg', 'jpeg']


# 判断文件后缀是否在列表中
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[-1] in ALLOW_EXTENSIONS

@app.route("/photo/<algorithm>/<cameraIP>/<date>/<imageId>.jpg")
def get_frame(algorithm,cameraIP,date,imageId):
    # 图片上传保存的路径
    with open(r'/photo/{}/{}/{}/{}.jpg'.format(algorithm,cameraIP,date,imageId), 'rb') as f:
        image = f.read()
        resp = Response(image, mimetype="image/jpg")
        return resp

if __name__ == '__main__':
    app.run(host='10.34.1.153', port=5000, debug=True)