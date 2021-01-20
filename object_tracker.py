import os

# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = "CPU"
import time
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from queue import Queue
import datetime
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-v3',
                    'path to weights file')
flags.DEFINE_integer('size', 608, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', '',
                    'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
# flags.DEFINE_string('output_format', 'mp4v', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', True, 'dont show video output')
flags.DEFINE_boolean('info', True, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', True, 'count objects being tracked on screen')


def main(_argv):
    # Definition of the parameters
    # 余弦距离的控制阈值
    max_cosine_distance = 0.4
    # 描述的区域的最大值
    nn_budget = None
    # 非极大抑制的阈值
    nms_max_overlap = 1.0

    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    # otherwise load standard tensorflow saved model
    else:
        # tf模型不是tflite
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']
    # begin video capture
    try:
        # cap = cv2.VideoCapture("./outputVideo.mp4")
        cap = cv2.VideoCapture("rtsp://admin:a1234567@10.34.142.35/cam/realmonitor?channel=1&subtype=0")
        # cap = cv2.VideoCapture("./test8.jpg")
        # cap = cv2.VideoCapture("&")
        print("连接摄像头成功！")
    except:
        print("连接摄像头失败！")

    out = None
    #
    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    # while video is running
    global timer
    while True:
        return_value, frame = cap.read()
        # if timer != 0 and int(time.time()) != timer:
        #     continue
        # print(time.time())
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break

        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]
        # 删除多余边框
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )
        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)
        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]
        # read in all class names from config
        # 从config.py文件中读取所有classes
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)
        # by default allow all classes in .names file
        # allowed_classes = list(class_names.values())

        # custom allowed classes (uncomment line below to customize tracker for only people)
        # 设置需要检测目标的class
        # allowed_classes = ['uniform', 'un_uniform', 'out_uniform']
        allowed_classes = ['uniform', 'un_uniform', 'out_uniform']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        print(names)
        count = len(names)
        # if count==0:
        #     print("无对象"+str(time.time()))
        #     continue
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,
                        (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)
        # encode yolo detections and feed to tracker
        # 预测结果输入deepsort
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                      zip(bboxes, scores, names, features)]

        # initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        # 创建保存标志
        saveFlag = False
        # 预测线
        yx1 = 0
        yy1 = 530
        yx2 = 1150
        yy2 = 390
        cv2.line(frame, (yx1, yy1), (yx2, yy2), (255, 0, 0))
        # 告警线
        gx1 = 0
        gy1 = 800
        gx2 = 1920
        gy2 = 480
        cv2.line(frame, (gx1, gy1), (gx2, gy2), (255, 0, 0))
        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()
            gj = isBelong(bbox, gx1, gy1, gx2, gy2)
            # 判断是否进入预警区域
            if not isBelong(bbox, yx1, yy1, yx2, yy2):
                print("未进入预警g区域")
                continue
            print("进入预警/告警区域了")
            # 创建告警标志
            exceptionFlag = False
            # 获取到轨迹字典
            map = persondict.get(int(track.track_id))
            # 判断是否有轨迹
            if map is not None:
                mflag = map["mflag"]
                # 告警过 并且 现在在不在告警区域 说明是第二人 则删除此人轨迹
                if mflag == 1 and not gj:
                    print("删除轨迹" + str(int(track.track_id)))
                    del persondict[int(track.track_id)]
                    continue
                # 获取轨迹队列
                que = map["que"]
                if que.full():
                    que.get()
                # 属于预警区域，存入轨迹队列
                que.put(int(bbox[3]))
                # 获取趋势值
                count = 0
                # 更新趋势值
                print("队列为" + str(que.queue))
                if len(que.queue) > 1:
                    for i in range(0, len(que.queue) - 1):
                        if (que.queue[i] <= que.queue[i + 1]):
                            count += 1
                # 判断是否需要告警  轨迹趋势+是否属于告警区域+是否为第一次告警+   经过预警区域
                if ((int(bbox[3]) - map["first"] > 30) and count >= 10) and gj and mflag == 0:
                    # 达到阈值，确定画框
                    exceptionFlag = True
                    # 清空轨迹队列
                    print("********************清空轨迹队列**************************************")
                    que.queue.clear()
                    print(que.queue)
                    map["que"] = que
                    map["mflag"] = 1
                    map["first"] = 1080
                    map["count"] = 0
                    # del persondict[int(track.track_id)]
                    # print(persondict.get(int(track.track_id)))
                    # 打开保存标志
                    if not saveFlag:
                        saveFlag = True
                        print("保存照片")
                else:
                    # 更新字典
                    map["count"] = count
            else:
                # 创建轨迹字典
                map = {}
                # 存入轨迹队列和趋势值
                que = Queue(100)
                map["que"] = que
                map["count"] = 0
                if gj:
                    map["mflag"] = 1
                else:
                    map["mflag"] = 0
                que.put(int(bbox[3]))
                map["first"] = int(bbox[3])
                # 首次出现，存入字典
                persondict[int(track.track_id)] = map

            if exceptionFlag:
                cv2.putText(frame, "save", (100, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
                # draw bbox on screen
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1] - 30)),
                              (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17, int(bbox[1])), color,
                              -1)
                cv2.putText(frame, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75,
                            (255, 255, 255), 2)
            # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id),
                                                                                                    class_name, (
                                                                                                        int(bbox[0]),
                                                                                                        int(bbox[1]),
                                                                                                        int(bbox[2]),
                                                                                                        int(bbox[3]))))

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # if timer != 0 and int(time.time()) != timer:
        #     continue
        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)

        # if output flag is set, save video file
        if saveFlag:
            imageName = str(time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))) + ".jpg"
            imagePath = r"/photo/ism_uniform/35/" + str(datetime.datetime.now().strftime("%Y-%m-%d")) + "/"
            if not os.path.exists(r"/photo/ism_uniform/35/"):
                os.makedirs(r"/photo/ism_uniform/35/")
            if not os.path.exists(r"/photo/ism_uniform/35/" + "/" + str(datetime.datetime.now().strftime("%Y-%m-%d"))):
                os.makedirs(r"/photo/ism_uniform/35/" + "/" + str(datetime.datetime.now().strftime("%Y-%m-%d")))
            # 上传图片至文件服务器
            # img = cv2.resize(result, (1360, 765))
            # cv2.imwrite(imagePath + imageName, result, [cv2.IMWRITE_JPEG_QUALITY, 75])
            cv2.imwrite(imagePath+imageName, result)
            print("Save!!!")
        if FLAGS.output:
            out.write(result)
        # timer = int(time.time() + 0.5)
        print(persondict)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()


def isBelong(bbox, x1, y1, x2, y2):
    # 获取框的中点坐标
    xb = int((bbox[0] + bbox[2]) * 0.5)
    yb = int(bbox[3])
    # 获取线的范围
    k = (y2 - y1) / (x2 - x1)
    b = y2 - k * x2
    # print("y={}*x+{}".format(k, b))
    # print("{}---{}".format(xb, yb))
    y = k * xb + b
    # print("y={}".format(y))
    # 比较yb和y判断此点是否在区域中
    if yb > y:
        # 在区域中
        return True
    return False


if __name__ == '__main__':
    try:
        persondict = {}
        timer = 0
        app.run(main)
    except SystemExit:
        pass
