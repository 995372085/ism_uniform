import cv2
import time

if __name__ == '__main__':
    cap = cv2.VideoCapture("rtsp://admin:a1234567@10.34.142.35/cam/realmonitor?channel=1&subtype=0")
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = int(cap.get(cv2.CAP_PROP_FPS))
    # codec = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    # out = cv2.VideoWriter("testa.avi", codec, fps, (width, height))
    while True:
        return_value, frame = cap.read()
        # out.write(frame)
        cv2.imshow("out", frame)
        cv2.imwrite(str(time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))) + "-35.jpg", frame)
        print("保存" + str(time.strftime('%Y-%m-%d#%H:%M:%S', time.localtime(time.time()))) + ".jpg")
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()
