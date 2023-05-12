import cv2

# 设置gstreamer管道参数
def gstreamer_pipeline(
    capture_width=1920, #摄像头预捕获的图像宽度
    capture_height=1080, #摄像头预捕获的图像高度
    framerate=30,       #捕获帧率
):
    return (
        "v4l2src device=/dev/video4! "
        "video/x-raw, "
        "width=%d, height=%d, "
        "format=UYVY! "
        "videoconvert ! "
        "fpsdisplaysink video-sink=xvimagesink sync=false"
        % (
            capture_width,
            capture_height
        )
    )

if __name__ == "__main__":
    capture_width = 1920
    capture_height = 1080

    display_width = 1280
    display_height = 720

    framerate = 30			# 帧数
    flip_method = 0			# 方向

    # 创建管道
    print(gstreamer_pipeline(capture_width,capture_height,framerate))

    # #管道与视频流绑定
    #cap = cv2.VideoCapture(gstreamer_pipeline(capture_width,capture_height,framerate), cv2.CAP_GSTREAMER)
   
    cap = cv2.VideoCapture(4)
    print(cap.isOpened())
    if cap.isOpened():
        #window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)

        # 逐帧显示
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            ret_val, img = cap.read()
            #cv2.imshow("CSI Camera", img)

            keyCode = cv2.waitKey(30) & 0xFF
            if keyCode == 27:# ESC键退出
                break

        cap.release()
        #cv2.destroyAllWindows()
    else:
        print("打开摄像头失败") 