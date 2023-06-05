#include<opencv2/opencv.hpp>
#include<time.h>
#include<iostream>
#include<cmath>

using namespace std;
using namespace cv;

void test1(){

  VideoCapture capture;
  string pipeline = "v4l2src device=/dev/video4 ! video/x-raw,format=UYVY,width=1920,height=1080, \
  framerate=30/1! videoconvert ! appsink video-sink=xvimagesink sync=false";

  capture.open(pipeline, cv::CAP_GSTREAMER);

  cout << "是否成功打开： " << capture.isOpened() << endl;


  // Size S = Size((int)capture.get(CAP_PROP_FRAME_WIDTH),
	// 	(int)capture.get(CAP_PROP_FRAME_HEIGHT));
	// int fps = capture.get(CAP_PROP_FPS);
	// printf("current fps : %d \n", fps);
	// VideoWriter writer("/home/nvidia/frame_save/test2.avi", CAP_ANY, fps, S, true);

  Mat frame;
	namedWindow("camera-demo", 0);
  resizeWindow("camera-demo", 1422, 800); 
	while (capture.read(frame)) {
    //flip(frame, frame, -45);
		imshow("camera-demo", frame);
		//writer.write(frame);
		char c = waitKey(50);
		if (c == 27) {
			break;
		}
	}
	capture.release();
	//writer.release();

  // while(1){
  //   Mat frame;
  //   capture>>frame;
  //   resize(frame, frame, Size(1422, 800));
  //   imshow("摄像头采样画面", frame);
  //   waitKey(1);
  // }

}

void test2(){
  VideoCapture capture;
  string pipeline = "v4l2src device=/dev/video4 ! video/x-raw,format=UYVY,width=1920,height=1080, \
  framerate=30/1! videoconvert ! appsink video-sink=xvimagesink sync=false";

  capture.open(pipeline, cv::CAP_GSTREAMER);

  cout << "是否成功打开： " << capture.isOpened() << endl;


  Mat frame;
	namedWindow("camera-demo", 0);
  resizeWindow("camera-demo", 1422, 800);

  time_t start, end;
  start = clock();
  time_t interval = 4*pow(10, 6); //间隔 2S 拍一张
  int count = 1;
  string prefix = "save_picture/picture8_";

	while (capture.read(frame)) {
		imshow("camera-demo", frame);
    end = clock();
    if((end - start) > interval){
      imwrite(prefix + to_string(count) + ".jpg", frame);
      ++count;
      if(count == 10){
        break;
      }
      start = end;
    }
		char c = waitKey(50);
		if (c == 27) {
			break;
		}
	}
	capture.release();

}

int main(int argc,char** argv){

  test1();

  return 0;
}
