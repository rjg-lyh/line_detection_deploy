#include<opencv2/opencv.hpp>
#include<iostream>
#include<iostream>
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
	//VideoWriter writer("C:/Users/Dell/Desktop/picture/test.mp4", CAP_OPENCV_MJPEG, fps, S, true);

  while(1){
    Mat frame;
    capture>>frame;
    resize(frame, frame, Size(1422, 800));
    imshow("摄像头采样画面", frame);
    waitKey(1);
  }

}

void test2(){

  VideoCapture capture;
  // capture.open(4, VideoWriter::fourcc('M', 'J', 'P', 'G'),
	// 	75, //录制时的帧率，最好和相机采集帧率一致
	// 	Size(640, 480),
	// 	true);
  //capture.set(CAP_PROP_FRAME_WIDTH, 640);
  //capture.set(CAP_PROP_FRAME_HEIGHT, 480);
  //capture.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));
  cout << "是否成功打开： " << capture.isOpened() << endl;
  // while(1){
  //   Mat frame;
  //   capture>>frame;
  //   resize(frame, frame, Size(800, 800));
  //   imshow("摄像头采样画面", frame);
  //   waitKey(30);
  // }
}

int main(int argc,char** argv){

  test1();

  return 0;
}