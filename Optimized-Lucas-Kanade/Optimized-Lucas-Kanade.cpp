#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <math.h>
#include "unistd.h"
#include <iostream>
#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>
#include <numeric>
#define WINDOW "当前相机画面"
#define MCmD MouseClickedDelete_MinDistance
using namespace cv;
using namespace std;

Point2f point,previousPoint;
bool addRemovePt = false;
char filter_mode = 2;
Mat image;

void help()
{
    cout << "\nLukas-Kanade optical flow\n" << endl;
    cout << "\n"
        "\t press key A to setup auto-point-tracking\n"
        "\t click the mouse to add/delete new harris points\n" << endl;
}

static void onMouse(int event, int x, int y, int /*flags*/, void* /*param*/)
{
	switch(event)
	{	
		case CV_EVENT_LBUTTONDOWN:
        	{
				point = Point2f((float)x, (float)y);
				addRemovePt = true;
			    break;
		}
    	}
}
 
int main(int argc, char** argv)
{
    help();
    Point2f current_position;
    current_position.x = 0;
    current_position.y = 0;
    Point2f d_current_position;

    double MCmD = 9.0;
    int minNumber_corner = 8;
    int maxCorners = 200;
    VideoCapture cap;
    TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);
    Size subPixWinSize(10, 10), winSize(31, 31);//(31,31)
    double minDistance_Corners = 20;
    double quality_Corners = 0.1;
    int blockSize = 7;
    double K_corner = 0.01;
    const int MAX_COUNT = 100;
    bool needToInit = false;
    Mat gray, prevGray;
    vector<Point2f> points[2];
    int PyrLEVEL = 2; 
    double radiu_Corners = 5;

    if (argc == 1 || (argc == 2 && strlen(argv[1])== 1 && isdigit(argv[1][0])))
        {
            cap.open(argc == 2 ? argv[1][0] - '0' :0);
        }
    else if (argc == 2)
        cap.open(argv[1]);

    cap.open("VideoTest.mp4");
    namedWindow("Capture Window", 1);
    setMouseCallback("Capture Window", onMouse, 0); 
    Mat frame_1, frame_2;
    Mat diff_frame;
    cap >> frame_2;
    for (;;)
    {
        frame_2.copyTo(frame_1);
        cap >> frame_2;
        if (frame_2.empty())
            break;
        absdiff(frame_2, frame_1, diff_frame);
        diff_frame.copyTo(image);
        if (image.empty())
        {
            cout << "Capture None." << endl;
            break;
        }
    //************************ 相机标定数据 *****************************************

    //************************ 图像去畸变 *****************************************

        cvtColor(image, gray, COLOR_BGR2GRAY);
        if (needToInit)
        {
            goodFeaturesToTrack(gray, points[1],maxCorners, quality_Corners, minDistance_Corners, Mat(), blockSize, true, K_corner);
            cornerSubPix(gray, points[1],subPixWinSize, Size(-1, -1), termcrit);
            addRemovePt = false;
        }
        else if(!points[0].empty())
        {
            vector<uchar> status;
            vector<float> err;
            vector<Point2f> Dxy;
            double minEigTheshold = 0.001;
            calcOpticalFlowPyrLK(prevGray, gray,points[0], points[1], status, err, winSize,PyrLEVEL, termcrit, 0, minEigTheshold);
            size_t i, k;
            for (i = k = 0; i < points[1].size(); i++)
            {
                if (addRemovePt)
                {
                    if (norm(point -points[1][i]) <= MCmD)
                    {
                        addRemovePt = false;
                        continue;
                    }
                }
                if (!status[i])continue;
                points[1][k++] = points[1][i];
                Dxy.push_back(points[1][i]-points[0][i]);
                circle(image, points[1][i], radiu_Corners, Scalar(0, 0, 255), -1, 8);
            }
            points[1].resize(k);
            vector<double> DirectionJudge;
            for(i=0; i < Dxy.size(); i++)
            {
                if(Dxy[i].x != 0)   DirectionJudge.push_back(atan2(Dxy[i].y,Dxy[i].x));
                else if(Dxy[i].y>0) DirectionJudge.push_back(CV_PI/2);
                else if(Dxy[i].y<0) DirectionJudge.push_back(-CV_PI/2);
                else  DirectionJudge.push_back(0);
            }
            double sum = accumulate(DirectionJudge.begin(),DirectionJudge.end(),0);
            double accum  = 0.0;  
            for(i=0 ; i < Dxy.size(); i++)
            {  
                accum  += (DirectionJudge[i]-sum/DirectionJudge.size())*(DirectionJudge[i]-sum/DirectionJudge.size());  
            }
            double variance = sqrt(accum/(DirectionJudge.size()-1));   

            d_current_position.x = d_current_position.y = 0;
            for(i=0,k = 0; i < Dxy.size(); i++)
            {
                //************这里需要添加 "聚类" 以及 卡尔曼滤波 来区分动态物体和静态物体,识别到动态点则舍去*********************

                double temp_variance=0,temp_mean = (sum-DirectionJudge[i]) / (Dxy.size()-1);
                for(int temp=0; temp<Dxy.size();temp++)
                {
                    if(temp == i)continue;
                    temp_variance += (DirectionJudge[temp]-temp_mean)*(DirectionJudge[temp]-temp_mean);
                }
                temp_variance = sqrt(temp_variance/(DirectionJudge.size()-2));
                
                if(filter_mode == 1){
                    if( abs( (variance/temp_variance) *DirectionJudge[i] / ((sum - DirectionJudge[i])/ (DirectionJudge.size()-1))) > 3.6)//如果影响方差过大
                    {
                        continue;
                    }
                }
                else if(filter_mode == 2){
                    if( abs(variance/temp_variance) > 1.03)
                    {
                        if (abs(DirectionJudge[i] / ((sum - DirectionJudge[i])/ (DirectionJudge.size()-1))) > 1.5 )
                        {
                            continue;
                        }
                    }
                }
                points[1][k++] = points[1][i];
                d_current_position = Dxy[i] + d_current_position;
            }
            points[1].resize(k);
            if(points[1].size() != 0)
            {
                current_position.x += d_current_position.x/(double)points[1].size();
                current_position.y += d_current_position.y/(double)points[1].size();    
                cout << current_position << endl;
            }
            else
            {   
            }
        }
       
        if (addRemovePt && points[1].size() < (size_t)MAX_COUNT)
        {
            vector<Point2f> tmp;
            tmp.push_back(point);
            cornerSubPix(gray, tmp, winSize,cvSize(-1, -1), termcrit);
            points[1].push_back(tmp[0]);
            addRemovePt = false;
        }
 
        needToInit = false;
        imshow("Capture Window", image);
 
        char c = (char)waitKey(100);
        if (c == 27)
            break;
        switch (c)
        {
        case 'a':
            needToInit = true;
            break;
        case 'c':
            points[0].clear();
            points[1].clear();
            break;
        case 'n':
            break;
        }
        if(points[1].size()> 0 && points[1].size() < minNumber_corner)
        {
            needToInit = true;
            cout << "定位角点损失严重,寻找新角点." << endl;
        }
        // swap and continue
        std::swap(points[1], points[0]);
        cv::swap(prevGray, gray);
    }
    return 0;
}
