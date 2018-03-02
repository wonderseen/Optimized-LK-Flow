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
    // print a welcome message, and the OpenCV version
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
        /*
        case CV_EVENT_LBUTTONUP:
        		{	
				Point pt(x, y);
				line(image, previousPoint, pt, Scalar(0,0,255), 2, 5, 0);
				previousPoint = pt;
				imshow(WINDOW, image);
				p = false;
				break;
			}
		case CV_EVENT_MOUSEMOVE:
        		{	
				break;
            }
        */
    	}
}
 
int main(int argc, char** argv)
{
    help();
    Point2f current_position;
    current_position.x = 0;
    current_position.y = 0;
    Point2f d_current_position;
//*****************************检测采集卡是否有采集到图像*******************************
    /*
    FILE *camera,*grab;
    camera = fopen("dev/video1","rb");
    //grab = fopen("grab.raw","wb");
    int data[307200];
    cout << 1<<endl;
    fread(data, sizeof(data[0]), 307200, camera);
    */
    //fwrite(data, sizeof(data[0]), 307200, grab);
//***************************************初始化重要参数***************************
    double MCmD = 9.0;
    int minNumber_corner = 8;//最少角点数
    int maxCorners = 200;//设置角点识别的最大角点个数 
    VideoCapture cap;
    TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);//TermCriteria 迭代终止判断类型,包括了终止结束判断方式和一些阈值参数
    //CV_TERMCRIT_ITER、CV_TERMCRIT_EPS、CV_TERMCRIT_ITER+CV_TERMCRIT_EPS，分别代表着迭代终止条件为达到最大迭代次数终止，迭代到阈值终止，或者两者都作为迭代终止条件。
    Size subPixWinSize(10, 10), winSize(31, 31);//(31,31)
    double minDistance_Corners = 20;//设置寻找最强角点的最小比较半径 
    double quality_Corners = 0.1;//角点质量阈值 
    int blockSize = 7;          //协方差矩阵的窗口大小
    double K_corner = 0.01;     //Harris角点检测所需K值,越小检测到的角点越多,具体原因,待查!!!!
    const int MAX_COUNT = 100;  //设置最大特征点个数为100个
    bool needToInit = false;
    Mat gray, prevGray;
    vector<Point2f> points[2];  //记录当次和前一次的特征点,points[0]总是记录前一次的,points[1]记录最新的
    int PyrLEVEL = 2;           //检测动态光流所用的金字塔层数,0为不使用
    double radiu_Corners = 5;   //标签原点的半径

//********************************************打开电脑相机采集视频流方式1******************************************************//
    if (argc == 1 || (argc == 2 && strlen(argv[1])== 1 && isdigit(argv[1][0])))//如果没有设置参数,直接打开笔记本自带的相机.如果要打开其他相机,并且输入的相机代码是正确的
        /*
        isdigit()函数包含在ctype.h头文件中，
        原型： int isdigit(char c); 　　
        用法：#include <ctype.h> 　　
        功能：判断字符c是否为数字
        说明：当c为数字0-9时，返回非零值，否则返回零。 
        */
        {
            cap.open(argc == 2 ? argv[1][0] - '0' :0);
        }
        //if argc == 2 ; cap.open(argv[1][0]-'0') ; else cap.open(0);
    else if (argc == 2)
        cap.open(argv[1]);

//********************************************打开视频流方式2*******************************************************************//
    cap.open("VideoTest.avi");
    if (!cap.isOpened())
    {
        cout << "无法正常初始化相机...\n";
        return 0;
    }
    else cout << "正常初始化相机...\n";
 

//********************************************为画面设置鼠标反馈线程
    namedWindow("Capture Window", 1);
    setMouseCallback("Capture Window", onMouse, 0);
 
 //************************************************光流法处理过程
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




    //************************ 图像预处理 ***************************************
        //
        cvtColor(image, gray, COLOR_BGR2GRAY);

        if (needToInit)
        {
            //检测角点
            goodFeaturesToTrack(gray, points[1],maxCorners, quality_Corners, minDistance_Corners, Mat(), blockSize, 1, K_corner);            
            //goodFeaturesToTrack函数可以计算Harris角点和shi-tomasi角点，但默认情况下计算的是shi-tomasi角点，函数原型如下：
            //
            /*
                void cv::goodFeaturesToTrack( InputArray _image, OutputArray _corners,
                                            int maxCorners, double qualityLevel, double minDistance,
                                            InputArray _mask, int blockSize,
                                            bool useHarrisDetector, double harrisK )
                _image：8位或32位浮点型输入图像，单通道
                _corners：保存检测出的角点
                maxCorners：角点数目最大值，如果实际检测的角点超过此值，则只返回前maxCorners个强角点
                qualityLevel：角点的品质因子
                minDistance：对于初选出的角点而言，如果在其周围minDistance范围内存在其他更强角点，则将此角点删除
                _mask：指定感兴趣区，如不需在整幅图上寻找角点，则用此参数指定ROI
                blockSize：计算协方差矩阵时的窗口大小
                useHarrisDetector：指示是否使用Harris角点检测，如不指定，则计算shi-tomasi角点
                harrisK：Harris角点检测需要的k值
                角点的强度计算方法：若采用Fast-9-16,计算连续的9个位置与中心位置的差值的绝对值，取最小的一个差值作为其强度值。
                这里的K越小,检测到的角点越多
            */

            //对像素级角点的进一步优化,寻找亚像素角点,更精确
            cornerSubPix(gray, points[1],subPixWinSize, Size(-1, -1), termcrit);
            /*
                cv::goodFeaturesToTrack()提取到的角点只能达到像素级别，在很多情况下并不能满足实际的需求，
                这时，我们则需要使用cv::cornerSubPix()对检测到的角点作进一步的优化计算，可使角点的精度达到亚像素级别。
                    void cv::cornerSubPix(
                        cv::InputArray image, // 输入图像  
                        cv::InputOutputArray corners, // corners：输入角点的初始坐标以及精准化后的坐标用于输出。
                        cv::Size winSize, // 搜索窗口边长的一半,区域大小为 NXN; N=(winSize*2+1)  
                        cv::Size zeroZone, // 类似于winSize，但是总具有较小的范围，Size(-1,-1)表示忽略  
                        cv::TermCriteria criteria // 停止优化的标准  
                    );  
            */
            addRemovePt = false;
            
        }
        //*****************************************************一旦初始化得到角点后,这部分最重要,用来做光流定位!
        else if(!points[0].empty())
        {
            vector<uchar> status;   //得到新一帧每一个角点相对于前一帧的状态
            vector<float> err;      //误差矢量
            vector<Point2f> Dxy;
            double minEigTheshold = 0.001;//-minEigTheshold：算法计算的光流等式的2x2常规矩阵的最小特征值。
            calcOpticalFlowPyrLK(prevGray, gray,points[0], points[1], status, err, winSize,PyrLEVEL, termcrit, 0, minEigTheshold);
            
            size_t i, k;
            for (i = k = 0; i < points[1].size(); i++)
            {
                if (addRemovePt)
                {
                    if (norm(point -points[1][i]) <= MCmD)//如果鼠标点击的点离某个确定角点太近,取消该角点
                    {
                        addRemovePt = false;
                        continue;
                    }
                }
                if (!status[i])continue;
                //新的可用角点按顺序拍在前K位置
                points[1][k++] = points[1][i];
                //写入可用的光流矢量
                Dxy.push_back(points[1][i]-points[0][i]);
                //画角点
                circle(image, points[1][i], radiu_Corners, Scalar(0, 0, 255), -1, 8);
            }
            //把K后面的无用点都舍去
            points[1].resize(k);

             //********************输出光流矢量,根据角度差值,再进行一次角点滤波

            vector<double> DirectionJudge;
            for(i=0; i < Dxy.size(); i++)//转换的到角点移动矢量的角度
            {
                if(Dxy[i].x != 0)   DirectionJudge.push_back(atan2(Dxy[i].y,Dxy[i].x));
                else if(Dxy[i].y>0) DirectionJudge.push_back(CV_PI/2);
                else if(Dxy[i].y<0) DirectionJudge.push_back(-CV_PI/2);
                else  DirectionJudge.push_back(0);
            }
            //********************输出光流矢量,根据角度方差,再进行一次角点滤波
            double sum = accumulate(DirectionJudge.begin(),DirectionJudge.end(),0);
            //*********************************** 整体方差variance
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

                //**********************************局部方差********************************
                double temp_variance=0,temp_mean = (sum-DirectionJudge[i]) / (Dxy.size()-1);//互略某个点的方差
                for(int temp=0; temp<Dxy.size();temp++)
                {
                    if(temp == i)continue;
                    temp_variance += (DirectionJudge[temp]-temp_mean)*(DirectionJudge[temp]-temp_mean);
                }
                temp_variance = sqrt(temp_variance/(DirectionJudge.size()-2));
                
                if(filter_mode == 1){
                // Filter 1: 均值和方差系数相乘
                    //cout << " "  <<abs(variance/temp_variance);
                    //cout << abs(DirectionJudge[i] / ((sum - DirectionJudge[i])/ (DirectionJudge.size()-1)))<< "  ";
                    // cout << abs( (variance/temp_variance) *DirectionJudge[i] / ((sum - DirectionJudge[i])/ (DirectionJudge.size()-1)))<< endl;                
                    if( abs( (variance/temp_variance) *DirectionJudge[i] / ((sum - DirectionJudge[i])/ (DirectionJudge.size()-1))) > 3.6)//如果影响方差过大
                    {
                        //cout << DirectionJudge[i] << "   "<<((sum - DirectionJudge[i])/ (DirectionJudge.size()-1))<<endl;
                        //cout << "偏差:" << abs( (variance/temp_variance) *DirectionJudge[i] / ((sum - DirectionJudge[i])/ (DirectionJudge.size()-1)))<< endl;
                        continue;
                    }
                }
                else if(filter_mode == 2){
                // filter 2: 均值方差各自设置阈值
                    //cout << " "  <<abs(variance/temp_variance);
                    //cout << abs(DirectionJudge[i] / ((sum - DirectionJudge[i])/ (DirectionJudge.size()-1)))<< endl;
                    if( abs(variance/temp_variance) > 1.03)//如果影响方差过大
                    {
                        //cout << DirectionJudge[i] << "   "<<((sum - DirectionJudge[i])/ (DirectionJudge.size()-1))<<endl;
                        if (abs(DirectionJudge[i] / ((sum - DirectionJudge[i])/ (DirectionJudge.size()-1))) > 1.5 )//如果偏离均值过大
                        {
                            //cout << "方差偏差:" << abs(variance/temp_variance) << " 均值偏差:" << abs(DirectionJudge[i] / ((sum - DirectionJudge[i])/ (DirectionJudge.size()-1)))<< endl;
                            continue;
                        }
                    }
                }
                points[1][k++] = points[1][i];
                d_current_position = Dxy[i] + d_current_position;
            }
            points[1].resize(k);
            
            //cout << "d_current_position = " << d_current_position.x/(double)Dxy.size() << d_current_position.y/(double)Dxy.size() << endl;
            if(points[1].size() != 0)// avoid the divisor get 0 causing interruption
            {
                current_position.x += d_current_position.x/(double)points[1].size();
                current_position.y += d_current_position.y/(double)points[1].size();    
                cout << current_position << endl;
            }
            else
            {   
            }
        }
       
        // 如果用到了鼠标点击增添光流r
        if (addRemovePt && points[1].size() < (size_t)MAX_COUNT)
        {
            //在新增加的点周围迭代找个新角点
            vector<Point2f> tmp;
            tmp.push_back(point);
            cornerSubPix(gray, tmp, winSize,cvSize(-1, -1), termcrit);
            //添加新找到的角点
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
        if(points[1].size()> 0 && points[1].size() < minNumber_corner) //角点剩余少于4个的时候,重新寻找一波角点
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
