/**
 * Owner: WonderSeen
 * 2018.3.13
 * Xiamen University, China 
 * Lisence: WonderSeen
 */
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#pragma once  

#define DEBUG
#ifdef DEGUG
#   include <iostream>
#endif

#ifdef DEBUG
#   if(defined WIN32 || defined _WIN32 || defined WINCE)
#       define WDX_EXPORTS __declspec(dllexport)
#   else
#       define WDX_EXPORTS __attribute__ ((visibility ("default")))//(selectany)
#   endif
#endif

using namespace cv;
using namespace std;



namespace wonderseen 
{
    class CV_EXPORTS Corner
    {
        // methods
        public:
            Corner(){};
            Corner(const size_t num) : corner_State( true ){ this->corners.resize(num); };
            ~Corner(){};

            typedef const _InputArray& InputArray;
            bool goodFeaturesToTrack(   Mat& image1, Mat& image2, int maxCorners, double qualityLevel, double minDistance,
                                        int blockSize = 3, double k = 0.04 )
            /* compute the harris corners in pyramid images */
            {
                Mat image = image2-image1;
                if(image.type() != CV_8UC1)
                {
                    cout << image.type() << endl;
                    cvtColor(image, image, COLOR_BGR2GRAY);
                }

                // conv windows
                double alpha = 1.5;
                
                // pyramid image
                // diff1
                Mat lever1, diff1, lever2, diff2, lever3, diff3; 
                Mat lever11, lever22, lever33; 
                
                lever1 = image;
                GaussianBlur(image, lever1, Size(blockSize,blockSize), alpha, alpha);
                diff1 = lever1 - lever11;

                // diff2
                cv::resize(lever1, lever2, Size(), 0.25, 0.25);
                GaussianBlur(lever2, lever22, Size(blockSize,blockSize), alpha, alpha);
                diff2 = lever2 - lever22;

                // diff3
                cv::resize(lever2, lever3, Size(), 0.5, 0.5);
                GaussianBlur(lever3, lever33, Size(blockSize,blockSize), alpha, alpha);
                diff3 = lever3 - lever33;

                Mat dx1(lever1.size(), CV_32FC1, Scalar(0.0) );
                Mat dx2(lever2.size(), CV_32FC1, Scalar(0.0) );
                Mat dx3(lever3.size(), CV_32FC1, Scalar(0.0) );
                Mat dy1(lever1.size(), CV_32FC1, Scalar(0.0) );
                Mat dy2(lever2.size(), CV_32FC1, Scalar(0.0) );
                Mat dy3(lever3.size(), CV_32FC1, Scalar(0.0) );
                Mat transX1(lever1.size(), CV_32FC1, Scalar(0.0) );
                Mat transX2(lever2.size(), CV_32FC1, Scalar(0.0) );
                Mat transX3(lever3.size(), CV_32FC1, Scalar(0.0) );
                Mat transY1(lever1.size(), CV_32FC1, Scalar(0.0) );
                Mat transY2(lever2.size(), CV_32FC1, Scalar(0.0) );
                Mat transY3(lever3.size(), CV_32FC1, Scalar(0.0) );

                //computeXYvector(lever1, dx1, dy1, transX1, transY1);
                //computeXYvector(lever2, dx2, dy2, transX2, transY2);
                computeXYvector(lever3, dx3, dy3, transX3, transY3);
                
                cv::resize(image1, image1, Size(), 0.125, 0.125);
                cv::resize(image2, image2, Size(), 0.125, 0.125);
                transPreToNew(image1, transX3, transY3);
                cv::resize(image1, image1, Size(), 8, 8);
                cv::resize(image2, image2, Size(), 8, 8);
                imshow("preTransform", image1);
                imshow("Now", image2);
                waitKey(10000);

                cv::resize(dx1, dx1, Size(), 1, 1);
                cv::resize(dy1, dy1, Size(), 1, 1);
                cv::resize(dx2, dx2, Size(), 4, 4);
                cv::resize(dy2, dy2, Size(), 4, 4);
                cv::resize(dx3, dx3, Size(), 8, 8);
                cv::resize(dy3, dy3, Size(), 8, 8);

                Mat resultx = dx3;
                Mat resulty = dy3;

                Mat resultxy;
                magnitude(resultx, resulty, resultxy);
                resultxy = (resultxy > 100 );
                DisplayFlow(resultx, resulty, 200.);

                Mat Erodeelement = getStructuringElement(MORPH_RECT, Size(4,4));
                erode(resultxy, resultxy, Erodeelement);

                //imshow("resultx", resultx);
                //imshow("resulty", resulty);
                //imshow("resultxy", resultxy);
                //waitKey(10);

                Mat harris_l3;

                if(corners.size() != 0)
                {
                    corner_State = true;
                    return true;
                }
                else
                    return false;
            }

            void conv(Mat& mask, Mat& raw, Mat& output, size_t step=1)
            {
                int imageW = raw.cols, imageH = raw.rows;
                int blockW = mask.cols, blockH = mask.rows;
                for(int x=0; x<imageW; x+=step)
                    for(int y=0; y<imageH; y+=step)
                        for(int i=0; i<blockW; i++)
                            for(int j=0; j<blockH; j++)
                            {
                                
                            }
            }


            void transPreToNew(Mat& image, Mat& transX, Mat& transY)
            {
                int imageW = image.cols-1;
                int imageH = image.rows-1;
                for(int i=0; i<imageW; i++)
                    for(int j=0; j<imageH; j++)
                    {   
                        int NewX = transX.at<float>(Point(i,j)) < imageW ? transX.at<char>(Point(i,j)) : imageW;
                        int NewY = transY.at<float>(Point(i,j)) < imageH ? transY.at<char>(Point(i,j)) : imageH;
                        image.at<char>(Point(i,j)) = image.at<char>(Point(NewX, NewY));
                    }
            }




            void computeXYvector(Mat& diff, 
                                 Mat& outdx, Mat& outdy, 
                                 Mat& transX, Mat& transY )
            {
                // calculate the top pymarid flow XY vector
                float dx=0., dy=0.;
                for(int i=0; i<diff.cols-1; i++)
                    for(int j=0; j<diff.rows-1; j++)
                    {
                        // 计算差分图像的x和y梯度
                        outdx.at<float>(j,i) = float(diff.at<char>(j,i+1) - diff.at<char>(j, i-1));
                        //outdx.at<float>(j,i) += float(diff.at<char>(j+1,i+1) - diff.at<char>(j-1, i-1));
                        //outdx.at<float>(j,i) += float(diff.at<char>(j-1,i+1) - diff.at<char>(j+1, i-1));

                        outdy.at<float>(j,i) = float(diff.at<char>(j,i+1) - diff.at<char>(j, i-1));
                        //outdy.at<float>(j,i) += float(diff.at<char>(j+1,i+1) - diff.at<char>(j-1, i-1));
                        //outdy.at<float>(j,i) += float(diff.at<char>(j-1,i+1) - diff.at<char>(j+1, i-1));
                        dx += outdx.at<float>(j,i);
                        dy += outdy.at<float>(j,i);
                    }

                // 根据梯度矩阵,计算转移特征向量和特征值,作为位移向量
                double myArray[2][2] = {
                    dx*dx, dx*dy,
                    dx*dy, dy*dy,
                };

                cv::Mat myMat = cv::Mat(2, 2, CV_64FC1, myArray);
                Mat eValuesMat, eVectorsMat;
                eigen(myMat, eValuesMat, eVectorsMat);
                double max, min;
                minMaxIdx(eValuesMat, &min, &max);
                
                // 下面把 xy方向的矢量直接应用到原图上, transdX上每个位置的值,就是该位置在下一帧到达的新的x坐标值
                if(max < 0.001)
                {
                    for(int i=0; i<diff.cols-1; i++)
                        for(int j=0; j<diff.rows-1; j++)
                        {
                            outdx.at<float>(j,i) = i;
                            outdy.at<float>(j,i) = j;
                        }
                }
                else 
                {
                    for(int i=0; i<diff.cols-1; i++)
                        for(int j=0; j<diff.rows-1; j++)
                        {
                            cout << transX.at<float>(j,i) << endl;
                            transX.at<float>(j,i) = i +
                             outdy.at<float>(j,i)/dy * (eVectorsMat.at<float>(0,0)/eVectorsMat.at<float>(0,1) ) * eValuesMat.at<float>(0,0);
                            transY.at<float>(j,i) = i +
                             outdx.at<float>(j,i)/dx * (eVectorsMat.at<float>(1,0)/eVectorsMat.at<float>(1,1) ) * eValuesMat.at<float>(1,1);
                        }
                }
            }

            void DisplayFlow(Mat& flowx, Mat& flowy, float threshold)
            {
                Mat dis(flowx.size(), CV_8UC3, Scalar(0,0,0));
                for(int i=0; i<flowx.cols-1; i++)
                    for(int j=0; j<flowx.rows-1; j++)
                    {
                        if( abs(flowx.at<float>(j,i)) > threshold || abs(flowy.at<float>(j,i)) > threshold )
                        drawArrow(dis, Point(i,j), \
                                    Point(i+flowx.at<float>(j,i)/10, j+flowy.at<float>(j,i)/10),\ 
                                    10, 10, Scalar(0, 255, 255), 1, 1);
                    }
                imshow("arrow", dis);
                waitKey(10);
            }

            void drawArrow(Mat& img, Point pStart, Point pEnd, int len, int alpha, Scalar color, int thickness, int lineType)
            {
                const double PI = 3.1415926;
                Point arrow;
                //计算 θ 角（最简单的一种情况在下面图示中已经展示，关键在于 atan2 函数，详情见下面）   
                double angle = atan2((double)(pStart.y - pEnd.y), (double)(pStart.x - pEnd.x));  
                line(img, pStart, pEnd, color, thickness, lineType);   
                //计算箭角边的另一端的端点位置（上面的还是下面的要看箭头的指向，也就是pStart和pEnd的位置） 
                arrow.x = pEnd.x + len * cos(angle + PI * alpha / 180);
                arrow.y = pEnd.y + len * sin(angle + PI * alpha / 180);
                line(img, pEnd, arrow, color, thickness, lineType);
                arrow.x = pEnd.x + len * cos(angle - PI * alpha / 180);
                arrow.y = pEnd.y + len * sin(angle - PI * alpha / 180);
                line(img, pEnd, arrow, color, thickness, lineType);
            }

            #ifdef DEGUG
            virtual void ShowState()
            {
                cout << "corner_State is " << flag << endl;
            }
            #endif

            
        // variables
        public:
            bool corner_State;
            enum CornerType{ HARRIS_CORNER=0, SHI_TOMASI_CORNER=1 };
            vector<Point> corners;
    };
}