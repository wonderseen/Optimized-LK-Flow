/**
 * Author: WonderSeen
 * 2018.3 .13
 * Xiamen University, China 
 * Lisence: WonderSeen
 */
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#pragma once //只要在头文件的最开始加入这条杂注，就能够保证头文件只被编译一次。

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

#define SIFTPoint vector<Point2d>

namespace wonderseen 
{   
    typedef const Mat& InputMat;
    typedef const Mat& OutputMat;
    class CV_EXPORTS Corner
    {
        // methods
        public:
            Corner(){};
            Corner(const size_t num) : corner_State( true ){ this->corners.resize(num); };
            ~Corner(){};
            
            /* two corner add_up */
            friend double operator+( const vector<Point> & b);
            vector<Point> operator+(const vector<Point> & b)
            {
                corners.insert(corners.end(), b.begin(), b.end());
                return corners;
            }

            // inline 相当于提前静态编译,宏替换函数
            inline bool goodSIFTToTrack( InputMat image1, InputMat image2, SIFTPoint &fp,
                                int maxCorners, double qualityLevel, double minDistance,
                                int blockSize = 3, double k = 0.04 )
            /* compute the SIFT-Harris corners in pyramid images */
            {
                Mat image = image2/2. + image1/2.;
                if(image.type() != CV_8UC1)
                {
                    cout << image.type() << endl;
                    cvtColor(image, image, COLOR_BGR2GRAY);
                }

                // gaussion strength
                double alpha = 1.2; // 1.2**4 = 2
                
                // pyramid image
                Mat diff1, diff2, diff3; 
                Mat lever1, lever2, lever3, lever4; 

                image.convertTo(image,CV_32FC1);
                cv::resize(image, image, Size(), 0.5, 0.5, INTER_NEAREST);
                
                GaussianBlur(image, lever1, Size(blockSize,blockSize), alpha, alpha);
                GaussianBlur(lever1, lever2, Size(blockSize,blockSize), alpha, alpha);
                GaussianBlur(lever2, lever3, Size(blockSize,blockSize), alpha, alpha);
                GaussianBlur(lever3, lever4, Size(blockSize,blockSize), alpha, alpha);
                
                // gau-diff
                diff1 = lever2 - lever1;
                diff2 = lever3 - lever2;
                diff3 = lever4 - lever3;

                // get fp and draw
                genFeaturePoint(diff1, fp, 10.);
                Mat src(image.size(),CV_8UC3, Scalar(0,0,0));
                for(int i=0; i<fp.size(); i++)
                {
                    circle(src, fp[i], 2, Scalar(0,255,255), -1, 8);
                }

                imshow("diff1", diff1);
                imshow("diff2", diff2);
                imshow("diff3", diff3);
                imshow("raw_gray", src);
                waitKey(10);

                if(corners.size() != 0)
                {
                    corner_State = true;
                    return true;
                }
                else
                    return false;
            }


            inline void genFeaturePoint(InputMat inputmat,  SIFTPoint &fp, float threshold=1.f)
            {
                /**
                 * threshold  -- must bigger than 0. 
                 *               if I(j,i)-I(x,y) bigger than threshold, judge (x,y) as non-fp
                 *               
                 * fp         -- feature points
                 * blockW/blockH/stepx/stepy
                 *            -- compare windows parameter
                */
                int imageW = inputmat.cols, imageH = inputmat.rows;
                int blockW = 1, blockH = 1; // must be odd, half of the blocksize
                int stepx = 2, stepy = 2;
                for(int x=blockW; x<imageW-blockW; x+=stepx)
                    for(int y=blockH; y<imageH-blockH; y+=stepy)
                    {
                        bool goodFeature = true;
                        float tmp = inputmat.at<float>(y,x);
                        for(int i=-blockW; i<=blockW; i++)
                            for(int j=-blockH; j<=blockH; j++)
                            {
                                // judge whether the temp is the smallest of a small region
                                if(inputmat.at<float>(j,i)-tmp < threshold)
                                {
                                    goodFeature = false;
                                    break;
                                }
                            }
                        if(goodFeature)
                        {
                            fp.push_back(cv::Point2d(x,y));
                        }
                    }
                if(fp.size() == 0)
                {
                    cout << "no feature points!" << endl;
                    return;
                }
            }


            const void conv(const InputMat mask, const InputMat raw, OutputMat output, size_t step=1)
            /**
             * convolution on a map.
             * Not done yet.
            */
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


            inline void transPreToNew(Mat& image, Mat& transX, Mat& transY)
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


            inline void computeXYvector(Mat& diff, 
                                 Mat& outdx, Mat& outdy, 
                                 Mat& transX, Mat& transY )
            {
                // calculate the top pymarid flow XY vector
                float dx=0., dy=0.;
                for(int i=0; i<diff.cols-1; i++)
                    for(int j=0; j<diff.rows-1; j++)
                    {
                        // cal the gradient of x_coordinate and y_coorinate
                        outdx.at<float>(j,i) = float(diff.at<char>(j,i+1) - diff.at<char>(j, i-1));
                        //outdx.at<float>(j,i) += float(diff.at<char>(j+1,i+1) - diff.at<char>(j-1, i-1));
                        //outdx.at<float>(j,i) += float(diff.at<char>(j-1,i+1) - diff.at<char>(j+1, i-1));

                        outdy.at<float>(j,i) = float(diff.at<char>(j,i+1) - diff.at<char>(j, i-1));
                        //outdy.at<float>(j,i) += float(diff.at<char>(j+1,i+1) - diff.at<char>(j-1, i-1));
                        //outdy.at<float>(j,i) += float(diff.at<char>(j-1,i+1) - diff.at<char>(j+1, i-1));
                        dx += outdx.at<float>(j,i);
                        dy += outdy.at<float>(j,i);
                    }

                // calculate eigen vectors(特征向量) and eigen values(特征值) as global flow
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
                        drawArrow(dis, Point(i,j), 
                                    Point(i+flowx.at<float>(j,i)/10, j+flowy.at<float>(j,i)/10),
                                    10, 10, Scalar(0, 255, 255), 1, 1);
                    }
                imshow("arrow", dis);
                waitKey(10);
            }

            void drawArrow(Mat& img, Point pStart, Point pEnd, int len, int alpha, Scalar color, int thickness, int lineType)
            /**
             * Draw an arrow.
             * Step 1:
             *      draw the vector of (start -> end)
             * Step 2:
             *      draw the 2-side header at the defined end point
            */
            {
                const double PI = 3.1415926;
                Point arrow;
                // step 1
                double angle = atan2((double)(pStart.y - pEnd.y), (double)(pStart.x - pEnd.x));  
                line(img, pStart, pEnd, color, thickness, lineType);   
                // step 2
                arrow.x = pEnd.x + len * cos(angle + PI * alpha / 180);
                arrow.y = pEnd.y + len * sin(angle + PI * alpha / 180);
                line(img, pEnd, arrow, color, thickness, lineType);
                arrow.x = pEnd.x + len * cos(angle - PI * alpha / 180);
                arrow.y = pEnd.y + len * sin(angle - PI * alpha / 180);
                line(img, pEnd, arrow, color, thickness, lineType);
            }

            #ifdef DEGUG
            virtual void ShowState()// virtual是实现多态动态绑定的基础,通过指针可以访问到子类的同名方法
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
