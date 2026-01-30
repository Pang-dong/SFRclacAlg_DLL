#ifndef __MIDDLE_H__
#define __MIDDLE_H__
#include "opencv2/opencv.hpp"
#include "Spline.h"
using namespace cv;
class AlgMiddle
{
public: 
	AlgMiddle(){}
	~AlgMiddle(){}
	static Mat ahamming(int n, double mid);
   static Mat deriv1(const cv::Mat &input, cv::Mat fil);
	static double centroid(const cv::Mat &input);
	static Mat polyfit(std::vector<cv::Point2d>& in_point, int n);
	static Mat fir2fix(int n, int m);
	static Mat project(const cv::Mat &bb, double loc, double slope, int fac);
	static Mat cent(const cv::Mat &a, int center);
	static cv::Mat complex_abs(const cv::Mat &input);
	static void sampesfr(const cv::Mat &dat, float fre,
		double del, cv::Mat &eff, float &freqval, float &sfrval);
};

#endif  //__MIDDLE_H__
