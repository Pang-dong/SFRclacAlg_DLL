// SFRClacAlg.cpp : 定义 DLL 应用程序的导出函数。
//

#include "stdafx.h"
#include "SFRClacAlg.h"


#include "middle.h"

// 这是已导出类的构造函数。
// 有关类定义的信息，请参阅 SFRClacAlg.h
using namespace cv;
CSFRClacAlg::CSFRClacAlg()
{
}
CSFRClacAlg::~CSFRClacAlg()
{

}
int CSFRClacAlg::EdgeSFR_HB(unsigned char* img, int imgW, int imgH, double cyclepixel, double& sfrVal)
{
	//确保边缘从左到右是暗到亮或亮到暗  目的是标准化边缘的方向
	if (img == nullptr)
		return Image_BufferError;
	//如果是整个图像的话需要传ROI
	//Mat src(height, width, CV_8UC3, bufrgb, width * 3);
	//Mat matROI(src, Rect(stROI.left, stROI.top, stROI.right - stROI.left, stROI.bottom - stROI.top));
	//Mat img_src = img.clone();
	Mat img_src(imgH, imgW, CV_8UC1, img);
	int img_height = imgH;
	int img_width = imgW;
	
	cv::Rect roiA(0, 0, 5, img_height);
	cv::Rect roiB(img_width - 6, 0, 6, img_height);
	cv::Mat A = img_src(roiA);
	cv::Mat B = img_src(roiB);
	double tleft = cv::sum(A).val[0];
	double tright = cv::sum(B).val[0];
	cv::Mat fil1 = (cv::Mat_<double>(1, 2) << 0.5, -0.5);
	cv::Mat fil2 = (cv::Mat_<double>(1, 3) << 0.5, 0, -0.5);
	//通过比较图像最左侧5列和最右侧6列的像素和，判断边缘的方向（从亮到暗还是从暗到亮），并相应地调整导数滤波器的系数。
	//利用上述已得的4倍超采样的边缘扩散序列ESF，卷积一个有限差分滤波器[-1 / 2, 0, 1 / 2],
	//得出线扩散序列LSF, 这实际等同于对相邻2点ESF进行均值滤波后，再微分
	if (tleft > tright)
	{
		fil1 = (cv::Mat_<double>(1, 2) << -0.5, 0.5);
		fil2 = (cv::Mat_<double>(1, 3) << -0.5, 0, 0.5);
	}
	double test = abs((tleft - tright) / (tleft + tright));
	// 边缘质量检查 
	//如果边缘的对比度太低（(tleft - tright) / (tleft + tright) 的绝对值小于0.3），则认为边缘质量不佳，返回-3
	if (abs(test) < 0.3)
	{
		return Image_QualityError;
	}
	cv::Mat filme = cv::Mat::zeros(1, 3, CV_64FC1);
	double slout = 0;

	//汉明窗
	cv::Mat win1 = AlgMiddle::ahamming(img_width, (img_width + 1) / 2.0);//
															  //SaveMatData(win1, "ahamming.xls");
															  //求一阶导数 使用一阶导数滤波器 fil1 计算每行的梯度。
	cv::Mat c = AlgMiddle::deriv1(img_src, fil1);
	//SaveMatData(c, "deriv1.xls");
	//SaveMatData(c, L"deriv1.xls",CV_64FC1);
	//计算质心 对每行的梯度数据应用汉明窗后，计算质心位置，得到边缘的粗略位置
	cv::Mat loc = cv::Mat::zeros(1, img_height, CV_64FC1);
	auto *ptr_loc = loc.ptr<double>();//存储质心位置的集合


	for (int i = 0; i < img_height; i++)
	{
		cv::Mat temp = c.row(i).t();
		temp = temp.mul(win1);
		double centroid_temp = AlgMiddle::centroid(temp) - 0.5;
		ptr_loc[i] = centroid_temp;
	}

	//SaveMatData(loc, L"centroid.xls", CV_64FC1);
	//曲线拟合
	std::vector<cv::Point2d> fit_points;
	for (int i = 0; i < img_height; i++)
	{
		//fit_points.emplace_back(i, ptr_loc[i]);
		fit_points.push_back(cv::Point2d(i, ptr_loc[i]));

	}
	//对这些质心位置进行线性拟合，得到边缘的直线方程 y = kx + b。
	cv::Mat fitme = AlgMiddle::polyfit(fit_points, 1);//这里和Matlab位置不一样
	cv::Mat place = cv::Mat::zeros(img_height, 1, CV_64FC1);
	auto *ptr_place = place.ptr<double>();
	double fitme_a = fitme.at<double>(1, 0);
	double fitme_b = fitme.at<double>(0, 0);
	for (int i = 0; i < img_height; i++)
	{
		ptr_place[i] = fitme_a * (i + 1) + fitme_b;
		cv::Mat win2 = AlgMiddle::ahamming(img_width, ptr_place[i]);

		cv::Mat temp = c.row(i).t();
		temp = temp.mul(win2);
		double centroid_temp = AlgMiddle::centroid(temp);
		ptr_loc[i] = centroid_temp;
	}
	fit_points.clear();
	for (int i = 0; i < img_height; i++)
	{
		//fit_points.emplace_back(i, ptr_loc[i]);
		fit_points.push_back(cv::Point2d(i, ptr_loc[i]));
	}
	fitme = AlgMiddle::polyfit(fit_points, 1);//这里和Matlab位置不一样

								   //Limit number of lines to integer
								   //对应oldflag=0

	double kval = fitme.at<double>(1, 0);
	double bval = fitme.at<double>(0, 0);

	int nlin1 = img_height * abs(fitme.at<double>(1, 0));
	nlin1 = round(nlin1 / abs(fitme.at<double>(1, 0)));
	cv::Rect interger_roi(0, 0, img_width, nlin1);
	cv::Mat img_roi_new = img_src(interger_roi);

	//检查边缘的倾斜角度是否在合理范围内（1°到20°）。如果角度太小或太大，可能影响超采样的精度，返回-1。
	double vslope = abs(fitme.at<double>(1, 0));
	double slope_deg = 180 * atan(abs(vslope)) / CV_PI;
	if ((slope_deg < 1/*3.5*/) || (slope_deg >20))
	{
		//std::cout << "High slope warning : " << slope_deg << " degrees\n";
		return -1;//返回-1值 表示异常
	}

	double del2 = 0;
	//Correct sampling inverval for sampling parallel to edge
	//对应oldflag=0
	double del = 1;
	int nbin = 4;//设置超采样因子 nbin = 4，即每个像素采样4次。
	double delfac = cosf(atan(vslope));
	del = del * delfac;
	del2 = del / nbin;

	int nn = img_width * nbin;
	cv::Mat mtf = cv::Mat::zeros(nn, 1, CV_64FC1);
	int nn2 = nn / 2 + 1;

	//Derivative correction
	cv::Mat dcorr = AlgMiddle::fir2fix(nn2, 3);

	cv::Mat freq = cv::Mat::zeros(nn, 1, CV_64FC1);
	auto ptr_freq = freq.ptr<double>();
	for (int i = 0; i < nn; i++)
	{
		ptr_freq[i] = nbin * i / (del*nn);
	}

	int freqlim = 1;
	if (1 == nbin)
	{
		freqlim = 2;
	}

	int nn2out = round(nn2*freqlim / 2.0);

	double nfreq = nn / (2.0*del*nn); // half - sampling frequency
	cv::Mat win = AlgMiddle::ahamming(nbin*img_width, (nbin*img_width + 1) / 2.0);

	//Large SFR loop for each color record
	cv::Mat esf = cv::Mat::zeros(nn, 1, CV_64FC1);
	// project and bin data in 4x sampled array
	cv::Mat point = AlgMiddle::project(img_roi_new, loc.at<double>(0, 0), fitme.at<double>(1, 0), nbin);
	esf = point;
	//	SaveMatData(esf, L"esf.xls", CV_64FC1);
	//compute first derivative via FIR(1x3) filter fil
	//对ESF数据应用导数滤波器 fil2（3点滤波器）得到LSF（线扩散函数）
	c = AlgMiddle::deriv1(point.t(), fil2);// 使用3点滤波器计算导数
	c = c.t();

	cv::Mat psf = c;// LSF
	double mid = AlgMiddle::centroid(c);
	//SaveMatData(psf, L"lsf.xls", CV_64FC1);
	//计算LSF的质心，并对其进行中心化（将峰值移到中间）
	cv::Mat temp = AlgMiddle::cent(c, round(mid));// 中心化
	c = temp;
	// apply window(symmetric Hamming)
	//对LSF数据应用汉明窗，减少频谱泄漏。
	c = win.mul(c);// 加汉明窗

				   //Transform, scale and correct for FIR filter response
				   //temp = abs(fft(c, nn));
	cv::dft(c, temp, cv::DFT_COMPLEX_OUTPUT);// 傅里叶变换
	temp = AlgMiddle::complex_abs(temp);// 取模
							 //SaveMatData(temp, L"complex_abs.xls", CV_64FC1);
	cv::Rect roi_temp(0, 0, 1, nn2);
	cv::Mat temp1 = temp(roi_temp).clone();
	temp1 = temp1 / temp1.at<double>(0, 0);
	cv::Mat mtf_roi = mtf(roi_temp);
	temp1.copyTo(mtf_roi);
	//对应oldflag=0
	//应用导数滤波器的频率响应校正（dcorr），补偿滤波器对MTF的影响
	mtf_roi = mtf_roi.mul(dcorr); // 滤波器响应校正
								  //SaveMatData(mtf_roi, L"mtf_roi.xls", CV_64FC1);
	std::vector<cv::Point2d> dat;
	auto ptr_mtf = mtf.ptr<double>();
	for (int i = 0; i < nn2out; i++)
	{
		//dat.emplace_back(ptr_freq[i], ptr_mtf[i]);	
		dat.push_back(cv::Point2d(ptr_freq[i], ptr_mtf[i]));
	}
	cv::Mat dat_mat(dat);//仅用于显示

						 //Sampling efficiency
						 //cv::Mat val = (cv::Mat_<double>(1, 2) << 0.1, 0.5);
	cv::Mat val = (cv::Mat_<double>(1, 1) << 0.5);//只关注MTF50

	cv::Mat eff, freqval, sfrval;
	//sampeff(dat_mat, val, del, eff, freqval, sfrval);
	// 	float freqval_mat, sfrval_mat;
	// 	auto val_mat = freqval.ptr<double>();
	// 	freqval_mat = val_mat[0];
	// 	val_mat = sfrval.ptr<double>();
	// 	sfrval_mat = val_mat[0];
	float /*fre = 0.25,*/ freqval1 = 0, sfrval1 = 0;
	AlgMiddle::sampesfr(dat_mat, cyclepixel, del, eff, freqval1, sfrval1);
	// Plot SFRs on same axes
	//后面就是绘图和保存结果了
	// 
	//	SaveMatData(dat_mat, L"sfrdat_mat.xls", CV_64FC1);
	std::vector<cv::Mat> channels(3);
	cv::split(dat_mat, channels);

	cv::Mat dat_output;
	cv::hconcat(channels[0], channels[1], dat_output);
	auto ptr_1 = channels[0].ptr<double>();
	auto ptr_2 = channels[1].ptr<double>();
	//打印输出
	for (int i = 0; i < dat_output.rows; i++)
	{
		std::cout << ptr_1[i] << "," << ptr_2[i] << std::endl;
	}
	//SaveMatData(dat_output, L"sfr_output.xls", CV_64FC1);
	if (sfrval1 < 0 || sfrval1 > 1)
		return SFR_VALUEERROR;
	sfrVal = sfrval1;
	return SFRClac_ErrorCode::Clac_Success;
}