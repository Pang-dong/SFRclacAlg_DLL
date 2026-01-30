#include "middle.h"
#include <vector>
cv::Mat AlgMiddle::ahamming(int n, double mid)
{
	cv::Mat hammingData = cv::Mat::zeros(n, 1, CV_64FC1);
	double wid1 = mid - 1;
	double wid2 = n - mid;
	double wid = max(wid1, wid2);
	const double pie = 3.141592653589793;
	auto *ptr = hammingData.ptr<double>();
	for (int i = 0; i < n; i++)
	{
		double arg = i - mid + 1;
		ptr[i] = cosf(pie*arg / wid);
	}
	hammingData = hammingData * 0.46 + 0.54;

	return hammingData;
}

cv::Mat AlgMiddle::deriv1(const cv::Mat &input, cv::Mat fil)
{
	int nlin = input.rows;
	int npix = input.cols;
	int nn = fil.cols;
	cv::Mat calc_mat;
	input.convertTo(calc_mat, fil.type());
	cv::Mat b = cv::Mat::zeros(nlin, npix, calc_mat.type());
	for (int i = 0; i < nlin; i++)
	{
		if (3 == nn)
		{
			cv::Mat row_mat = calc_mat.row(i);
			cv::Mat temp;
			cv::filter2D(row_mat, temp, fil.type(), fil, cv::Point(-1, -1), 0, cv::BORDER_ISOLATED);
			temp = -temp;//与matlab结果相差一个负号
			temp.at<double>(0, 0) = temp.at<double>(0, 1);
			cv::Mat temp_zero = cv::Mat::zeros(1, 1, temp.type());
			cv::Mat temp_1;
			cv::hconcat(temp_zero, temp, temp_1);
			cv::Rect roi_rect(0, 0, npix, 1);
			temp = temp_1(roi_rect).clone();
			cv::Mat row_b = b.row(i);
			temp.copyTo(row_b);
		}
		else if (2 == nn)
		{
			cv::Mat row_mat = calc_mat.row(i);
			cv::Mat temp;
			cv::filter2D(row_mat, temp, fil.type(), fil);
			temp = -temp;
			temp.at<double>(0, 0) = temp.at<double>(0, 1);
			cv::Mat row_b = b.row(i);
			temp.copyTo(row_b);
		}
	}
	return b;
}

double AlgMiddle::centroid(const cv::Mat &input)
{
	int n = input.rows;
	cv::Mat n_mat = cv::Mat::zeros(1, n, input.type());
	auto *ptr = n_mat.ptr<double>();
	for (int i = 0; i < n; i++)
	{
		ptr[i] = i + 1;
	}

	double sumx = cv::sum(input).val[0];
	if (sumx < 1e-4)
	{
		return 0;
	}
	else
	{
		double loc = cv::sum(n_mat*input).val[0] / sumx;
		return loc;//-0.5 shift for FIR phase
	}
}
Mat AlgMiddle::polyfit(std::vector<cv::Point2d>& in_point, int n)
{
	int size = in_point.size();
	//所求未知数个数
	int x_num = n + 1;
	//构造矩阵U和Y
	Mat mat_u(size, x_num, CV_64F);
	Mat mat_y(size, 1, CV_64F);

	for (int i = 0; i < mat_u.rows; ++i)
		for (int j = 0; j < mat_u.cols; ++j)
		{
			mat_u.at<double>(i, j) = pow(in_point[i].x, j);
		}

	for (int i = 0; i < mat_y.rows; ++i)
	{
		mat_y.at<double>(i, 0) = in_point[i].y;
	}

	//矩阵运算，获得系数矩阵K
	Mat mat_k(x_num, 1, CV_64F);
	mat_k = (mat_u.t()*mat_u).inv()*mat_u.t()*mat_y;
	//std::cout << mat_k << std::endl;
	return mat_k;
}
cv::Mat AlgMiddle::fir2fix(int n, int m)
{
	cv::Mat correct = cv::Mat::ones(n, 1, CV_64FC1);

	m = m - 1;
	int scale = 1;
	auto ptr = correct.ptr<double>();
	for (int i = 1; i < n; i++)
	{
		ptr[i] = abs(CV_PI*(i + 1)*m / (2 * (n + 1))) / sinf(CV_PI*(i + 1)*m / (2 * (n + 1)));
		ptr[i] = 1 + scale * (ptr[i] - 1);
		if (ptr[i] > 10) //limiting the correction to the range[1, 10]
		{
			ptr[i] = 10;
		}
	}
	return correct;
}
cv::Mat AlgMiddle::project(const cv::Mat &bb, double loc, double slope, int fac)
{
	int nlin = bb.rows;
	int npix = bb.cols;
	int big = 0;
	int nn = npix * fac;
	// smoothing window
	cv::Mat win = ahamming(nn, fac*loc);

	slope = 1 / slope;

	int offset = round(fac*(0 - (nlin - 1) / slope));
	int del = abs(offset);
	if (offset > 0)
	{
		offset = 0;
	}

	cv::Mat barray = cv::Mat::zeros(2, nn + del + 100, CV_64FC1);
	auto *ptr_barray_1 = barray.ptr<double>(0);
	auto *ptr_barray_2 = barray.ptr<double>(1);

	for (int n = 0; n < npix; n++)
	{
		for (int m = 0; m < nlin; m++)
		{
			int x = n;
			int y = m;
			int ling = ceil((x - y / slope)*fac) + 1 - offset;
			ling = ling - 1;//与matalb的坐标差别
			ptr_barray_1[ling] = ptr_barray_1[ling] + 1;
			ptr_barray_2[ling] = ptr_barray_2[ling] + bb.at<uchar>(m, n);
		}
	}

	int start = 1 + round(0.5*del);
	int nz = 0;
	int status = 0;
	for (int i = start - 1; i < start + nn; i++)
	{
		if (0 == ptr_barray_1[i])
		{
			nz++;
			status = 0;
			if (1 == i)
			{
				ptr_barray_1[i] = ptr_barray_1[i + 1];
			}
			else
			{
				ptr_barray_1[i] = (ptr_barray_1[i - 1] + ptr_barray_2[i + 1]) / 2.0;
			}
		}
	}

	cv::Mat point = cv::Mat::zeros(nn, 1, CV_64FC1);
	auto *ptr_point = point.ptr<double>();
	for (int i = 0; i < nn; i++)
	{
		ptr_point[i] = ptr_barray_2[i + start - 1] / ptr_barray_1[i + start - 1];
	}

	return point;
}
Mat AlgMiddle::cent(const cv::Mat &a, int center)
{
	int n = a.rows;
	cv::Mat b = cv::Mat::zeros(n, 1, a.type());
	int mid = round((n + 1) / 2.0);
	int del = round(center - mid);
	auto ptr_a = a.ptr<double>();
	auto ptr_b = b.ptr<double>();
	if (del > 0)
	{
		for (int i = 0; i < n - del; i++)
		{
			ptr_b[i] = ptr_a[i + del];
		}
	}
	else
	{
		for (int i = -del; i < n; i++)
		{
			ptr_b[i] = ptr_a[i + del];
		}
	}

	return b;
}
cv::Mat AlgMiddle::complex_abs(const cv::Mat &input)
{
	std::vector<cv::Mat> channels(3);
	channels.clear();
	cv::split(input, channels);
	auto ptr_1 = channels[0].ptr<double>();
	auto ptr_2 = channels[1].ptr<double>();
	cv::Mat output = cv::Mat::zeros(input.rows, 1, CV_64FC1);
	auto ptr_output = output.ptr<double>();
	for (int i = 0; i < input.rows; i++)
	{
		double a = ptr_1[i];
		double b = ptr_2[i];
		ptr_output[i] = sqrtf(a*a + b * b);
	}
	return output;
}
void AlgMiddle::sampesfr(const cv::Mat &dat, float fre,
	double del, cv::Mat &eff, float &freqval, float &sfrval)
{

	std::vector<cv::Mat> channels(3);
	channels.clear();
	cv::split(dat, channels);
	auto ptr_1 = channels[0].ptr<double>();//x
	auto ptr_2 = channels[1].ptr<double>();//y
	Point2f point1, point2;



	std::vector<double>x;
	std::vector<double>y;
	x.clear();
	y.clear();
	int row = channels[0].rows;
	int col = channels[0].cols;
	for (int i = 0; i < channels[0].rows; ++i) {
		for (int j = 0; j < channels[0].cols; ++j) {
			x.push_back(channels[0].at<double>(i, j));
		}
	}

	for (int i = 0; i < channels[1].rows; ++i) {
		for (int j = 0; j < channels[1].cols; ++j) {
			y.push_back(channels[1].at<double>(i, j));
		}
	}
	if (x.size() < 2)
	{
		sfrval = 0;
		return;
	}
	double *x0 = new double[x.size()];
	double *y0 = new double[y.size()];
	for (int i = 0; i < x.size(); i++)
	{
		x0[i] = x.at(i);
		y0[i] = y.at(i);
	}

	SplineSpace::SplineInterface* sp = new SplineSpace::Spline(x0, y0, x.size());	//使用接口，且使用默认边界条件
	double x_fre = fre;
	double y_sfr;
	sp->SinglePointInterp(x_fre, y_sfr);	//求x的插值结果y
	if (y_sfr < 0) {
		//cout << "sfr value error" << endl;
	}
	sfrval = y_sfr;
	freqval = fre;
	delete[]sp;
	delete[]x0;
	delete[]y0;
	return;


}
