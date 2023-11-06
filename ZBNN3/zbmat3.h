#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

std::vector<double> MaToVe(cv::Mat vin) {

	cv::Mat flat = vin.reshape(1, vin.total() * vin.channels());
	std::vector<double> out = vin.isContinuous() ? flat : flat.clone();
	return out;
}

cv::Mat vectorToMat64(const std::vector<double>& data, int rows, int cols) {
    if (data.size() != rows * cols) {
        throw std::invalid_argument("Size of data vector does not match rows and cols parameters.");
    }

    cv::Mat mat(rows, cols, CV_64F);
    memcpy(mat.data, data.data(), data.size() * sizeof(double));
    return mat;
}