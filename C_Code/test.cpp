#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main(){
	Mat img = imread("攻略.jpg");
	if(img.empty())
    {
        fprintf(stderr, "Can not load image!\n");
        return -1;
    }
    //显示图像
    imshow("original picture", img);
    //此函数等待按键，按键盘任意键就返回
    waitKey();
	return 0;
}