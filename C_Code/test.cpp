#include <bits/stdc++.h>
#include <opencv2/core.hpp>
#include <opencv2/face.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace std;
using namespace cv;
using namespace cv::face;


// PS: Because of the version of my opencv is 3.4.2, I
//     choose to use the document of 3.4.2 to write the code.
//     但是他的教程使用版本仍然是2.4下使用的. 所以应该没有差别

// 包含的三个算法:
//  1. Eigenfaces(EigenFaceRecognizer::create)
//  2. Fisherfaces(FisherFaceRecognizer::create)
//  3. Local Binary Patterns Histograms(LBPHFaceRecognizer:create)

// Eigenfaces: 其实是用PCA找到主方向进行
// 利用PCA分解之后 1. 将所有training samples映射到PCA子空间上
//   2. 将query映射到PCA子空间上
//   3. 找到与query最邻近的training sample图片
// Tips:
//   1. 因为400张100*100的图片需要计算 XX^T 是 10000*10000 的matrix所以可以计算计算X^TX 只需要计算400*400的矩阵(PS:我之前就验证过这个事情...)
//   2. 上面 1. 的处理得到的特征值是不会变化的，但是特征向量需要左乘一个X矩阵才能得到对应的10000维的向量
//   3.

// 将矩阵数据范围转化为0-255的数值
static Mat norm_0_255(InputArray _src){
    Mat src = _src.getMat();
    Mat dst;
    switch(src.channels()){
        case 1:
            cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
            break;
        case 3:
            cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
            break;
        default:
            src.copyTo(dst);
            break;
    }
    return dst;
}

// 读取CSV文件
static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator=';'){
    for(int i = 0; i < 10; i +=20){
        cout<<i<<endl;
    }
    // PS: 第二个参数是指定mode的，in表示读文件
    std::ifstream file(filename.c_str(), ifstream::in);
    if(!file){
        string error_message = "No valid input file was given, please check the given filename.";
        // CV_Error是调用错误处理 参数1: 错误码, 参数2: 错误信息
        CV_Error(Error::StsBadArg, error_message);
    }
    string line, path, classlabel;
    while(getline(file, line)){
        stringstream liness(line);
        // 第三个参数是停止读入的字符 默认是\n 这里使用;所以起到一个分割的作用
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()){
            // imread第二个参数是读入模式 0 是 IMREAD_GRAYSCALE 代表的数字，表示灰度数据
            images.push_back(imread(path, 0));
            // atoi将字符串转化为数字
            labels.push_back(atoi(classlabel.c_str()));
        }else{
            cout<<"Some errors occured during reading the file: "<<path<<endl;
        }
    }
}

int main(int argc, const char *argv[]){
    for(int i = 0; i < 100; i +=95){
        cout<<i<<endl;
    }
    //检查有效的命令行参数的数量
    if (argc < 2){
        //argc < 2 就表示没有参数的输入，此时会报错(因为应该指定CSV文件的路径)
        cout<<"usage: "<<argv[0]<<" <csv.ext> <output_folder>"<<endl;
        exit(1);
    }
    // 如果没有参数指定 output的位置 那么进行直接展示而不是保存
    string output_folder = ".";
    if (argc == 3){
        output_folder = string(argv[2]);
    }
    // 指定csv的位置
    string fn_csv = string(argv[1]);
    // 保存images和labels的vector
    vector<Mat> images;
    vector<int> labels;
    // 读入数据 因为可能没有给定的文件，所以可能会报错
    try{
        // 进行数据库文件的读入
        read_csv(fn_csv, images, labels);
    } catch(cv::Exception& e) {
        // 文件不存在的时候报错 结束
        cerr<<"Error opening file \""<<fn_csv<<"\". Reason: "<<e.msg<<endl;
        exit(1);
    }
    // 文件数目不足 报错退出
    if(images.size() <= 1){
        string error_message = "This demo needs at least 2 images to work. Please add more images to the dataset!";
        CV_Error(Error::StsError, error_message);
    }
    // 获取image的高度, 是后续将images还原成原本的size的变量
    int height = images[0].rows;
    // 将最后一个元素取出来作为query
    Mat testSample = images[images.size() - 1];
    int testLabel = labels[labels.size() - 1];
    images.pop_back();
    labels.pop_back();
    // 下面构建了一个Eigenfaces model去进行人脸识别 其需要训练
    // 1. 如果想要指定保留的特征向量的数目
    //   使用: EigenFaceRecognizer::create(num);
    // 2. 如果想要加上一个置信系数阈值
    //   使用: EigenFaceRecognizer::create(num, threshold);
    // 3. 如果想使用全部的特征向量
    //   使用: EigenFaceRecognizer::create();
    //   或者: EigenFaceRecognizer::create(0);
    // 4. 同理，如果想使用全部特征向量并且加上置信系数的阈值
    //   使用: EigenFaceRecognizer::create(0, threshold);
    // PS: threshold是什么目前没有明白，文档中写的是: The threshold applied in the prediction.
    Ptr<EigenFaceRecognizer> model = EigenFaceRecognizer::create();
    model->train(images, labels);
    int predictedLabel = model->predict(testSample);
    string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
    cout<<result_message<<endl;
    // 获取EigenValues 和 EigenVectors
    Mat eigenvalues = model->getEigenValues();
    Mat W = model->getEigenVectors();
    Mat mean = model->getMean();
    // 进行display或者保存
    if(argc == 2){
        // reshape的参数 1. channel数 2. row数目
        imshow("mean", norm_0_255(mean.reshape(1, height)));
    }else{
        imwrite(format("%s/mean.png", output_folder.c_str()), norm_0_255(mean.reshape(1, height)));
    }
    // 展示或者保存EigenFaces
    for(int i = 0; i < min(10, W.cols); i ++){
        string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
        cout<<msg<<endl;
        // EigenVector
        Mat ev = W.col(i).clone();
        // 将EigenVector reshape到原图像的尺度上并且进行normalization
        Mat grayscale = norm_0_255(ev.reshape(1, height));
        // 进行对图片进行Jet colormap并且展示
        Mat cgrayscale;
        applyColorMap(grayscale, cgrayscale, COLORMAP_JET);
        if(argc == 2){
            imshow(format("eigenface_%d", i), cgrayscale);
        }else{
            imwrite(format("%s/eigenface_%d.png", output_folder.c_str(), i), norm_0_255(cgrayscale));
        }
    }

    // 展示或者保存重建的face
    for(int num_components = min(W.cols, 10); num_components < min(W.cols, 300); num_components+=15){
        // 将eigenvector从model中slice出来
        Mat evs = Mat(W, Range::all(), Range(0, num_components));
        // subspaceProject()三个参数分别为 W: 特征向量 mean: 均值 src: 原始数据(要转为行向量)
        // 进行的是 y = E_d^T*(x-mean(x)) 操作
        Mat projection = LDA::subspaceProject(evs, mean, images[0].reshape(1,1));
        // subplaceReconstruct()三个参数分别也是上面的三个
        // 进行的是 result = mean(x) + E_d*y 的操作
        Mat reconstruction = LDA::subplaceReconstruct(evs, mean, projection);

        // 进行 normalization并且改变形状
        reconstruction = norm_0_255(reconstruction.reshape(1, images[0].rows));
        // 展示或者保存
        if(argc == 2){
            imshow(format("eigenface_reconstruction_%d", num_components), reconstruction);
        }else{
            imwrite(format("%s/eigenface_reconstruction_%d.png", output_folder.c_str(), num_components), reconstruction);
        }
    }
    if(argc == 2){
        waitKey(0);
    }
    return 0;
}












