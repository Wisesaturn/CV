// Computer Vision #1. Multiband blending for assignment
// 201721052 송재한

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
// 이미지를 확대,축소할 수 있는 헤더 (이미지 프로세싱)
#include <iostream>
#define Pyramid_level Lev
// 피라미드 레벨 축약하여 'Lev'로 쓸려고 정의

using namespace cv;
using namespace std;
const int Pyramid_level = 6;
// 피라미드 레벨 지정 (Lev로 축약해서 사용)

Mat reconstruct(const vector<Mat> &pyramid)
// 원리는 사이즈를 맞춰준 다음 다 더해주면 Original!!
{
    Mat outside = pyramid.back();
    // 제일 끝에 있는 이미지를 갖고왔다
    for (int i = int(pyramid.size())-2; i >= 0; i--)
    {
        // auto로 쓰면 size_t unsigned long으로 계산되는데, unsigned이기 때문에 0에서 음수로 가면
        // 무진장 큰 값으로 변한다! (조건에선 이상 없으므로 그대로 진행 => 오류)
        // 그렇기 때문에 i--할 땐 auto보단 int로 사용하자!
        pyrUp(outside, outside, pyramid[i].size());
        // 사이즈를 pyramid[i]에 맞게 키운다
        // out 변수 자체에서 이미지를 키운다
        outside += pyramid[i];
        // pyramid[i]와 out을 더한다
        // pyramid[i]와 out를 더한 값이 계속 중첩되어 쌓인다 = 원본이 된다
    }
    return outside;
}

vector<Mat> GaussianPyramid(const Mat &source)
{
    // copy하지 말고 그대로 읽어서 보내주라 cosnt, & 해놓기
    vector<Mat> GaussianPyramid(Lev);
    GaussianPyramid[0] = source;
    for (auto i = 1; i < Lev; i++)
        pyrDown(GaussianPyramid[i-1], GaussianPyramid[i]);
    return GaussianPyramid;
}

vector<Mat> LaplacianPyramid_inputMat(const Mat &source)
{
    // copy하지 말고 그대로 읽어서 보내주라 cosnt, & 해놓기
    std::vector<Mat> Pyramid(Lev);
    Pyramid[0] = source.clone();
    // Pyramid[0] =src;로 사용하게 된다면 src를 나타내주지 않는다.
    // 이유는 src의 데이터가 다른 공간에 미리 저장되있고, Pyramid[0]은 단순히 포인터의 형식으로 읽어오는데
    // 이를 방지하려면 src.clone()을 사용하여 데이터를 그.대.로 복사해와야 src까지 출력할 수 있다.

    for (auto i = 1; i < Lev; i++)
    {
        pyrDown(Pyramid[i - 1], Pyramid[i]);
        //Pyramid[i]에 high 영역을 제외한 축소한 그림을 저장(down-sampling)
        Mat temp;
        // 임시 이미지
        pyrUp(Pyramid[i], temp, Pyramid[i - 1].size());
        // 원본 사이즈(pyramid[i-1].size())로 키워서 temp에 저장해라
        // pyramid[i]라는 down-sampling한 이미지를 다시 원본 사이즈로 키우기!
        Pyramid[i - 1] -= temp;
        // pyramid[i-1]에서 temp 이미지를 빼라
    }
    return Pyramid;
}

int main(int argc, const char *argv[])
{
    string path = "D:/Computer Vision/resources/";
    // 파일 source path 지정 (내 컴퓨터 기준)
    Mat apple = imread(path + "burt_apple.png");
    Mat orange = imread(path + "burt_orange.png");
    Mat divide = imread(path + "burt_mask.png");
    Mat divide_inverse;
    apple.convertTo(apple, CV_32F, 1/255.f);
    orange.convertTo(orange, CV_32F, 1 / 255.f);
    divide.convertTo(divide, CV_32F, 1 / 255.f);
    subtract(1, divide, divide_inverse);
    // source 이미지 불러오기

    vector<Mat> result(Lev);
    Mat answer;
    // Multiband-blending 완료용 Mat

    vector<Mat> Pyramid_apple, Pyramid_orange, Pyramid_mask, Pyramid_mask_inverse;
    vector<Mat> adding_pyr(Lev), adding_pyr2(Lev);
    // 피라미드용 vector 선언

    Pyramid_mask = GaussianPyramid(divide);
    Pyramid_mask_inverse = GaussianPyramid(divide_inverse);
    // 1단계. mask에 Gaussian Pyramid를 적용시키자

    Pyramid_apple = LaplacianPyramid_inputMat(apple);
    Pyramid_orange = LaplacianPyramid_inputMat(orange);
    // 2단계. 이미지에 Laplacian Pyramid를 적용시키자

    for (int i = 0; i < Lev; i++) {
        multiply(Pyramid_apple[i], Pyramid_mask[i], adding_pyr[i]);
        multiply(Pyramid_orange[i], Pyramid_mask_inverse[i], adding_pyr2[i]);
       add(adding_pyr[i], adding_pyr2[i], result[i]);
    }
    // 3단계. 이미지에 mask를 적용시키고 각각 더하자

    answer = reconstruct(result);
    // 4단계. 단계별로 나눈 이미지를 reconstruct하여 최종본을 만들자

    imshow("result", answer);
    waitKey();
    // 이미지 보여주기
    
    return 0;
}