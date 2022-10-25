# 📌 Computer Vision (opencv)

- 전공 수업 학습 자료를 모아놓았습니다. **(백업 및 데이터 저장용)**

### 📎 1. Multiband-blending

```c++
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
```

---

### 📎 2. Counting Coins

```c++
// Since 2021-11-05
// Computer Vision
// #2. Counting Coins for assignment
// 201721052 송재한

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
//opencv include

#include <iostream>
//standard include

/*
#define canny_up 125
#define canny_down 80
#define gaussian 15
*/
#define bilateral 105
#define hough_min 62
//threshold boundary

using namespace cv;
using namespace std;
//using namespace


int main()
{
    vector<Mat> coin_image(6);
    coin_image[0] = imread("./resource/assignment2/coins0.jpg", 0);
    coin_image[1] = imread("./resource/assignment2/coins1.jpg", 0);
    coin_image[2] = imread("./resource/assignment2/coins2.jpg", 0);
    coin_image[3] = imread("./resource/assignment2/coins3.jpg", 0);
    coin_image[4] = imread("./resource/assignment2/coins4.jpg", 0);
    coin_image[5] = imread("./resource/assignment2/coins5.jpg", 0);
    // Image Loading

    Mat find_edge;
    // find edge variable

    vector<Vec3f> Drawing;
    // for drawing variable

    vector<int> answer = { 10, 13, 9, 15, 16, 8 };
    // actual number of circles

    int count = 0;
    // for counting variable

    for (int i = count; i < coin_image.size(); i++) {

        bilateralFilter(coin_image[i], find_edge, 11, bilateral, bilateral);
        medianBlur(find_edge, find_edge, 5);
        // apply to blur (using medianblur, bilateralfilter)

        /*
        Canny(find_edge, find_edge, canny_down, canny_up);
        GaussianBlur(find_edge, find_edge, Size(gaussian, gaussian), 2);
        */

       // didn't use Canny and Gaussian (for Accuracy)

        HoughCircles(find_edge, Drawing, HOUGH_GRADIENT, 1, 80, 75, hough_min, 0, 0);
        // apply to Hough Circle Transform

        cout << "number of circles in coins" << i << ".jpg : " << Drawing.size() << endl;
        // print number of circles

         for (int j = 0; j < Drawing.size(); j++)
            {
                Vec<int, 3> info = Drawing[j];
                // input the information about circle
                Point center(info[0], info[1]);
                // input center of circle coordinate
                int radius = info[2];
                // input radius of circle

              circle(coin_image[i], center, radius, Scalar::all(0), 8);
              //Draw boundary of circle using center and radius
              circle(coin_image[i], center, 3, Scalar::all(0), 10);
              //Draw point of center
            }
         // Drawing the Circles
    }

    cout << "-------------------------------------" << endl;
    // use the line for divide

    for (const auto& n : coin_image) {
        imshow("result_image" + to_string(count), n);
        cout << "actual number of circles in coin" << count << ".jpg is : " << answer[count] << endl;
        count++;
    }
    // print actual number of circles and show images
    waitKey();

    return 0;
}
```

### 📎 3. Image Stitcing

```c++
// Since 2021-12-24
// Computer Vision
// #3. Image Stitching for assignment
// 201721052 송재한

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>
//opencv include

#include <iostream>
//standard include

using namespace cv;
using namespace std;
//using namespace

int main(int argc, char* argv[])
{
    int boundary_size = 3000;
    int threshold_descriptor = 650;
    double threshold_better = 4.1;
    double MinDist = 100;
    double MaxDist = 0;
    // set the variable

    Mat left = imread("./resource/assignment3/left.jpg");
    Mat right = imread("./resource/assignment3/right.jpg");
    // Image Loading (No Gray)

    if (left.empty() || right.empty())
    {
        cout << "Couldn't find the image!" << endl;
        return -1;
    }
    // IF - Couldn't find or open the image?

    cout << "left image size : " << left.rows << " * " << left.cols << endl;
    cout << "right image size : " << right.rows << " * " << right.cols << endl;

    if (left.rows > boundary_size || left.cols > boundary_size) {
        for (int i = 0; i < 2; i++)
            pyrDown(left, left);
    }
    if (right.rows > boundary_size || right.cols > boundary_size) {
        for (int i = 0; i < 2; i++)
            pyrDown(right, right);
    }
    // IF - being size of the image over 3000?


    Ptr<xfeatures2d::SURF> detector = xfeatures2d::SURF::create(threshold_descriptor);
    vector<KeyPoint> keypoints_left, keypoints_right;
    Mat descriptors_left, descriptors_right;
    detector->detectAndCompute(left, noArray(), keypoints_left, descriptors_left);
    detector->detectAndCompute(right, noArray(), keypoints_right, descriptors_right);
    // 1. Detect the keypoints and Calculate descriptor

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
    vector< DMatch > matches;
    matcher->match(descriptors_left, descriptors_right, matches);
    // 2. Matching descriptor

    Mat result_match;
    drawMatches(left, keypoints_left, right, keypoints_right, matches, result_match);
    // 3. Draw matches

    double Distance;
    for (int i = 0; i < descriptors_left.rows; i++) {
        Distance = matches[i].distance;
        if (Distance < MinDist)  MinDist = Distance;
        if (Distance > MaxDist)  MaxDist = Distance;
    }

    cout << "Max Distance is " << MaxDist << endl;
    cout << "Min Distance is " << MinDist << endl;

    // 4. Calculate distance between keypoints

    vector< DMatch > better;
    Mat result_better;

    for (size_t i = 0; i < descriptors_left.rows; i++) {
        if (matches[i].distance < threshold_better * MinDist)
            better.push_back(matches[i]);
    }
    drawMatches(left, keypoints_left, right, keypoints_right, better, result_better);

    // 5. find the better matches and Draw the better Matches

    vector<Point2f> object;
    vector<Point2f> background;

    for (size_t i = 0; i < better.size(); i++) {
        object.push_back(keypoints_left[better[i].queryIdx].pt);
        background.push_back(keypoints_right[better[i].trainIdx].pt);
    }

    Mat H;
    H = findHomography(background, object, RANSAC);
    cout << "HomoMatrix is " << H << endl;

    // 6. Calculate the Homography Matrix

    Mat result_wrap;
    warpPerspective(right, result_wrap, H, Size(right.cols + left.cols, right.rows));

    // 7. right image Wraping using H

    Mat result_Panorama = result_wrap.clone();
    Mat result_ROI = result_Panorama(Rect(0, 0, left.cols, left.rows));
    // == result_ROI (result_Panorama, Rect(0, 0, left.cols, le`ft.rows));
    left.copyTo(result_ROI);
    // ROI left Copy!

    // 8. Image stitching and decide the region of interest

    vector<Point> Real(result_wrap.cols * result_wrap.rows);

    for (size_t j = 0; j < result_Panorama.rows; j++) {
        for (size_t i = 0; i < result_Panorama.cols; i++) {
            if (result_Panorama.at<Vec3b>(j, i) != Vec3b(0, 0, 0))
                Real.push_back(Point(i, j));
        }
    }

    Rect boundary = boundingRect(Real);

    // 9. cut the black image

    imshow("left(background)", left);
    imshow("right(object)", right);
    imshow("Match", result_match);
    imshow("better_Match", result_better);
    imshow("wrap", result_wrap);
    imshow("wrap and cut", result_ROI);
    imshow("result", result_Panorama);
    imshow("result_black_cut", result_Panorama(boundary));
    waitKey();
    // FIN. show the result
    return 0;
}
```
