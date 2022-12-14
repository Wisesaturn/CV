# ๐ Computer Vision (opencv)

- ์ ๊ณต ์์ ํ์ต ์๋ฃ๋ฅผ ๋ชจ์๋์์ต๋๋ค. **(๋ฐฑ์ ๋ฐ ๋ฐ์ดํฐ ์ ์ฅ์ฉ)**

### ๐ 1. Multiband-blending

```c++
// Computer Vision #1. Multiband blending for assignment
// 201721052 ์ก์ฌํ

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
// ์ด๋ฏธ์ง๋ฅผ ํ๋,์ถ์ํ  ์ ์๋ ํค๋ (์ด๋ฏธ์ง ํ๋ก์ธ์ฑ)
#include <iostream>
#define Pyramid_level Lev
// ํผ๋ผ๋ฏธ๋ ๋ ๋ฒจ ์ถ์ฝํ์ฌ 'Lev'๋ก ์ธ๋ ค๊ณ  ์ ์

using namespace cv;
using namespace std;
const int Pyramid_level = 6;
// ํผ๋ผ๋ฏธ๋ ๋ ๋ฒจ ์ง์  (Lev๋ก ์ถ์ฝํด์ ์ฌ์ฉ)

Mat reconstruct(const vector<Mat> &pyramid)
// ์๋ฆฌ๋ ์ฌ์ด์ฆ๋ฅผ ๋ง์ถฐ์ค ๋ค์ ๋ค ๋ํด์ฃผ๋ฉด Original!!
{
    Mat outside = pyramid.back();
    // ์ ์ผ ๋์ ์๋ ์ด๋ฏธ์ง๋ฅผ ๊ฐ๊ณ ์๋ค
    for (int i = int(pyramid.size())-2; i >= 0; i--)
    {
        // auto๋ก ์ฐ๋ฉด size_t unsigned long์ผ๋ก ๊ณ์ฐ๋๋๋ฐ, unsigned์ด๊ธฐ ๋๋ฌธ์ 0์์ ์์๋ก ๊ฐ๋ฉด
        // ๋ฌด์ง์ฅ ํฐ ๊ฐ์ผ๋ก ๋ณํ๋ค! (์กฐ๊ฑด์์  ์ด์ ์์ผ๋ฏ๋ก ๊ทธ๋๋ก ์งํ => ์ค๋ฅ)
        // ๊ทธ๋ ๊ธฐ ๋๋ฌธ์ i--ํ  ๋ auto๋ณด๋จ int๋ก ์ฌ์ฉํ์!
        pyrUp(outside, outside, pyramid[i].size());
        // ์ฌ์ด์ฆ๋ฅผ pyramid[i]์ ๋ง๊ฒ ํค์ด๋ค
        // out ๋ณ์ ์์ฒด์์ ์ด๋ฏธ์ง๋ฅผ ํค์ด๋ค
        outside += pyramid[i];
        // pyramid[i]์ out์ ๋ํ๋ค
        // pyramid[i]์ out๋ฅผ ๋ํ ๊ฐ์ด ๊ณ์ ์ค์ฒฉ๋์ด ์์ธ๋ค = ์๋ณธ์ด ๋๋ค
    }
    return outside;
}

vector<Mat> GaussianPyramid(const Mat &source)
{
    // copyํ์ง ๋ง๊ณ  ๊ทธ๋๋ก ์ฝ์ด์ ๋ณด๋ด์ฃผ๋ผ cosnt, & ํด๋๊ธฐ
    vector<Mat> GaussianPyramid(Lev);
    GaussianPyramid[0] = source;
    for (auto i = 1; i < Lev; i++)
        pyrDown(GaussianPyramid[i-1], GaussianPyramid[i]);
    return GaussianPyramid;
}

vector<Mat> LaplacianPyramid_inputMat(const Mat &source)
{
    // copyํ์ง ๋ง๊ณ  ๊ทธ๋๋ก ์ฝ์ด์ ๋ณด๋ด์ฃผ๋ผ cosnt, & ํด๋๊ธฐ
    std::vector<Mat> Pyramid(Lev);
    Pyramid[0] = source.clone();
    // Pyramid[0] =src;๋ก ์ฌ์ฉํ๊ฒ ๋๋ค๋ฉด src๋ฅผ ๋ํ๋ด์ฃผ์ง ์๋๋ค.
    // ์ด์ ๋ src์ ๋ฐ์ดํฐ๊ฐ ๋ค๋ฅธ ๊ณต๊ฐ์ ๋ฏธ๋ฆฌ ์ ์ฅ๋์๊ณ , Pyramid[0]์ ๋จ์ํ ํฌ์ธํฐ์ ํ์์ผ๋ก ์ฝ์ด์ค๋๋ฐ
    // ์ด๋ฅผ ๋ฐฉ์งํ๋ ค๋ฉด src.clone()์ ์ฌ์ฉํ์ฌ ๋ฐ์ดํฐ๋ฅผ ๊ทธ.๋.๋ก ๋ณต์ฌํด์์ผ src๊น์ง ์ถ๋ ฅํ  ์ ์๋ค.

    for (auto i = 1; i < Lev; i++)
    {
        pyrDown(Pyramid[i - 1], Pyramid[i]);
        //Pyramid[i]์ high ์์ญ์ ์ ์ธํ ์ถ์ํ ๊ทธ๋ฆผ์ ์ ์ฅ(down-sampling)
        Mat temp;
        // ์์ ์ด๋ฏธ์ง
        pyrUp(Pyramid[i], temp, Pyramid[i - 1].size());
        // ์๋ณธ ์ฌ์ด์ฆ(pyramid[i-1].size())๋ก ํค์์ temp์ ์ ์ฅํด๋ผ
        // pyramid[i]๋ผ๋ down-samplingํ ์ด๋ฏธ์ง๋ฅผ ๋ค์ ์๋ณธ ์ฌ์ด์ฆ๋ก ํค์ฐ๊ธฐ!
        Pyramid[i - 1] -= temp;
        // pyramid[i-1]์์ temp ์ด๋ฏธ์ง๋ฅผ ๋นผ๋ผ
    }
    return Pyramid;
}

int main(int argc, const char *argv[])
{
    string path = "D:/Computer Vision/resources/";
    // ํ์ผ source path ์ง์  (๋ด ์ปดํจํฐ ๊ธฐ์ค)
    Mat apple = imread(path + "burt_apple.png");
    Mat orange = imread(path + "burt_orange.png");
    Mat divide = imread(path + "burt_mask.png");
    Mat divide_inverse;
    apple.convertTo(apple, CV_32F, 1/255.f);
    orange.convertTo(orange, CV_32F, 1 / 255.f);
    divide.convertTo(divide, CV_32F, 1 / 255.f);
    subtract(1, divide, divide_inverse);
    // source ์ด๋ฏธ์ง ๋ถ๋ฌ์ค๊ธฐ

    vector<Mat> result(Lev);
    Mat answer;
    // Multiband-blending ์๋ฃ์ฉ Mat

    vector<Mat> Pyramid_apple, Pyramid_orange, Pyramid_mask, Pyramid_mask_inverse;
    vector<Mat> adding_pyr(Lev), adding_pyr2(Lev);
    // ํผ๋ผ๋ฏธ๋์ฉ vector ์ ์ธ

    Pyramid_mask = GaussianPyramid(divide);
    Pyramid_mask_inverse = GaussianPyramid(divide_inverse);
    // 1๋จ๊ณ. mask์ Gaussian Pyramid๋ฅผ ์ ์ฉ์ํค์

    Pyramid_apple = LaplacianPyramid_inputMat(apple);
    Pyramid_orange = LaplacianPyramid_inputMat(orange);
    // 2๋จ๊ณ. ์ด๋ฏธ์ง์ Laplacian Pyramid๋ฅผ ์ ์ฉ์ํค์

    for (int i = 0; i < Lev; i++) {
        multiply(Pyramid_apple[i], Pyramid_mask[i], adding_pyr[i]);
        multiply(Pyramid_orange[i], Pyramid_mask_inverse[i], adding_pyr2[i]);
       add(adding_pyr[i], adding_pyr2[i], result[i]);
    }
    // 3๋จ๊ณ. ์ด๋ฏธ์ง์ mask๋ฅผ ์ ์ฉ์ํค๊ณ  ๊ฐ๊ฐ ๋ํ์

    answer = reconstruct(result);
    // 4๋จ๊ณ. ๋จ๊ณ๋ณ๋ก ๋๋ ์ด๋ฏธ์ง๋ฅผ reconstructํ์ฌ ์ต์ข๋ณธ์ ๋ง๋ค์

    imshow("result", answer);
    waitKey();
    // ์ด๋ฏธ์ง ๋ณด์ฌ์ฃผ๊ธฐ

    return 0;
}
```

---

### ๐ 2. Counting Coins

```c++
// Since 2021-11-05
// Computer Vision
// #2. Counting Coins for assignment
// 201721052 ์ก์ฌํ

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

### ๐ 3. Image Stitcing

```c++
// Since 2021-12-24
// Computer Vision
// #3. Image Stitching for assignment
// 201721052 ์ก์ฌํ

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
