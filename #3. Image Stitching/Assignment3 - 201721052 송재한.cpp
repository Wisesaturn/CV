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
    // == result_ROI (result_Panorama, Rect(0, 0, left.cols, left.rows));
    left.copyTo(result_ROI);
    // ROI 영역만 left Copy!

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