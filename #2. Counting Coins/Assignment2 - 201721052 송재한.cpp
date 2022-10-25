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