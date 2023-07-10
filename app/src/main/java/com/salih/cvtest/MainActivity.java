package com.salih.cvtest;

import android.os.Bundle;
import android.util.Log;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.InstallCallbackInterface;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FastFeatureDetector;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.util.LinkedList;
import java.util.List;


/**
 * Created by Salih AKAR 10:10 10.07.2023
 */
public class MainActivity extends AppCompatActivity {
    private final String TAG = "main ";

    private CameraBridgeViewBase cameraBridgeViewBase;
    private LoaderCallbackInterface loaderCallbackInterface = new LoaderCallbackInterface() {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.e(TAG, "OpenCV loaded successfully");
                    cameraBridgeViewBase.enableView();

                }
                break;
                default:
                    this.onManagerConnected(status);
                    break;
            }
        }

        @Override
        public void onPackageInstall(int operation, InstallCallbackInterface callback) {

        }
    };

    @Override
    protected void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.e(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, loaderCallbackInterface);
        } else {
            Log.e(TAG, "OpenCV library found inside package. Using it!");
            loaderCallbackInterface.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

//        if (OpenCVLoader.initDebug())
//            Log.e(TAG, "onCreate: YES lib");
//        else
//            Log.e(TAG, "onCreate: Nope lib");

        cameraBridgeViewBase = findViewById(R.id.cameraView);
//        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, loaderCallbackInterface);

        cameraBridgeViewBase.setCvCameraViewListener(new CameraBridgeViewBase.CvCameraViewListener() {
            @Override
            public void onCameraViewStarted(int width, int height) {
                Log.e(TAG, "onCameraViewStarted: ");
//                initMyProcessors();
                try {
                    needToDetect = Utils.loadResource(MainActivity.this, R.drawable.banana);
                    Imgproc.cvtColor(needToDetect, needToDetect, Imgproc.COLOR_RGB2GRAY);
                    Log.e(TAG, "onCameraViewStarted: " + needToDetect.width());
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }

            @Override
            public void onCameraViewStopped() {
                Log.e(TAG, "onCameraViewStopped: ");

            }

            @Override
            public Mat onCameraFrame(Mat inputFrame) {
//                Log.e(TAG, "onCameraFrame: ");
//                return inputFrame;
//                return processImage(inputFrame);
//                return detectMyImage(inputFrame);
//                return detection2(inputFrame);
                return detection3(inputFrame);
            }

        });

    }


    private Point matchLoc;
    private Core.MinMaxLocResult mmr;

    private Mat detection3(Mat inputFrame) {

        if (needToDetect.empty())
            return inputFrame;

        Mat source = inputFrame.clone();
        Imgproc.cvtColor(source, source, Imgproc.COLOR_RGB2GRAY);
        int resCols = source.cols() - needToDetect.cols() + 1;
        int resRows = source.rows() - needToDetect.rows() + 1;
        Mat result = new Mat(resRows, resCols, CvType.CV_32FC1);

        Imgproc.matchTemplate(source, needToDetect, result, Imgproc.TM_CCOEFF);
//        Imgproc.threshold(result, result, 0.9, 1, Imgproc.THRESH_TOZERO);
        Core.normalize(result, result, 0, 1, Core.NORM_MINMAX, -1, new Mat());
        if (result.empty())
            return inputFrame;
        Core.MinMaxLocResult mmr = Core.minMaxLoc(result);

        Point matchLoc = mmr.minLoc;
        Log.e(TAG, "detection3: " + mmr.maxVal + " " + mmr.minVal);
        Imgproc.rectangle(inputFrame, matchLoc, new Point(matchLoc.x + needToDetect.cols(), matchLoc.y + needToDetect.rows()), new Scalar(0, 255, 0));


        return inputFrame;

    }

    private Mat detection2(Mat inputFrame) {
        if (needToDetect.empty())
            return inputFrame;

        Mat source = inputFrame.clone();
        Imgproc.cvtColor(source, source, Imgproc.COLOR_RGB2GRAY);

        int result_cols = source.cols() - needToDetect.cols() + 1;
        int result_rows = source.rows() - needToDetect.rows() + 1;
        Mat result = new Mat(result_rows, result_cols, CvType.CV_32FC1);

        Imgproc.matchTemplate(source, needToDetect, result, Imgproc.TM_CCORR_NORMED);
        Imgproc.threshold(result, result, 0.9, 1, Imgproc.THRESH_TOZERO);

//        while (true) {
//            mmr = Core.minMaxLoc(result);
//            matchLoc = mmr.maxLoc;
//            Log.e(TAG, "detection2: " + mmr.maxVal);
//            if (mmr.maxVal >= 0.9) {
//                Imgproc.rectangle(inputFrame, matchLoc, new Point(matchLoc.x + needToDetect.cols(), matchLoc.y + needToDetect.rows()), new Scalar(0, 255, 0));
//                Log.e(TAG, "detection2: detected");
//                break;
//            } else
//                break;
//        }
        mmr = Core.minMaxLoc(result);
        matchLoc = mmr.minLoc;
        Log.e(TAG, "detection2: " + mmr.minVal);
        if (mmr.minVal >= 0.9) {
            Imgproc.rectangle(inputFrame, matchLoc, new Point(matchLoc.x + needToDetect.cols(), matchLoc.y + needToDetect.rows()), new Scalar(0, 255, 0), 5);
        }

//        result = null;

        return inputFrame;

    }

    private Mat processImage(Mat frame) {
        MatOfKeyPoint matOfKeyPoint = new MatOfKeyPoint();
        try {
            FastFeatureDetector fastFeatureDetector = FastFeatureDetector.create(FastFeatureDetector.FAST_N);
            fastFeatureDetector.detect(frame, matOfKeyPoint);
            Scalar redColor = new Scalar(255, 0, 0);
            Mat mRgba = frame.clone();
            Imgproc.cvtColor(frame, mRgba, Imgproc.COLOR_RGBA2RGB);

            Features2d.drawKeypoints(mRgba, matOfKeyPoint, mRgba, redColor, 1);
            frame.release();
            return mRgba;

        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private Scalar RED = new Scalar(255, 0, 0);
    private Scalar GREEN = new Scalar(0, 255, 0);
    private FeatureDetector detector;
    private DescriptorExtractor descriptor;
    private DescriptorMatcher matcher;
    private Mat img1;
    private Mat descriptors1, descriptors2;
    private MatOfKeyPoint keyPoint1, keyPoint2;

    private Mat needToDetect;

    private void initMyProcessors() {
//        detector = FeatureDetector.create(FeatureDetector.ORB);
//        descriptor = DescriptorExtractor.create(DescriptorExtractor.ORB);
//        matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
//        img1 = new Mat();
        try {
            Mat img = Utils.loadResource(this, R.drawable.test);
//            Imgproc.cvtColor(img, img1, Imgproc.COLOR_RGB2GRAY);
//            img1.convertTo(img1, 0);

            needToDetect = img.clone();
            Imgproc.cvtColor(needToDetect, needToDetect, Imgproc.COLOR_RGB2GRAY);


//            descriptors1 = new Mat();
//            keyPoint1 = new MatOfKeyPoint();
//            detector.detect(img1, keyPoint1);
//            descriptor.compute(img1, keyPoint1, descriptors1);

        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException(e);
        }


    }

    private Mat detectMyImage(Mat frame) {
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BGR2GRAY);
        descriptors2 = new Mat();
        keyPoint2 = new MatOfKeyPoint();
        detector.detect(frame, keyPoint2);
        descriptor.compute(frame, keyPoint2, descriptors2);

//        match
        MatOfDMatch matches = new MatOfDMatch();
        if (img1.type() == frame.type()) {
            try {
                matcher.match(descriptors1, descriptors2, matches);
            } catch (Exception e) {
                Log.e(TAG, "detectMyImage: failed");
                e.printStackTrace();
            }
        } else {
            Log.e(TAG, "detectMyImage: return 1");
            Imgproc.cvtColor(frame, frame, Imgproc.COLOR_GRAY2RGBA);
            return frame;
        }

        List<DMatch> matchList = matches.toList();

        Double max = 0.0;
        Double min = 100.0;

        for (int i = 0; i < matchList.size(); i++) {
            Double dist = (double) matchList.get(i).distance;
            if (dist < min)
                min = dist;
            if (dist > max)
                max = dist;
        }

        LinkedList<DMatch> good_match = new LinkedList<>();
        for (int i = 0; i < matchList.size(); i++) {
            if (matchList.get(i).distance <= (1.5 * min))
                good_match.addLast(matchList.get(i));
        }

        MatOfDMatch lastMatches = new MatOfDMatch();
        lastMatches.fromList(good_match);
        Mat outImage = new Mat();
        MatOfByte drawnMatches = new MatOfByte();
        if (frame.empty() || frame.cols() < 1 || frame.rows() < 1) {
            Imgproc.cvtColor(frame, frame, Imgproc.COLOR_GRAY2RGBA);
            Log.e(TAG, "detectMyImage: return 2");
            return frame;
        }


        Features2d.drawMatches(img1, keyPoint1, frame, keyPoint2, lastMatches, outImage, GREEN, RED, drawnMatches, Features2d.NOT_DRAW_SINGLE_POINTS);
        Imgproc.resize(outImage, outImage, frame.size());
        return outImage;

    }


//           try {
//            img = Utils.loadResource(getApplicationContext(), R.drawable.resim);
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//        Imgproc.cvtColor(img, img, Imgproc.COLOR_RGB2BGRA);
//        Mat img_result = img.clone();
//        Imgproc.Canny(img, img_result, 80, 90);
//        Bitmap img_bitmap = Bitmap.createBitmap(img_result.cols(), img_result.rows(),Bitmap.Config.ARGB_8888);
//        Utils.matToBitmap(img_result, img_bitmap);
//        ImageView imageView = findViewById(R.id.img2);
//        imageView.setVisibility(View.VISIBLE);
//        imageView.setImageBitmap(img_bitmap);


//    /

}
