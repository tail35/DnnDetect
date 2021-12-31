#include "DnnDetect.h"



static cv::Mat visualize(cv::Mat input, cv::Mat faces, bool print_flag=false, double fps=-1, int thickness=2)
{
    cv::Mat output = input.clone();

    if (fps > 0) {
        cv::putText(output, cv::format("FPS: %.2f", fps), cv::Point2i(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
    }

    for (int i = 0; i < faces.rows; i++)
    {
        if (print_flag) {
            cout << "Face " << i
                 << ", top-left coordinates: (" << faces.at<float>(i, 0) << ", " << faces.at<float>(i, 1) << "), "
                 << "box width: " << faces.at<float>(i, 2)  << ", box height: " << faces.at<float>(i, 3) << ", "
                 << "score: " << faces.at<float>(i, 14) << "\n";
        }

        // Draw bounding box
        cv::rectangle(output, cv::Rect2i(int(faces.at<float>(i, 0)), int(faces.at<float>(i, 1)), int(faces.at<float>(i, 2)), int(faces.at<float>(i, 3))), cv::Scalar(0, 255, 0), thickness);
        // Draw landmarks
        cv::circle(output, cv::Point2i(int(faces.at<float>(i, 4)),  int(faces.at<float>(i, 5))),  2, cv::Scalar(255,   0,   0), thickness);
        cv::circle(output, cv::Point2i(int(faces.at<float>(i, 6)),  int(faces.at<float>(i, 7))),  2, cv::Scalar(  0,   0, 255), thickness);
        cv::circle(output, cv::Point2i(int(faces.at<float>(i, 8)),  int(faces.at<float>(i, 9))),  2, cv::Scalar(  0, 255,   0), thickness);
        cv::circle(output, cv::Point2i(int(faces.at<float>(i, 10)), int(faces.at<float>(i, 11))), 2, cv::Scalar(255,   0, 255), thickness);
        cv::circle(output, cv::Point2i(int(faces.at<float>(i, 12)), int(faces.at<float>(i, 13))), 2, cv::Scalar(  0, 255, 255), thickness);
        // Put score
        cv::putText(output, cv::format("%.4f", faces.at<float>(i, 14)), cv::Point2i(int(faces.at<float>(i, 0)), int(faces.at<float>(i, 1))+15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
    }
    return output;
}


static cv::Mat visualize_back(cv::Mat input, cv::Mat faces,int scale,int thickness = 2)
{
 cv::Mat output = input.clone();
 for (int i = 0; i < faces.rows; i++)
 {
  // Draw bounding box
  cv::rectangle(output, cv::Rect2i(int(faces.at<float>(i, 0))*scale, int(faces.at<float>(i, 1))*scale, int(faces.at<float>(i, 2)) * scale, int(faces.at<float>(i, 3)) * scale), cv::Scalar(255, 0, 0), thickness);
 }
 return output;
}



void TestDnnPhoto() {
  cv::String modelPath = "yunet.onnx";
  // �ڶ�������ȡͼ��
  Mat img = imread("D:\\mpeg1\\0.jpg");
  // ����������ʼ��FaceDetectorYN
  Ptr<FaceDetectorYN> faceDetector = FaceDetectorYN::create(modelPath, "", img.size());
  // ���Ĳ��������������������浽һ��Mat��
  Mat faces;
  faceDetector->detect(img, faces);

  cv::Mat vis_image = visualize(img, faces, true);

  cv::String input = "111";

  cv::namedWindow(input, cv::WINDOW_AUTOSIZE);
  cv::imshow(input, vis_image);
  cv::waitKey(0);


 int k = 0;
 std::cin >> k;
}

void TestDnnCaptrue() {
  cv::String modelPath = "yunet.onnx";
  // �ڶ�������ȡͼ��
  Mat img;
  img.cols = 1280;
  img.rows = 720;  
  // ����������ʼ��FaceDetectorYN
  Ptr<FaceDetectorYN> faceDetector = FaceDetectorYN::create(modelPath, "", img.size());

  int deviceId = 0;
  cv::VideoCapture cap(0);
  //cap.open(deviceId, cv::CAP_ANY);
  //int frameWidth = int(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  //int frameHeight = int(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  
  cv::Mat frame;
  cv::TickMeter tm;
  while (cv::waitKey(30) != 27) //Press any key to exit
  {
   cap >> frame;

   std::chrono::time_point<std::chrono::steady_clock>  start = std::chrono::steady_clock::now();

   cv::Mat faces;
   tm.start();
   
   cv::Mat src(frame);
   //float scale = 0.088;   
   //float scale = 0.15;
   float scale = 0.5;
   int width = src.cols * scale;
   int hight = src.rows * scale;
   cv::resize(src, frame, cv::Size(width, hight));
   //faceDetector->setInputSize(cv::Size(src.cols, src.rows));
   faceDetector->setInputSize(cv::Size(width, hight));
   faceDetector->detect(frame, faces);
   tm.stop();
   
   //cv::Mat vis_frame = visualize(frame, faces, false, tm.getFPS());
   float k = 1.0 / scale;
   faces.cols = faces.cols * k;
   faces.rows = faces.rows * k;
   cv::Mat vis_frame = visualize_back(src,faces, k);
   imshow("libfacedetection demo", vis_frame);
   
   tm.reset();
   std::chrono::time_point<std::chrono::steady_clock>  end = std::chrono::steady_clock::now();
   auto dur = end - start;
   std::cout << "dur:" << dur.count() / 1000 / 1000 << std::endl;

  }


  //// ���Ĳ��������������������浽һ��Mat��
  //Mat faces;
  //faceDetector->detect(img, faces);

  //cv::Mat vis_image = visualize(img, faces, true);

  //cv::String input = "111";

  //cv::namedWindow(input, cv::WINDOW_AUTOSIZE);
  //cv::imshow(input, vis_image);
  //cv::waitKey(0);


  int k = 0;
  std::cin >> k;
}

int DnnDetect()
{
    //cv::CommandLineParser parser(argc, argv,
    //    "{help  h           |            | Print this message.}"
    //    "{input i           |            | Path to the input image. Omit for detecting on default camera.}"
    //    "{backend_id        | 0          | Backend to run on. 0: default, 1: Halide, 2: Intel's Inference Engine, 3: OpenCV, 4: VKCOM, 5: CUDA}"
    //    "{target_id         | 0          | Target to run on. 0: CPU, 1: OpenCL, 2: OpenCL FP16, 3: Myriad, 4: Vulkan, 5: FPGA, 6: CUDA, 7: CUDA FP16, 8: HDDL}"
    //    "{model m           | yunet.onnx | Path to the model. Download yunet.onnx in https://github.com/ShiqiYu/libfacedetection.train/tree/master/tasks/task1/onnx.}"
    //    "{score_threshold   | 0.9        | Filter out faces of score < score_threshold.}"
    //    "{nms_threshold     | 0.3        | Suppress bounding boxes of iou >= nms_threshold.}"
    //    "{top_k             | 5000       | Keep top_k bounding boxes before NMS.}"
    //    "{save  s           | false      | Set true to save results. This flag is invalid when using camera.}"
    //    "{vis   v           | true       | Set true to open a window for result visualization. This flag is invalid when using camera.}"
    //);
    //if (argc == 1 || parser.has("help"))
    //{
    //    parser.printMessage();
    //    return -1;
    //}

    //cv::String modelPath = parser.get<cv::String>("model");
    cv::String modelPath = "yunet.onnx";

    //int backendId = parser.get<int>("backend_id");
    int backendId = 0;

    //int targetId = parser.get<int>("target_id");
    int targetId = 0;

    //float scoreThreshold = parser.get<float>("score_threshold");
    float scoreThreshold = 0.9;
    //float nmsThreshold = parser.get<float>("nms_threshold");
    float nmsThreshold = 0.3;

    //int topK = parser.get<int>("top_k");
    int topK = 5000;

    //bool save = parser.get<bool>("save");
    bool save = false;
    //bool vis = parser.get<bool>("vis");
    bool vis = true;

    std::filesystem::path mypath = std::filesystem::current_path();
    std::string str = mypath.generic_string()+"\\yunet.onnx";
    modelPath = str.c_str();
     
    //Initialize FaceDetectorYN
    cv::Ptr<cv::FaceDetectorYN> detector = cv::FaceDetectorYN::create(modelPath, "", cv::Size(320, 320), scoreThreshold, nmsThreshold, topK, backendId, targetId);
     

    //If input is an image
    //if (parser.has("input"))
    if(false)
    {
        cv::String input = "";//parser.get<cv::String>("input");
        cv::Mat image = cv::imread(input);

        
        cv::Mat faces;
        detector->detect(image, faces);

        cv::Mat vis_image = visualize(image, faces, true);
        if(save)
        {
            cout << "result.jpg saved.\n";
            cv::imwrite("result.jpg", vis_image);
        }
        if (vis)
        {
            cv::namedWindow(input, cv::WINDOW_AUTOSIZE);
            cv::imshow(input, vis_image);
            cv::waitKey(0);
        }
    }
    else
    {
        int deviceId = 0;
        cv::VideoCapture cap(0);
        cap.open(deviceId, cv::CAP_ANY);
        int frameWidth = int(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int frameHeight = int(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        detector->setInputSize(cv::Size(frameWidth, frameHeight));

        cv::Mat frame;
        cv::TickMeter tm;
        while(cv::waitKey(30) != 27) //Press any key to exit
        {
            cap >> frame;            
            cv::Mat faces;
            //tm.start();
            //imshow("libfacedetection demo", frame);
            detector->detect(frame, faces);
            //tm.stop();
            //cv::Mat vis_frame = visualize(frame, faces, false, tm.getFPS());
            //imshow("libfacedetection demo", vis_frame);
            //tm.reset();
        }
    }
    return true;
}