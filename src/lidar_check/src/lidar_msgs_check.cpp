#ifndef PCL_NO_PRECOMPILE
#define PCL_NO_PRECOMPILE
#endif

#include <opencv2/imgcodecs.hpp>

#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>

#include <vector>
#include <string>

#include <iostream>
#include <fstream>

std::vector<std::string> _LIDAR_TOPICS{
    "/ouster/top/points", 
    "/ouster/left_top_os2/points", 
    "/ouster/left_bottom_os2/points", 
    "/ouster/left_os0/points"
};

float k_min_thre = 0.00001;

// Point Cloud type structs
// namespace cyngn::localization {
struct PointXYZITR {
  PCL_ADD_POINT4D;
  float intensity;
  double timestamp;
  uint16_t ring;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

struct PointXYZITR_helper {
  PCL_ADD_POINT4D;
  uint8_t intensity;
  double timestamp;
  uint16_t ring;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

struct EIGEN_ALIGN16 PointXYZITR_Ouster_helper {
  PCL_ADD_POINT4D;
  float intensity;
  uint32_t timestamp;
  uint16_t reflectivity;
  uint8_t ring;
  uint16_t ambient;
  uint32_t range;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZITR,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity,
                                                                          intensity)(double, timestamp,
                                                                                     timestamp)(uint16_t, ring, ring));

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZITR_helper,
                                  (float, x, x)(float, y, y)(float, z, z)(uint8_t, intensity,
                                                                          intensity)(double, timestamp,
                                                                                     timestamp)(uint16_t, ring, ring));

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZITR_Ouster_helper,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(uint32_t, timestamp, t)(
        uint16_t, reflectivity, reflectivity)(uint8_t, ring, ring)(uint16_t, ambient, ambient)(uint32_t, range, range));

// Parser
static int cnt = 0;
bool ConvertOusterPointType(const pcl::PointCloud<PointXYZITR_Ouster_helper>::Ptr &input,
                              pcl::PointCloud<PointXYZITR>::Ptr &output, double time_s, const std::string &output_dir) {
  if (input == nullptr || output == nullptr) {
    return false;
  }


  static constexpr uint16_t kNumofScanLines = 128;
  auto is_valid_scanline = [&](const auto &pt) {
    if (pt.ring > kNumofScanLines) return false;
    return true;
  };

  // timestamp of ouster lidar points are relative wrt the first column
  static constexpr int64_t kScanDurationNS = 100 * 1e6;  // 100ms
  auto is_valid_timestamp = [&](const auto &pt) {
    if (pt.timestamp > kScanDurationNS) return false;
    return true;
  };

  uint32_t invalid_timestamp_count = 0;
  uint32_t invalid_scan_line_count = 0;
  auto img = cv::Mat(128, 1024, CV_8UC1);
  double inval_count = 0;

  for (size_t h = 0; h < input->height; h++) {
    for (size_t w = 0; w < input->width; w++) {
      const auto &input_point = input->at(w, h);
      // check the points with bad ring number
      if (!is_valid_scanline(input_point)) {
        invalid_scan_line_count++;
        continue;
      }

      // check the points with bad timestamp
      if (!is_valid_timestamp(input_point)) {
        invalid_timestamp_count++;
      }
      if (std::abs(input_point.x) < k_min_thre || std::abs(input_point.y) < k_min_thre || std::abs(input_point.z) < k_min_thre) {

          img.at<uint8_t>(h, w) = 0;
          inval_count++;
          continue;
      } 
      img.at<uint8_t>(h, w) = 255;
      PointXYZITR output_point;
      output_point.x = input_point.x;
      output_point.y = input_point.y;
      output_point.z = input_point.z;
      output_point.intensity = input_point.reflectivity / 255.0f;
      output_point.ring = input_point.ring;
      output_point.timestamp = input_point.timestamp;
      output->push_back(std::move(output_point));
    }
  }
  std::string fname = output_dir + "/" + std::to_string(time_s) + ".jpg";
  cv::imwrite(fname, img);
  return true;
}



void write_output_file(std::vector<int>& output_data, const std::string& name, const std::string &output_dir) {
    // output files
    std::string filename = output_dir + name + ".txt";
    std::ofstream outputFile(filename, std::ios_base::out);
    
    if (outputFile.is_open()) {
        for (const auto& number : output_data) {
            outputFile << number << "\n";
        }

        outputFile.close();
        std::cout << "Vector output to file successfully.\n";
    } else {
        std::cout << "Unable to open the file.\n";
    }
}

int main(int argc, char* argv[])
{
    if (argc != 4) {
      std::cout << "rosrun lidar_check lidar_check_node bag_name topic_name output_dir\n";
      return -1;
    }
    std::string bag_name = argv[1];
    std::string topic = argv[2];
    std::string output_dir = argv[3];

    // Open the ROS bag file
    rosbag::Bag bag;
    bag.open(bag_name, rosbag::bagmode::Read);
    std::cout << "lidar topic = " << topic << std::endl;
    rosbag::View view(bag, rosbag::TopicQuery(topic));

    std::vector<int> pt_per_frame_list;
    std::vector<int> zero_pt_per_frame_list;
    // Iterate over the messages in the view
    
    for (const rosbag::MessageInstance& msg : view)
    {
        if (msg.getDataType() == "sensor_msgs/PointCloud2") {
            sensor_msgs::PointCloud2ConstPtr cloud_msg = msg.instantiate<sensor_msgs::PointCloud2>();
            pcl::PointCloud<PointXYZITR_Ouster_helper>::Ptr ouster_cloud_in_helper(new pcl::PointCloud<PointXYZITR_Ouster_helper>);
            pcl::fromROSMsg(*cloud_msg, *ouster_cloud_in_helper);

            // save pcd
            std::string fname = output_dir + "/" + std::to_string(cloud_msg->header.stamp.toSec()) + ".pcd";
            pcl::io::savePCDFileASCII (fname, *ouster_cloud_in_helper);

            pcl::PointCloud<PointXYZITR>::Ptr output(new pcl::PointCloud<PointXYZITR>);
            ConvertOusterPointType(ouster_cloud_in_helper, output, cloud_msg->header.stamp.toSec(), output_dir);

            // stats
            int cnt_zeros_pt_per_frame = 0; 
            for (size_t i = 0; i < (output->points).size(); ++i) {
                float x = (output->points)[i].x;
                float y = (output->points)[i].y;
                float z = (output->points)[i].z;

                if (std::abs(x) < k_min_thre && std::abs(y) < k_min_thre && std::abs(z) < k_min_thre) {
                    cnt_zeros_pt_per_frame++;
                }
            }
            pt_per_frame_list.push_back((output->points).size());
            zero_pt_per_frame_list.push_back(cnt_zeros_pt_per_frame);
        }
    }

    // Close the ROS bag file
    bag.close();

    // write files
    std::string str1 = "point_cnt";
    std::string str2 = "zero_point_cnt";
    write_output_file(pt_per_frame_list, str1, output_dir);
    write_output_file(zero_pt_per_frame_list, str2, output_dir);
    return 0;
}