#ifndef ADVANCED_REFLECTOR_DETECTOR_NODE_HPP_
#define ADVANCED_REFLECTOR_DETECTOR_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <ultralytics_ros/msg/yolo_result.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/registration/ndt.h>

namespace advanced_reflector_detector {

class AdvancedReflectorDetectorNode : public rclcpp::Node {
public:
  explicit AdvancedReflectorDetectorNode(const rclcpp::NodeOptions &options);

private:
  void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
  void yoloCallback(const ultralytics_ros::msg::YoloResult::SharedPtr msg);
  void lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  void processPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr lidar_msg,
                        const ultralytics_ros::msg::YoloResult::SharedPtr yolo_msg);
  Eigen::Vector4f project2Dto3D(float u, float v, float depth);
  void clusterPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &clusters);
  void estimateReflectorPose(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cluster,
                            geometry_msgs::msg::Pose &pose);
  void transformToGlobalFrame(const geometry_msgs::msg::Pose &local_pose,
                             geometry_msgs::msg::Pose &global_pose);
  void projectToGroundPlane(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                           pcl::PointCloud<pcl::PointXYZ>::Ptr &projected_cloud);
  void estimateGroundPlane(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                          Eigen::Vector4f &plane_coefficients);
  void expandVolume(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr &volume_cloud,
                    float height);
  bool matchWithModel(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                      float &score);

  // Subscribers
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
  rclcpp::Subscription<ultralytics_ros::msg::YoloResult>::SharedPtr yolo_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;

  // Publishers
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr roi_cloud_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr reflector_pose_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr projected_cloud_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr volume_cloud_pub_;

  // TF
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // Data
  sensor_msgs::msg::CameraInfo camera_info_;
  bool camera_info_received_ = false;
  ultralytics_ros::msg::YoloResult::SharedPtr yolo_result_;
  geometry_msgs::msg::PoseStamped::SharedPtr vehicle_pose_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr reflector_model_;

  // Parameters
  std::string lidar_topic_;
  std::string yolo_topic_;
  std::string camera_info_topic_;
  std::string pose_topic_;
  std::string roi_cloud_topic_;
  std::string reflector_poses_topic_;
  std::string reflector_markers_topic_;
  std::string projected_cloud_topic_;
  std::string volume_cloud_topic_;
  std::string reflector_class_id_;
  float volume_height_;
  float ndt_score_threshold_;
};

}  // namespace advanced_reflector_detector

#endif  // ADVANCED_REFLECTOR_DETECTOR_NODE_HPP_