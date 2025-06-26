#include "advanced_reflector_detector/advanced_reflector_detector_node.hpp"
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_eigen/tf2_eigen.h>

namespace advanced_reflector_detector {

AdvancedReflectorDetectorNode::AdvancedReflectorDetectorNode(const rclcpp::NodeOptions &options)
    : Node("advanced_reflector_detector_node", options) {
  // Declare parameters
  declare_parameter<std::string>("lidar_topic", "/lidar_points");
  declare_parameter<std::string>("yolo_topic", "/yolo_result");
  declare_parameter<std::string>("camera_info_topic", "/camera_info");
  declare_parameter<std::string>("pose_topic", "/localization/pose");
  declare_parameter<std::string>("roi_cloud_topic", "/roi_pointcloud");
  declare_parameter<std::string>("reflector_poses_topic", "/reflector_poses");
  declare_parameter<std::string>("reflector_markers_topic", "/reflector_markers");
  declare_parameter<std::string>("projected_cloud_topic", "/projected_pointcloud");
  declare_parameter<std::string>("volume_cloud_topic", "/volume_pointcloud");
  declare_parameter<std::string>("reflector_class_id", "reflector");
  declare_parameter<float>("volume_height", 0.5); // Default height: 0.5m
  declare_parameter<float>("ndt_score_threshold", 0.8); // Default NDT score threshold

  // Get parameters
  lidar_topic_ = get_parameter("lidar_topic").as_string();
  yolo_topic_ = get_parameter("yolo_topic").as_string();
  camera_info_topic_ = get_parameter("camera_info_topic").as_string();
  pose_topic_ = get_parameter("pose_topic").as_string();
  roi_cloud_topic_ = get_parameter("roi_cloud_topic").as_string();
  reflector_poses_topic_ = get_parameter("reflector_poses_topic").as_string();
  reflector_markers_topic_ = get_parameter("reflector_markers_topic").as_string();
  projected_cloud_topic_ = get_parameter("projected_cloud_topic").as_string();
  volume_cloud_topic_ = get_parameter("volume_cloud_topic").as_string();
  reflector_class_id_ = get_parameter("reflector_class_id").as_string();
  volume_height_ = get_parameter("volume_height").as_double();
  ndt_score_threshold_ = get_parameter("ndt_score_threshold").as_double();

  // QoS 설정: BestEffort
  auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort();

  // Initialize subscribers
  yolo_sub_ = create_subscription<ultralytics_ros::msg::YoloResult>(
      yolo_topic_, qos, std::bind(&AdvancedReflectorDetectorNode::yoloCallback, this, std::placeholders::_1));
  lidar_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      lidar_topic_, qos, std::bind(&AdvancedReflectorDetectorNode::lidarCallback, this, std::placeholders::_1));
  camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      camera_info_topic_, qos, std::bind(&AdvancedReflectorDetectorNode::cameraInfoCallback, this, std::placeholders::_1));
  pose_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      pose_topic_, qos, std::bind(&AdvancedReflectorDetectorNode::poseCallback, this, std::placeholders::_1));

  // Initialize publishers
  roi_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(roi_cloud_topic_, qos);
  reflector_pose_pub_ = create_publisher<geometry_msgs::msg::PoseArray>(reflector_poses_topic_, qos);
  marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(reflector_markers_topic_, qos);
  projected_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(projected_cloud_topic_, qos);
  volume_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(volume_cloud_topic_, qos);

  // Initialize TF
  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // Initialize reflector model (simple cube for demonstration)
  reflector_model_ = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
  for (float x = -0.1; x <= 0.1; x += 0.02) {
    for (float y = -0.1; y <= 0.1; y += 0.02) {
      for (float z = -0.1; z <= 0.1; z += 0.02) {
        reflector_model_->push_back(pcl::PointXYZ(x, y, z));
      }
    }
  }
  reflector_model_->width = reflector_model_->size();
  reflector_model_->height = 1;
  reflector_model_->is_dense = true;

  RCLCPP_INFO(this->get_logger(), "Advanced Reflector Detector Node initialized");
}

void AdvancedReflectorDetectorNode::cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
  camera_info_ = *msg;
  camera_info_received_ = true;
  RCLCPP_INFO(this->get_logger(), "Received camera intrinsic parameters");
}

void AdvancedReflectorDetectorNode::yoloCallback(const ultralytics_ros::msg::YoloResult::SharedPtr msg) {
  yolo_result_ = msg;
  if (!msg->detections.detections.empty() && !msg->detections.detections[0].results.empty()) {
    RCLCPP_INFO(this->get_logger(), "YOLO detection class_id: %s",
                msg->detections.detections[0].results[0].hypothesis.class_id.c_str());
  }
}

void AdvancedReflectorDetectorNode::poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
  vehicle_pose_ = msg;
  RCLCPP_INFO(this->get_logger(), "Received vehicle pose");
}

void AdvancedReflectorDetectorNode::lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
  if (!camera_info_received_) {
    RCLCPP_WARN(this->get_logger(), "Camera info not received yet");
    return;
  }
  if (!vehicle_pose_) {
    RCLCPP_WARN(this->get_logger(), "Vehicle pose not received yet");
    return;
  }
  processPointCloud(msg, yolo_result_);
}

Eigen::Vector4f AdvancedReflectorDetectorNode::project2Dto3D(float u, float v, float depth) {
  float fx = camera_info_.k[0];
  float fy = camera_info_.k[4];
  float cx = camera_info_.k[2];
  float cy = camera_info_.k[5];

  Eigen::Vector4f point;
  point[0] = (u - cx) * depth / fx;
  point[1] = (v - cy) * depth / fy;
  point[2] = depth;
  point[3] = 1.0;
  return point;
}

void AdvancedReflectorDetectorNode::processPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr lidar_msg,
                                                     const ultralytics_ros::msg::YoloResult::SharedPtr yolo_msg) {
  if (!yolo_msg || yolo_msg->detections.detections.empty()) {
    RCLCPP_WARN(this->get_logger(), "No YOLO detections available");
    return;
  }

  // Convert ROS PointCloud2 to PCL
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(*lidar_msg, *cloud);

  // Downsample point cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
  voxel_grid.setInputCloud(cloud);
  voxel_grid.setLeafSize(0.05f, 0.05f, 0.05f); // 5cm resolution
  voxel_grid.filter(*downsampled_cloud);

  // Transform point cloud to camera frame
  geometry_msgs::msg::TransformStamped transform;
  try {
    transform = tf_buffer_->lookupTransform(camera_info_.header.frame_id, lidar_msg->header.frame_id,
                                           lidar_msg->header.stamp);
  } catch (tf2::TransformException &ex) {
    RCLCPP_ERROR(this->get_logger(), "TF lookup failed: %s", ex.what());
    return;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  Eigen::Matrix4f tf_matrix;
  tf2::fromMsg(transform.transform, tf_matrix);
  pcl::transformPointCloud(*downsampled_cloud, *transformed_cloud, tf_matrix);

  // Process each YOLO detection
  geometry_msgs::msg::PoseArray global_poses;
  global_poses.header = lidar_msg->header;
  global_poses.header.frame_id = "map";
  visualization_msgs::msg::MarkerArray markers;
  int marker_id = 0;

  for (const auto &detection : yolo_msg->detections.detections) {
    if (!detection.results.empty() && detection.results[0].hypothesis.class_id != reflector_class_id_) continue;

    // Define 3D ROI
    const auto &bbox = detection.bbox;
    float u_min = bbox.center.position.x - bbox.size_x / 2.0;
    float u_max = bbox.center.position.x + bbox.size_x / 2.0;
    float v_min = bbox.center.position.y - bbox.size_y / 2.0;
    float v_max = bbox.center.position.y + bbox.size_y / 2.0;
    float depth_min = 1.0;
    float depth_max = 50.0;

    Eigen::Vector4f p1 = project2Dto3D(u_min, v_min, depth_min);
    Eigen::Vector4f p2 = project2Dto3D(u_max, v_max, depth_max);

    pcl::CropBox<pcl::PointXYZ> crop_box;
    crop_box.setMin(Eigen::Vector4f(p1[0], p1[1], depth_min, 1.0));
    crop_box.setMax(Eigen::Vector4f(p2[0], p2[1], depth_max, 1.0));
    crop_box.setInputCloud(transformed_cloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr roi_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    crop_box.filter(*roi_cloud);

    // Publish ROI cloud
    sensor_msgs::msg::PointCloud2 roi_msg;
    pcl::toROSMsg(*roi_cloud, roi_msg);
    roi_msg.header = lidar_msg->header;
    roi_cloud_pub_->publish(roi_msg);

    // Cluster ROI cloud
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters;
    clusterPointCloud(roi_cloud, clusters);

    for (const auto &cluster : clusters) {
      // Estimate local pose
      geometry_msgs::msg::Pose local_pose;
      estimateReflectorPose(cluster, local_pose);

      // Transform to global frame
      geometry_msgs::msg::Pose global_pose;
      transformToGlobalFrame(local_pose, global_pose);
      global_poses.poses.push_back(global_pose);

      // Project to ground plane
      pcl::PointCloud<pcl::PointXYZ>::Ptr projected_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      projectToGroundPlane(cluster, projected_cloud);

      // Publish projected cloud
      sensor_msgs::msg::PointCloud2 projected_msg;
      pcl::toROSMsg(*projected_cloud, projected_msg);
      projected_msg.header = lidar_msg->header;
      projected_msg.header.frame_id = "map";
      projected_cloud_pub_->publish(projected_msg);

      // Estimate ground plane
      Eigen::Vector4f plane_coefficients;
      estimateGroundPlane(projected_cloud, plane_coefficients);

      // Expand volume
      pcl::PointCloud<pcl::PointXYZ>::Ptr volume_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      expandVolume(projected_cloud, volume_cloud, volume_height_);

      // Publish volume cloud
      sensor_msgs::msg::PointCloud2 volume_msg;
      pcl::toROSMsg(*volume_cloud, volume_msg);
      volume_msg.header = lidar_msg->header;
      volume_msg.header.frame_id = "map";
      volume_cloud_pub_->publish(volume_msg);

      // Match with model
      float score;
      if (matchWithModel(volume_cloud, score) && score > ndt_score_threshold_) {
        // Create visualization marker
        visualization_msgs::msg::Marker marker;
        marker.header = lidar_msg->header;
        marker.header.frame_id = "map";
        marker.ns = "reflectors";
        marker.id = marker_id++;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose = global_pose;
        marker.scale.x = 0.2;
        marker.scale.y = 0.2;
        marker.scale.z = 0.2;
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        marker.color.a = 1.0;
        markers.markers.push_back(marker);
      }
    }
  }

  reflector_pose_pub_->publish(global_poses);
  marker_pub_->publish(markers);
}

void AdvancedReflectorDetectorNode::transformToGlobalFrame(const geometry_msgs::msg::Pose &local_pose,
                                                          geometry_msgs::msg::Pose &global_pose) {
  geometry_msgs::msg::TransformStamped transform;
  try {
    transform = tf_buffer_->lookupTransform("map", camera_info_.header.frame_id,
                                           vehicle_pose_->header.stamp);
  } catch (tf2::TransformException &ex) {
    RCLCPP_ERROR(this->get_logger(), "Global TF lookup failed: %s", ex.what());
    return;
  }

  tf2::doTransform(local_pose, global_pose, transform);
}

void AdvancedReflectorDetectorNode::projectToGroundPlane(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                                        pcl::PointCloud<pcl::PointXYZ>::Ptr &projected_cloud) {
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  coefficients->values.resize(4);
  coefficients->values[0] = 0.0;
  coefficients->values[1] = 0.0;
  coefficients->values[2] = 1.0;
  coefficients->values[3] = 0.0;

  pcl::ProjectInliers<pcl::PointXYZ> proj;
  proj.setModelType(pcl::SACMODEL_PLANE);
  proj.setInputCloud(cloud);
  proj.setModelCoefficients(coefficients);
  proj.filter(*projected_cloud);
}

void AdvancedReflectorDetectorNode::estimateGroundPlane(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                                       Eigen::Vector4f &plane_coefficients) {
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setDistanceThreshold(0.1);
  seg.setInputCloud(cloud);
  seg.segment(*inliers, *coefficients);

  plane_coefficients[0] = coefficients->values[0];
  plane_coefficients[1] = coefficients->values[1];
  plane_coefficients[2] = coefficients->values[2];
  plane_coefficients[3] = coefficients->values[3];
}

void AdvancedReflectorDetectorNode::expandVolume(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                                pcl::PointCloud<pcl::PointXYZ>::Ptr &volume_cloud,
                                                float height) {
  Eigen::Vector4f min_pt, max_pt;
  pcl::getMinMax3D(*cloud, min_pt, max_pt);

  pcl::CropBox<pcl::PointXYZ> crop_box;
  crop_box.setMin(Eigen::Vector4f(min_pt[0], min_pt[1], min_pt[2] - height / 2, 1.0));
  crop_box.setMax(Eigen::Vector4f(max_pt[0], max_pt[1], max_pt[2] + height / 2, 1.0));
  crop_box.setInputCloud(cloud);
  crop_box.filter(*volume_cloud);
}

bool AdvancedReflectorDetectorNode::matchWithModel(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                                  float &score) {
  pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
  ndt.setTransformationEpsilon(0.01);
  ndt.setStepSize(0.1);
  ndt.setResolution(0.1);
  ndt.setInputSource(cloud);
  ndt.setInputTarget(reflector_model_);
  Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();
  pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  ndt.align(*aligned_cloud, init_guess);

  score = ndt.getFitnessScore();
  return ndt.hasConverged();
}

void AdvancedReflectorDetectorNode::clusterPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                                     std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &clusters) {
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud(cloud);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance(0.1);
  ec.setMinClusterSize(10);
  ec.setMaxClusterSize(1000);
  ec.setSearchMethod(tree);
  ec.setInputCloud(cloud);
  ec.extract(cluster_indices);

  for (const auto &indices : cluster_indices) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto &idx : indices.indices) {
      cluster->push_back((*cloud)[idx]);
    }
    cluster->width = cluster->size();
    cluster->height = 1;
    cluster->is_dense = true;
    clusters.push_back(cluster);
  }
}

void AdvancedReflectorDetectorNode::estimateReflectorPose(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cluster,
                                                         geometry_msgs::msg::Pose &pose) {
  Eigen::Vector4f centroid;
  pcl::compute3DCentroid(*cluster, centroid);

  pose.position.x = centroid[0];
  pose.position.y = centroid[1];
  pose.position.z = centroid[2];
  pose.orientation.w = 1.0;
}

}  // namespace advanced_reflector_detector

RCLCPP_COMPONENTS_REGISTER_NODE(advanced_reflector_detector::AdvancedReflectorDetectorNode)