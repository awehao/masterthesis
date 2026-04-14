#pragma once

#include <memory>
#include <string>
#include <vector>

#include "nav2_core/controller.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "tf2_ros/buffer.h"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"

#include "nav2_core/goal_checker.hpp"

#include <Eigen/Dense>
#include <osqp/osqp.h>

namespace ammr_navigation
{

/**
 * MPC Local Planner for AMMR
 *
 * 系統架構（來自論文 mpc.txt）：
 * 1. 底盤運動學模型（單輪車模型，局部線性化）
 * 2. OSQP 求解 QP 問題
 * 3. 反應式安全層（剔除高風險速度）
 * 4. [TODO] Kalman filter 障礙物預測
 */
class MPCController : public nav2_core::Controller
{
public:
  MPCController() = default;
  ~MPCController() override = default;

  // ---- Nav2 Controller 介面 ----
  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name,
    std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

  void cleanup() override;
  void activate() override;
  void deactivate() override;

  void setPlan(const nav_msgs::msg::Path & path) override;

  geometry_msgs::msg::TwistStamped computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped & pose,
    const geometry_msgs::msg::Twist & velocity,
    nav2_core::GoalChecker * goal_checker) override;

  void setSpeedLimit(const double & speed_limit, const bool & percentage) override;

private:
  // ---- MPC 核心 ----

  /**
   * 在預測時域內找到全域路徑上的參考點序列
   */
  std::vector<geometry_msgs::msg::PoseStamped> getLocalReference(
    const geometry_msgs::msg::PoseStamped & robot_pose,
    int horizon);

  /**
   * 建立並求解 QP 問題
   * 狀態：[x, y, θ]，控制：[v, ω]
   * 回傳：最優控制序列第一步
   */
  geometry_msgs::msg::Twist solveQP(
    const geometry_msgs::msg::PoseStamped & robot_pose,
    const geometry_msgs::msg::Twist & current_vel,
    const std::vector<geometry_msgs::msg::PoseStamped> & reference);

  /**
   * 反應式安全層：依障礙物距離限制速度上限
   */
  double computeSafeSpeedLimit(
    const geometry_msgs::msg::PoseStamped & robot_pose);

  // ---- 狀態 ----
  rclcpp_lifecycle::LifecycleNode::WeakPtr node_;
  rclcpp::Logger logger_{rclcpp::get_logger("MPCController")};
  std::string plugin_name_;
  std::shared_ptr<tf2_ros::Buffer> tf_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  nav_msgs::msg::Path global_plan_;

  // ---- MPC 參數 ----
  int    horizon_;          // 預測時域步數 N
  double dt_;               // 控制週期 (s)
  double v_max_;            // 最大線速度
  double v_min_;            // 最小線速度（負值允許倒退）
  double w_max_;            // 最大角速度
  double a_max_;            // 最大線加速度
  double az_max_;           // 最大角加速度
  double speed_limit_{1.0}; // 外部速度限制（0~1）

  // ---- Cost 權重 ----
  double Q_pos_;    // 位置誤差權重
  double Q_theta_;  // 航向誤差權重
  double R_v_;      // 線速度控制量代價
  double R_w_;      // 角速度控制量代價

  // ---- OSQP ----
  OSQPWorkspace * osqp_solver_{nullptr};
};

}  // namespace ammr_navigation
