/**
 * MPC Local Planner for AMMR
 *
 * 架構：
 *   - 線性化單輪車模型（局部線性化於當前狀態）
 *   - 濃縮式 QP 配方（condensed formulation）
 *   - OSQP（ROS2 Jazzy 版本 API：OSQPWorkspace / OSQPData / csc）
 *   - 反應式安全層（依 costmap 障礙物距離限制速度上限）
 */

#include "ammr_navigation/mpc_controller.hpp"

// OSQP 完整標頭（含 c_float / c_int / csc / OSQPWorkspace）
#include <osqp/osqp.h>
#include <osqp/cs.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "nav2_costmap_2d/cost_values.hpp"
#include "nav2_util/node_utils.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "tf2/utils.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

PLUGINLIB_EXPORT_CLASS(ammr_navigation::MPCController, nav2_core::Controller)

namespace ammr_navigation
{

// ─────────────────────────────────────────────────────────────────────────────
// 內部輔助：Eigen dense → OSQP csc 格式
// ─────────────────────────────────────────────────────────────────────────────

/**
 * 持有 OSQP csc 矩陣所需的 C 陣列記憶體。
 * 必須在 osqp_setup 期間保持存活。
 */
struct OsqpCsc {
  std::vector<c_float> x;   // 非零值
  std::vector<c_int>   i;   // row indices
  std::vector<c_int>   p;   // col pointers (size = n_cols + 1)
  csc                  mat;

  /**
   * @param M                要轉換的 Eigen 矩陣
   * @param upper_tri_only   true → 只儲存上三角（對稱矩陣 P 用）
   */
  void build(const Eigen::MatrixXd & M, bool upper_tri_only = false)
  {
    const int rows = static_cast<int>(M.rows());
    const int cols = static_cast<int>(M.cols());
    x.clear();
    i.clear();
    p.clear();
    p.push_back(0);

    for (int col = 0; col < cols; ++col) {
      for (int row = 0; row < rows; ++row) {
        if (upper_tri_only && row > col) {
          continue;
        }
        if (std::abs(M(row, col)) > 1e-12) {
          x.push_back(static_cast<c_float>(M(row, col)));
          i.push_back(static_cast<c_int>(row));
        }
      }
      p.push_back(static_cast<c_int>(x.size()));
    }

    std::memset(&mat, 0, sizeof(mat));
    mat.m    = static_cast<c_int>(rows);
    mat.n    = static_cast<c_int>(cols);
    mat.nzmax = static_cast<c_int>(x.size());
    mat.nz   = -1;   // -1 表示 CSC 格式
    mat.x    = x.empty() ? nullptr : x.data();
    mat.i    = i.empty() ? nullptr : i.data();
    mat.p    = p.data();
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// 工具函式
// ─────────────────────────────────────────────────────────────────────────────

static inline double normalizeAngle(double a)
{
  while (a >  M_PI) {a -= 2.0 * M_PI;}
  while (a < -M_PI) {a += 2.0 * M_PI;}
  return a;
}

// ─────────────────────────────────────────────────────────────────────────────
// Nav2 Lifecycle
// ─────────────────────────────────────────────────────────────────────────────

void MPCController::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  node_        = parent;
  plugin_name_ = name;
  tf_          = tf;
  costmap_ros_ = costmap_ros;

  auto node = node_.lock();
  logger_ = node->get_logger();

  auto declare_int = [&](const std::string & pname, int def, int & out) {
    nav2_util::declare_parameter_if_not_declared(
      node, plugin_name_ + "." + pname, rclcpp::ParameterValue(def));
    node->get_parameter(plugin_name_ + "." + pname, out);
  };
  auto declare_dbl = [&](const std::string & pname, double def, double & out) {
    nav2_util::declare_parameter_if_not_declared(
      node, plugin_name_ + "." + pname, rclcpp::ParameterValue(def));
    node->get_parameter(plugin_name_ + "." + pname, out);
  };

  declare_int("horizon",   10,    horizon_);
  declare_dbl("dt",        0.1,   dt_);
  declare_dbl("v_max",     0.5,   v_max_);
  declare_dbl("v_min",    -0.1,   v_min_);
  declare_dbl("w_max",     1.0,   w_max_);
  declare_dbl("a_max",     1.0,   a_max_);
  declare_dbl("az_max",    2.0,   az_max_);
  declare_dbl("Q_pos",    10.0,   Q_pos_);
  declare_dbl("Q_theta",   5.0,   Q_theta_);
  declare_dbl("R_v",       0.1,   R_v_);
  declare_dbl("R_w",       0.1,   R_w_);

  RCLCPP_INFO(logger_,
    "[MPCController] configured — horizon=%d  dt=%.2f  v_max=%.2f  w_max=%.2f",
    horizon_, dt_, v_max_, w_max_);
}

void MPCController::cleanup()
{
  if (osqp_solver_) {
    osqp_cleanup(osqp_solver_);
    osqp_solver_ = nullptr;
  }
  RCLCPP_INFO(logger_, "[MPCController] cleaned up");
}

void MPCController::activate()   {}
void MPCController::deactivate() {}

void MPCController::setPlan(const nav_msgs::msg::Path & path)
{
  global_plan_ = path;
}

void MPCController::setSpeedLimit(const double & speed_limit, const bool & percentage)
{
  if (percentage) {
    speed_limit_ = std::clamp(speed_limit / 100.0, 0.0, 1.0);
  } else {
    speed_limit_ = (v_max_ > 1e-6) ? std::clamp(speed_limit / v_max_, 0.0, 1.0) : 1.0;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 主控制迴圈
// ─────────────────────────────────────────────────────────────────────────────

geometry_msgs::msg::TwistStamped MPCController::computeVelocityCommands(
  const geometry_msgs::msg::PoseStamped & pose,
  const geometry_msgs::msg::Twist & velocity,
  nav2_core::GoalChecker * /*goal_checker*/)
{
  geometry_msgs::msg::TwistStamped cmd;
  cmd.header.frame_id = "base_footprint";
  cmd.header.stamp    = pose.header.stamp;

  // ── 轉換 pose 到全域路徑所在的 frame（通常 map）────────────────────────────
  // Controller server 給的 pose 是 local costmap 的 global_frame（odom），
  // 而全域路徑是 map frame，必須轉換後才能比較座標。
  geometry_msgs::msg::PoseStamped pose_in_plan_frame = pose;
  if (!global_plan_.poses.empty() &&
      global_plan_.header.frame_id != pose.header.frame_id)
  {
    try {
      tf_->transform(pose, pose_in_plan_frame,
                     global_plan_.header.frame_id,
                     tf2::Duration(std::chrono::milliseconds(100)));
    } catch (tf2::TransformException & ex) {
      RCLCPP_WARN(logger_, "[MPC] TF transform failed: %s", ex.what());
      return cmd;
    }
  }

  auto reference = getLocalReference(pose_in_plan_frame, horizon_);
  if (reference.empty()) {
    RCLCPP_WARN(logger_, "[MPC] 參考軌跡為空，輸出零速度");
    return cmd;
  }

  auto twist = solveQP(pose_in_plan_frame, velocity, reference);

  // 反應式安全層（用已轉換的 pose，或直接用原始 pose 也可因為只查 costmap 距離）
  double safe_v = computeSafeSpeedLimit(pose_in_plan_frame);
  safe_v = std::min(safe_v, v_max_ * speed_limit_);

  twist.linear.x  = std::clamp(twist.linear.x,  v_min_, safe_v);
  twist.angular.z = std::clamp(twist.angular.z, -w_max_, w_max_);

  // Debug log（1Hz）
  RCLCPP_INFO_THROTTLE(logger_, *rclcpp::Clock::make_shared(), 1000,
    "[MPC] pose[%s]=(%.2f,%.2f,%.2f°) ref0=(%.2f,%.2f,%.2f°) cmd v=%.3f w=%.3f safe_v=%.3f",
    pose_in_plan_frame.header.frame_id.c_str(),
    pose_in_plan_frame.pose.position.x, pose_in_plan_frame.pose.position.y,
    tf2::getYaw(pose_in_plan_frame.pose.orientation) * 180.0 / M_PI,
    reference[0].pose.position.x, reference[0].pose.position.y,
    tf2::getYaw(reference[0].pose.orientation) * 180.0 / M_PI,
    twist.linear.x, twist.angular.z, safe_v);

  cmd.twist = twist;
  return cmd;
}

// ─────────────────────────────────────────────────────────────────────────────
// 局部參考軌跡提取
// ─────────────────────────────────────────────────────────────────────────────

std::vector<geometry_msgs::msg::PoseStamped> MPCController::getLocalReference(
  const geometry_msgs::msg::PoseStamped & robot_pose,
  int horizon)
{
  if (global_plan_.poses.empty()) {
    return {};
  }

  const double rx = robot_pose.pose.position.x;
  const double ry = robot_pose.pose.position.y;

  // ── 找最近路徑點（向前搜尋，避免選到身後點）──────────────────────────────
  double min_dist_sq = std::numeric_limits<double>::max();
  size_t closest = 0;
  for (size_t k = 0; k < global_plan_.poses.size(); ++k) {
    double dx = global_plan_.poses[k].pose.position.x - rx;
    double dy = global_plan_.poses[k].pose.position.y - ry;
    double d  = dx * dx + dy * dy;
    if (d < min_dist_sq) {
      min_dist_sq = d;
      closest     = k;
    }
  }

  // ── 沿路徑以固定弧長間隔取 N 個參考點 ────────────────────────────────────
  // 每步前進距離 step_dist；idx 只往前走，不倒退
  std::vector<geometry_msgs::msg::PoseStamped> ref;
  ref.reserve(horizon);

  const double step_dist = std::max(v_max_ * dt_, 0.05);
  size_t idx = closest;

  for (int k = 0; k < horizon; ++k) {
    // 前進 step_dist 弧長
    double remaining = step_dist;
    while (remaining > 1e-6 && idx + 1 < global_plan_.poses.size()) {
      const auto & cur  = global_plan_.poses[idx];
      const auto & next = global_plan_.poses[idx + 1];
      double dx = next.pose.position.x - cur.pose.position.x;
      double dy = next.pose.position.y - cur.pose.position.y;
      double d  = std::sqrt(dx * dx + dy * dy);
      if (d <= remaining) {
        remaining -= d;
        ++idx;
      } else {
        break;  // 剩餘距離不足一個 segment
      }
    }
    ref.push_back(global_plan_.poses[idx]);
  }

  // 不足則以終點填補
  while (static_cast<int>(ref.size()) < horizon) {
    ref.push_back(global_plan_.poses.back());
  }

  // ── 從路徑方向計算參考 yaw（NavFn 不設置 waypoint orientation）────────────
  // 使用相鄰參考點的方向向量，讓 MPC 追蹤正確航向
  for (int k = 0; k + 1 < static_cast<int>(ref.size()); ++k) {
    double dx = ref[k + 1].pose.position.x - ref[k].pose.position.x;
    double dy = ref[k + 1].pose.position.y - ref[k].pose.position.y;
    if (dx * dx + dy * dy > 1e-6) {
      tf2::Quaternion q;
      q.setRPY(0.0, 0.0, std::atan2(dy, dx));
      ref[k].pose.orientation = tf2::toMsg(q);
    }
  }
  // 最後一點繼承倒數第二點的 yaw
  if (ref.size() >= 2) {
    ref.back().pose.orientation = ref[ref.size() - 2].pose.orientation;
  }

  return ref;
}

// ─────────────────────────────────────────────────────────────────────────────
// QP 求解（濃縮式 MPC + OSQP）
// ─────────────────────────────────────────────────────────────────────────────
//
// 狀態  X  = [x, y, θ]    (Nx = 3)
// 控制  U  = [v, ω]        (Nu = 2)
//
// 離散化線性化單輪車：X_{k+1} = A*X_k + B*u_k
//   A = I + dt*[[0,0,-v_r sin θ₀],[0,0,v_r cos θ₀],[0,0,0]]
//   B = dt*[[cos θ₀,0],[sin θ₀,0],[0,1]]
//
// 濃縮預測：X_bar = Sx*X₀ + Su*U_bar
//
// 成本：J = (X_bar-X_ref)ᵀ Q̄ (X_bar-X_ref) + U_barᵀ R̄ U_bar
//
// OSQP 最小化 ½xᵀPx + qᵀx：
//   P = Suᵀ Q̄ Su + R̄
//   q = Suᵀ Q̄ (Sx X₀ - X_ref)

geometry_msgs::msg::Twist MPCController::solveQP(
  const geometry_msgs::msg::PoseStamped & robot_pose,
  const geometry_msgs::msg::Twist & current_vel,
  const std::vector<geometry_msgs::msg::PoseStamped> & reference)
{
  geometry_msgs::msg::Twist zero;

  const int N  = horizon_;
  const int Nx = 3;
  const int Nu = 2;

  // ── 1. 當前狀態 ──────────────────────────────────────────────────────────
  const double x0  = robot_pose.pose.position.x;
  const double y0  = robot_pose.pose.position.y;
  const double th0 = tf2::getYaw(robot_pose.pose.orientation);

  Eigen::VectorXd X0(Nx);
  X0 << x0, y0, th0;

  // ── 2. 參考狀態向量 ──────────────────────────────────────────────────────
  Eigen::VectorXd X_ref(Nx * N);
  for (int k = 0; k < N; ++k) {
    double th_r = tf2::getYaw(reference[k].pose.orientation);
    th_r = th0 + normalizeAngle(th_r - th0);
    X_ref(Nx * k + 0) = reference[k].pose.position.x;
    X_ref(Nx * k + 1) = reference[k].pose.position.y;
    X_ref(Nx * k + 2) = th_r;
  }

  // ── 3. 線性化（以當前狀態和速度為展開點） ────────────────────────────────
  const double v_r = (std::abs(current_vel.linear.x) > 0.05)
                     ? current_vel.linear.x : 0.2;

  Eigen::Matrix3d A = Eigen::Matrix3d::Identity();
  A(0, 2) = -v_r * std::sin(th0) * dt_;
  A(1, 2) =  v_r * std::cos(th0) * dt_;

  Eigen::MatrixXd B(Nx, Nu);
  B << std::cos(th0) * dt_,  0.0,
       std::sin(th0) * dt_,  0.0,
       0.0,                   dt_;

  // ── 4. 濃縮矩陣 Sx, Su ──────────────────────────────────────────────────
  Eigen::MatrixXd Sx(Nx * N, Nx);
  Eigen::MatrixXd Su = Eigen::MatrixXd::Zero(Nx * N, Nu * N);

  {
    Eigen::Matrix3d Apow = A;
    for (int k = 0; k < N; ++k) {
      Sx.block(Nx * k, 0, Nx, Nx) = Apow;

      Eigen::Matrix3d Aj = Eigen::Matrix3d::Identity();
      for (int j = 0; j <= k; ++j) {
        Su.block(Nx * k, Nu * (k - j), Nx, Nu) = Aj * B;
        Aj = Aj * A;
      }
      Apow = Apow * A;
    }
  }

  // ── 5. 成本矩陣 ─────────────────────────────────────────────────────────
  Eigen::MatrixXd Q_bar = Eigen::MatrixXd::Zero(Nx * N, Nx * N);
  for (int k = 0; k < N; ++k) {
    Q_bar(Nx * k + 0, Nx * k + 0) = Q_pos_;
    Q_bar(Nx * k + 1, Nx * k + 1) = Q_pos_;
    Q_bar(Nx * k + 2, Nx * k + 2) = Q_theta_;
  }

  Eigen::MatrixXd R_bar = Eigen::MatrixXd::Zero(Nu * N, Nu * N);
  for (int k = 0; k < N; ++k) {
    R_bar(Nu * k + 0, Nu * k + 0) = R_v_;
    R_bar(Nu * k + 1, Nu * k + 1) = R_w_;
  }

  const Eigen::MatrixXd SuTQ    = Su.transpose() * Q_bar;
  Eigen::MatrixXd       P_dense = SuTQ * Su + R_bar;
  P_dense = 0.5 * (P_dense + P_dense.transpose());  // 強制對稱

  const Eigen::VectorXd E0    = Sx * X0 - X_ref;
  const Eigen::VectorXd q_vec = SuTQ * E0;

  // ── 6. 約束矩陣 ──────────────────────────────────────────────────────────
  const int n_vars = Nu * N;
  const int n_box  = Nu * N;
  const int n_acc  = Nu * N;
  const int n_con  = n_box + n_acc;

  Eigen::MatrixXd A_con = Eigen::MatrixXd::Zero(n_con, n_vars);
  Eigen::VectorXd l_con(n_con), u_con(n_con);

  // (a) 速度箱型約束
  A_con.block(0, 0, n_box, n_vars) = Eigen::MatrixXd::Identity(n_box, n_vars);
  for (int k = 0; k < N; ++k) {
    l_con(Nu * k + 0) = v_min_;   u_con(Nu * k + 0) = v_max_;
    l_con(Nu * k + 1) = -w_max_;  u_con(Nu * k + 1) =  w_max_;
  }

  // (b) 加速度約束
  const double v_cur  = current_vel.linear.x;
  const double w_cur  = current_vel.angular.z;
  const double dv_max = a_max_  * dt_;
  const double dw_max = az_max_ * dt_;

  // 第 0 步相對於當前速度
  A_con(n_box + 0, 0) = 1.0;
  l_con(n_box + 0)    = v_cur - dv_max;
  u_con(n_box + 0)    = v_cur + dv_max;

  A_con(n_box + 1, 1) = 1.0;
  l_con(n_box + 1)    = w_cur - dw_max;
  u_con(n_box + 1)    = w_cur + dw_max;

  // 第 1..N-1 步相對於前一步
  for (int k = 1; k < N; ++k) {
    const int rv = n_box + Nu * k + 0;
    const int rw = n_box + Nu * k + 1;

    A_con(rv, Nu * k + 0)     =  1.0;
    A_con(rv, Nu * (k-1) + 0) = -1.0;
    l_con(rv) = -dv_max;  u_con(rv) = dv_max;

    A_con(rw, Nu * k + 1)     =  1.0;
    A_con(rw, Nu * (k-1) + 1) = -1.0;
    l_con(rw) = -dw_max;  u_con(rw) = dw_max;
  }

  // ── 7. 轉換為 CSC 並建立 OSQPData ────────────────────────────────────────
  OsqpCsc P_csc, A_csc;
  P_csc.build(P_dense, /*upper_tri_only=*/true);
  A_csc.build(A_con,   /*upper_tri_only=*/false);

  std::vector<c_float> q_osqp(q_vec.data(), q_vec.data() + n_vars);
  std::vector<c_float> l_osqp(l_con.data(), l_con.data() + n_con);
  std::vector<c_float> u_osqp(u_con.data(), u_con.data() + n_con);

  OSQPData data;
  data.n = static_cast<c_int>(n_vars);
  data.m = static_cast<c_int>(n_con);
  data.P = &P_csc.mat;
  data.A = &A_csc.mat;
  data.q = q_osqp.data();
  data.l = l_osqp.data();
  data.u = u_osqp.data();

  // ── 8. OSQP 求解 ─────────────────────────────────────────────────────────
  if (osqp_solver_) {
    osqp_cleanup(osqp_solver_);
    osqp_solver_ = nullptr;
  }

  OSQPSettings settings;
  osqp_set_default_settings(&settings);
  settings.verbose    = 0;
  settings.warm_start = 0;
  settings.max_iter   = 400;
  settings.eps_abs    = static_cast<c_float>(1e-4);
  settings.eps_rel    = static_cast<c_float>(1e-4);

  const c_int exit_flag = osqp_setup(&osqp_solver_, &data, &settings);

  if (exit_flag != 0) {
    RCLCPP_WARN(logger_, "[MPC] OSQP setup 失敗（code=%d），輸出零速度",
                static_cast<int>(exit_flag));
    return zero;
  }

  osqp_solve(osqp_solver_);

  const c_int status = osqp_solver_->info->status_val;
  if (status != OSQP_SOLVED && status != OSQP_SOLVED_INACCURATE) {
    RCLCPP_WARN(logger_, "[MPC] OSQP 未收斂：%s", osqp_solver_->info->status);
    return zero;
  }

  // ── 9. 取出第一步最優控制 ─────────────────────────────────────────────
  geometry_msgs::msg::Twist result;
  result.linear.x  = static_cast<double>(osqp_solver_->solution->x[0]);
  result.angular.z = static_cast<double>(osqp_solver_->solution->x[1]);

  return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// 反應式安全層
// ─────────────────────────────────────────────────────────────────────────────

double MPCController::computeSafeSpeedLimit(
  const geometry_msgs::msg::PoseStamped & robot_pose)
{
  auto * costmap = costmap_ros_->getCostmap();
  if (!costmap) {return v_max_;}

  // 只偵測機器人前方 ±90° 的障礙物（後方不限速）
  const double check_radius = 0.6;
  const int    n_ray        = 12;    // 前方半圓，每 15°
  const double r_step       = 0.05;

  const double rx      = robot_pose.pose.position.x;
  const double ry      = robot_pose.pose.position.y;
  const double heading = tf2::getYaw(robot_pose.pose.orientation);

  double min_obs_dist = check_radius;

  for (int ray = 0; ray < n_ray; ++ray) {
    // 前方 ±90°：從 heading-π/2 到 heading+π/2
    const double angle = heading - M_PI_2 + M_PI * ray / (n_ray - 1);
    for (double r = r_step; r <= check_radius; r += r_step) {
      const double wx = rx + r * std::cos(angle);
      const double wy = ry + r * std::sin(angle);
      unsigned int mx, my;
      if (!costmap->worldToMap(wx, wy, mx, my)) {break;}
      if (costmap->getCost(mx, my) >= nav2_costmap_2d::LETHAL_OBSTACLE) {
        if (r < min_obs_dist) {min_obs_dist = r;}
        break;
      }
    }
  }

  // 平滑限速：check_radius 以外全速，接近 0 才趨近 0
  const double ratio = std::clamp(min_obs_dist / check_radius, 0.0, 1.0);
  return ratio * v_max_;
}

}  // namespace ammr_navigation
