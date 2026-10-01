#ifndef XIAOBEI_HARDWARE__XIAOBEI_HARDWARE_INTERFACE_HPP_
#define XIAOBEI_HARDWARE__XIAOBEI_HARDWARE_INTERFACE_HPP_

#include "hardware_interface/handle.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "rclcpp/macros.hpp"
#include "rclcpp_lifecycle/node_interfaces/lifecycle_node_interface.hpp"
#include "rclcpp/rclcpp.hpp"

#include <memory>
#include <string>
#include <vector>

namespace xiaobei_hardware
{
class XiaobeiHardwareInterface : public hardware_interface::SystemInterface
{
public:
  RCLCPP_SHARED_PTR_DEFINITIONS(XiaobeiHardwareInterface)

  // 初始化：读取URDF参数，配置串口
  hardware_interface::CallbackReturn on_init(
    const hardware_interface::HardwareInfo & info) override;

  // 导出状态接口（读：位置、速度）
  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;

  // 导出命令接口（写：目标位置）
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

  // 激活（开始控制）
  hardware_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;

  // 停用（停止控制）
  hardware_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  // 核心循环：读取舵机真实位置
  hardware_interface::return_type read(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

  // 核心循环：写入目标位置给舵机
  hardware_interface::return_type write(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  // 串口参数
  std::string serial_port_;
  int baud_rate_;

  // 存储关节数据的变量
  // commands: MoveIt 发过来的目标位置
  std::vector<double> hw_commands_;
  
  // states: 从舵机读回来的真实位置和速度
  std::vector<double> hw_states_position_;
  std::vector<double> hw_states_velocity_;
};

}  // namespace xiaobei_hardware

#endif  // XIAOBEI_HARDWARE__XIAOBEI_HARDWARE_INTERFACE_HPP_