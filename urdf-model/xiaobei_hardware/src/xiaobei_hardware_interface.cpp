#include "xiaobei_hardware/xiaobei_hardware_interface.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "rclcpp/rclcpp.hpp"
#include "scservo_sdk/SCServo.h" 
#include <map>

namespace xiaobei_hardware
{

SMS_STS st; 

// =================================================================
//                      机器人关节校准表
// =================================================================
struct JointConfig {
    int id;             // 舵机真实ID
    double offset;      // 零点偏移 (摆正时舵机的脉冲数，默认2048)
    double direction;   // 方向: 1.0 (正向) 或 -1.0 (反向)
};

std::map<std::string, JointConfig> joint_map;

// 根据你的描述生成的预设配置
void init_joint_config() {
    // ⚠️ 注意：这里的配置是基于你的描述推测的。
    // 如果某个关节运动反了，请把对应的 1.0 改成 -1.0 (或者反过来)
    // 如果关节初始位置歪了，请修改 2048.0 这个数值

    // === 左臂 (Left Arm) IDs: 1-6 ===
    // 描述: ID1(前+), ID2(里+), ID3(外旋+), ID4(前+), ID5(外旋+), ID6(外+)
    joint_map["1"] = {1, 2048.0, 1.0};  // 左肩前后
    joint_map["2"] = {2, 2048.0, 1.0};  // 左肩左右
    joint_map["3"] = {3, 2048.0, 1.0};  // 左大臂旋转
    joint_map["4"] = {4, 2048.0, 1.0};  // 左肘
    joint_map["5"] = {5, 2048.0, 1.0};  // 左小臂旋转
    joint_map["6"] = {6, 2048.0, 1.0};  // 左手腕
    joint_map["7"] = {7, 2048.0, 1.0};  // 左夹爪

    // === 右臂 (Right Arm) IDs: 51-56 ===
    // 描述: ID51(后+), ID52(外+), ID53(里旋+), ID54(后+), ID55(里旋+), ID56(里+)
    // 注意：这里的方向描述和左臂很多是反的，所以我预设为 -1.0
    joint_map["51"] = {51, 2048.0, -1.0}; // 右肩前后 (预设反向)
    joint_map["52"] = {52, 2048.0, -1.0}; // 右肩左右 (预设反向)
    joint_map["53"] = {53, 2048.0, -1.0}; // 右大臂旋转
    joint_map["54"] = {54, 2048.0, -1.0}; // 右肘
    joint_map["55"] = {55, 2048.0, -1.0}; // 右小臂旋转
    joint_map["56"] = {56, 2048.0, -1.0}; // 右手腕
    joint_map["57"] = {57, 2048.0, -1.0}; // 右夹爪

    // === 头部 (Head) ===
    joint_map["101"] = {101, 2048.0, 1.0};
    joint_map["102"] = {102, 2048.0, 1.0};
}

hardware_interface::CallbackReturn XiaobeiHardwareInterface::on_init(
  const hardware_interface::HardwareInfo & info)
{
  if (hardware_interface::SystemInterface::on_init(info) !=
      hardware_interface::CallbackReturn::SUCCESS)
  {
    return hardware_interface::CallbackReturn::ERROR;
  }

  // 初始化校准表
  init_joint_config();

  serial_port_ = info_.hardware_parameters["serial_port"];
  baud_rate_ = std::stoi(info_.hardware_parameters["baud_rate"]);

  hw_states_position_.resize(info_.joints.size(), 0.0);
  hw_states_velocity_.resize(info_.joints.size(), 0.0);
  hw_commands_.resize(info_.joints.size(), 0.0);

  RCLCPP_INFO(rclcpp::get_logger("XiaobeiHardware"), "Opening Port: %s at %d", serial_port_.c_str(), baud_rate_);
  
  if(!st.begin(baud_rate_, serial_port_.c_str())){
      RCLCPP_ERROR(rclcpp::get_logger("XiaobeiHardware"), "Failed to open serial port!");
      return hardware_interface::CallbackReturn::ERROR;
  }

  return hardware_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface> XiaobeiHardwareInterface::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> state_interfaces;
  for (uint i = 0; i < info_.joints.size(); i++)
  {
    state_interfaces.emplace_back(hardware_interface::StateInterface(
      info_.joints[i].name, hardware_interface::HW_IF_POSITION, &hw_states_position_[i]));
    state_interfaces.emplace_back(hardware_interface::StateInterface(
      info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &hw_states_velocity_[i]));
  }
  return state_interfaces;
}

std::vector<hardware_interface::CommandInterface> XiaobeiHardwareInterface::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> command_interfaces;
  for (uint i = 0; i < info_.joints.size(); i++)
  {
    command_interfaces.emplace_back(hardware_interface::CommandInterface(
      info_.joints[i].name, hardware_interface::HW_IF_POSITION, &hw_commands_[i]));
  }
  return command_interfaces;
}

hardware_interface::CallbackReturn XiaobeiHardwareInterface::on_activate(const rclcpp_lifecycle::State & /*previous_state*/)
{
  RCLCPP_INFO(rclcpp::get_logger("XiaobeiHardware"), "Activating... Enabling Torque.");
  for (auto const& [name, config] : joint_map) {
      st.EnableTorque(config.id, 1);
  }
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn XiaobeiHardwareInterface::on_deactivate(const rclcpp_lifecycle::State & /*previous_state*/)
{
  RCLCPP_INFO(rclcpp::get_logger("XiaobeiHardware"), "Deactivating...");
  for (auto const& [name, config] : joint_map) {
      st.EnableTorque(config.id, 0);
  }
  st.end();
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::return_type XiaobeiHardwareInterface::read(const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  for (std::size_t i = 0; i < info_.joints.size(); i++)
  {
      std::string joint_name = info_.joints[i].name;
      
      // 如果URDF里的关节名不在我们的表里，尝试用数字解析，或者跳过
      int id_temp = 0;
      double offset = 2048.0;
      double direction = 1.0;

      if (joint_map.find(joint_name) != joint_map.end()) {
          id_temp = joint_map[joint_name].id;
          offset = joint_map[joint_name].offset;
          direction = joint_map[joint_name].direction;
      } else {
          // Fallback: 如果没有配置，尝试直接解析名字
          try {
              id_temp = std::stoi(joint_name);
          } catch (...) { continue; }
      }

      // 读取位置
      int pos = st.ReadPos(id_temp);
      if(pos != -1){
          // 核心公式: 弧度 = 方向 * (当前脉冲 - 零点偏移) * 系数
          double radians = direction * (pos - offset) * (3.1415926 / 2048.0); 
          hw_states_position_[i] = radians;
      }

      // 读取速度
      int speed = st.ReadSpeed(id_temp);
      if(speed != -1) {
          double vel_rad_s = direction * speed * (3.1415926 / 2048.0); 
          hw_states_velocity_[i] = vel_rad_s;
      } else {
          hw_states_velocity_[i] = 0.0;
      }
  }
  return hardware_interface::return_type::OK;
}

hardware_interface::return_type XiaobeiHardwareInterface::write(const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  for (std::size_t i = 0; i < info_.joints.size(); i++)
  {
      std::string joint_name = info_.joints[i].name;
      
      int id_temp = 0;
      double offset = 2048.0;
      double direction = 1.0;

      if (joint_map.find(joint_name) != joint_map.end()) {
          id_temp = joint_map[joint_name].id;
          offset = joint_map[joint_name].offset;
          direction = joint_map[joint_name].direction;
      } else {
          try {
              id_temp = std::stoi(joint_name);
          } catch (...) { continue; }
      }

      double target_radians = hw_commands_[i];
      
      // 核心公式: 目标脉冲 = (目标弧度 / 方向 / 系数) + 零点偏移
      // 简化为: 目标脉冲 = (目标弧度 * 方向 * 转换率) + 偏移
      int target_pos = (int)(target_radians * direction * (2048.0 / 3.1415926) + offset);
      
      if(target_pos < 0) target_pos = 0;
      if(target_pos > 4095) target_pos = 4095;

      st.WritePosEx(id_temp, target_pos, 0, 0);
  }
  return hardware_interface::return_type::OK;
}

}  // namespace xiaobei_hardware

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(
  xiaobei_hardware::XiaobeiHardwareInterface, hardware_interface::SystemInterface)