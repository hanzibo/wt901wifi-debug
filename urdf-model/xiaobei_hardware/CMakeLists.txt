cmake_minimum_required(VERSION 3.8)
project(xiaobei_hardware)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)

# =========================================================
# 1. 收集舵机SDK的源文件
# =========================================================
file(GLOB SCSERVO_SOURCES "src/scservo_sdk/*.cpp")

# =========================================================
# 2. 编译库 (把你的接口代码 + 舵机SDK代码一起编译)
# =========================================================
add_library(${PROJECT_NAME} SHARED
  src/xiaobei_hardware_interface.cpp
  ${SCSERVO_SOURCES}
)

# =========================================================
# 3. 包含头文件路径
# =========================================================
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  # 这一行很重要，让代码能找到 scservo_sdk 的头文件
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/scservo_sdk>
)

ament_target_dependencies(${PROJECT_NAME}
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle
)

pluginlib_export_plugin_description_file(hardware_interface xiaobei_hardware.xml)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  FILES xiaobei_hardware.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_dependencies(hardware_interface pluginlib rclcpp rclcpp_lifecycle)

ament_package()