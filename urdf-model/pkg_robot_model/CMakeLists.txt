cmake_minimum_required(VERSION 3.10)
project(pkg_robot_model)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

# =========================================
# 安装指令
# =========================================
# 将 config, launch, meshes, urdf, worlds 文件夹安装到 share/pkg_robot_model 目录下
install(
  DIRECTORY config launch meshes urdf worlds
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS scripts/seed_joint_state.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
