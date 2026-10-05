cmake_minimum_required(VERSION 3.0.2)
project(sara_simple_planner)

find_package(catkin REQUIRED COMPONENTS
  rospy
  geometry_msgs
  std_msgs
  std_srvs
)

catkin_package()

catkin_install_python(
  PROGRAMS
    scripts/simple-planner.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
