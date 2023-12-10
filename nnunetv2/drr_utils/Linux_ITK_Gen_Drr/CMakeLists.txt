cmake_minimum_required(VERSION 3.12)
project(Linux_ITK_Gen_Drr)

# 设置编译标准
set(CMAKE_CXX_STANDARD 14)

# 查找 ITK 库的路径
set(ITK_DIR "/home/jjf/ITK")

# 查找 ITK 库
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

# 源文件列表
set(SOURCES gendrr.cpp)

# 生成共享库
add_library(itk_drr SHARED ${SOURCES})

# 链接 ITK 库到共享库
target_link_libraries(itk_drr ${ITK_LIBRARIES})