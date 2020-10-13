cmake_minimum_required(VERSION 2.8.3)
project(laikago_socket)

include_directories(include)
add_compile_options(-std=c++11)

set(EXTRA_LIBS -pthread liblaikago_comm.so lcm)

add_executable(client client.cpp control_api.cpp)
target_link_libraries(client ${EXTRA_LIBS})
