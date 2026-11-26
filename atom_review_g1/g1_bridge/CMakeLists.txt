cmake_minimum_required(VERSION 3.16)
project(g1_atom_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

get_filename_component(DEFAULT_G1_ATOM_PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(G1_ATOM_PROJECT_ROOT "${DEFAULT_G1_ATOM_PROJECT_ROOT}" CACHE PATH
    "Fixed G1 Atom project root embedded in the native runner")
set(G1_ATOM_REVIEW_STORE_PATH "${G1_ATOM_PROJECT_ROOT}/runtime/reviews.json" CACHE FILEPATH
    "Fixed visual-review store embedded in the native runner")
set(G1_ATOM_PREVIEW_ROOT "" CACHE PATH
    "Fixed exact-preview root embedded in the native runner")
if(NOT G1_ATOM_PREVIEW_ROOT)
  set(G1_ATOM_CONFIG_ENV "${G1_ATOM_PROJECT_ROOT}/config.env")
  if(EXISTS "${G1_ATOM_CONFIG_ENV}")
    file(STRINGS "${G1_ATOM_CONFIG_ENV}" G1_ATOM_PREVIEW_LINES
         REGEX "^G1_ATOM_PREVIEW_ROOT=" LIMIT_COUNT 1)
    if(G1_ATOM_PREVIEW_LINES)
      list(GET G1_ATOM_PREVIEW_LINES 0 G1_ATOM_PREVIEW_LINE)
      string(REGEX REPLACE "^G1_ATOM_PREVIEW_ROOT=" "" G1_ATOM_PREVIEW_ROOT
             "${G1_ATOM_PREVIEW_LINE}")
    endif()
  endif()
endif()
if(NOT G1_ATOM_PREVIEW_ROOT)
  message(FATAL_ERROR
          "G1_ATOM_PREVIEW_ROOT must be passed or defined in the fixed project config.env")
endif()
if(NOT IS_ABSOLUTE "${G1_ATOM_REVIEW_STORE_PATH}")
  get_filename_component(G1_ATOM_REVIEW_STORE_PATH
                         "${G1_ATOM_PROJECT_ROOT}/${G1_ATOM_REVIEW_STORE_PATH}" ABSOLUTE)
endif()
if(NOT IS_ABSOLUTE "${G1_ATOM_PREVIEW_ROOT}")
  get_filename_component(G1_ATOM_PREVIEW_ROOT
                         "${G1_ATOM_PROJECT_ROOT}/${G1_ATOM_PREVIEW_ROOT}" ABSOLUTE)
endif()

get_filename_component(DEFAULT_UNITREE_SDK_ROOT
                       "${CMAKE_CURRENT_SOURCE_DIR}/../../external/unitree_sdk2" ABSOLUTE)
set(UNITREE_SDK_ROOT "$ENV{UNITREE_SDK_ROOT}" CACHE PATH "Path to unitree_sdk2")
if(NOT UNITREE_SDK_ROOT)
  set(UNITREE_SDK_ROOT ${DEFAULT_UNITREE_SDK_ROOT})
endif()

find_package(OpenSSL REQUIRED)
find_package(Threads REQUIRED)

set(UNITREE_ARCH ${CMAKE_SYSTEM_PROCESSOR})
find_library(
  UNITREE_SDK_LIB
  unitree_sdk2
  PATHS ${UNITREE_SDK_ROOT}/lib/${UNITREE_ARCH}
  NO_DEFAULT_PATH
  REQUIRED
)
find_library(
  DDSC_LIB
  ddsc
  PATHS ${UNITREE_SDK_ROOT}/thirdparty/lib/${UNITREE_ARCH}
  NO_DEFAULT_PATH
  REQUIRED
)
find_library(
  DDSCXX_LIB
  ddscxx
  PATHS ${UNITREE_SDK_ROOT}/thirdparty/lib/${UNITREE_ARCH}
  NO_DEFAULT_PATH
  REQUIRED
)

add_executable(g1_atom_motion_cli g1_atom_motion_cli.cpp)
target_compile_definitions(
  g1_atom_motion_cli
  PRIVATE
    G1_ATOM_NATIVE_REVIEW_STORE_PATH="${G1_ATOM_REVIEW_STORE_PATH}"
    G1_ATOM_NATIVE_PREVIEW_ROOT="${G1_ATOM_PREVIEW_ROOT}"
)
target_include_directories(
  g1_atom_motion_cli
  SYSTEM PRIVATE
    ${UNITREE_SDK_ROOT}/include
    ${UNITREE_SDK_ROOT}/thirdparty/include
    ${UNITREE_SDK_ROOT}/thirdparty/include/ddscxx
)
target_link_libraries(
  g1_atom_motion_cli
  PRIVATE
    ${UNITREE_SDK_LIB}
    ${DDSC_LIB}
    ${DDSCXX_LIB}
    OpenSSL::Crypto
    Threads::Threads
)
target_compile_options(g1_atom_motion_cli PRIVATE -Wall -Wextra -Wpedantic)
