cmake_minimum_required(VERSION 3.18)
project(cis565_path_tracer LANGUAGES CUDA CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Enable C++11 for host code
set(CMAKE_CXX_STANDARD 11)
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

find_package(OpenGL REQUIRED)

if(UNIX)
    include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    find_package(glfw3 REQUIRED)
    find_package(GLEW REQUIRED)
    set(LIBRARIES glfw ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES})
else(UNIX)
    set(EXTERNAL "${CMAKE_SOURCE_DIR}/external")

    set(GLFW_ROOT_DIR ${EXTERNAL})
    set(GLFW_USE_STATIC_LIBS ON)
    find_package(GLFW REQUIRED)

    set(GLEW_ROOT_DIR ${EXTERNAL})
    set(GLEW_USE_STATIC_LIBS ON)
    find_package(GLEW REQUIRED)

    add_definitions(${GLEW_DEFINITIONS})
    include_directories(${GLEW_INCLUDE_DIR} ${GLFW_INCLUDE_DIR})
    set(LIBRARIES ${GLEW_LIBRARY} ${GLFW_LIBRARY} ${OPENGL_LIBRARY})
endif(UNIX)

set(GLM_ROOT_DIR "${CMAKE_SOURCE_DIR}/external")
find_package(GLM REQUIRED)
include_directories(${GLM_INCLUDE_DIRS})

# Include tinygltf header
include_directories("${CMAKE_SOURCE_DIR}/external/include")

# Include OIDN headers
include_directories("${CMAKE_SOURCE_DIR}/external/include/OpenImageDenoise")

# Define OIDN libraries
set(OIDN_LIBRARIES
    "${CMAKE_SOURCE_DIR}/external/lib/OpenImageDenoise.lib"
    "${CMAKE_SOURCE_DIR}/external/lib/OpenImageDenoise_core.lib"
    "${CMAKE_SOURCE_DIR}/external/lib/OpenImageDenoise_device_cuda.lib"
    "${CMAKE_SOURCE_DIR}/external/lib/OpenImageDenoise_device_cpu.lib"
    "${CMAKE_SOURCE_DIR}/external/lib/OpenImageDenoise_utils.lib"
    "${CMAKE_SOURCE_DIR}/external/lib/tbb12.lib"
)

# Define the destination directory
set(OIDN_DEST_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")

# Add a custom command to copy the OIDN libraries
foreach(OIDN_LIB ${OIDN_LIBRARIES})
    get_filename_component(OIDN_LIB_NAME ${OIDN_LIB} NAME)
    add_custom_command(
        OUTPUT "${OIDN_DEST_DIR}/${OIDN_LIB_NAME}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OIDN_LIB} ${OIDN_DEST_DIR}
        DEPENDS ${OIDN_LIB}
    )
    list(APPEND OIDN_COPY_TARGETS "${OIDN_DEST_DIR}/${OIDN_LIB_NAME}")
endforeach()

# Add a custom target to execute the copy command
add_custom_target(copy_oidn_libs ALL DEPENDS ${OIDN_COPY_TARGETS})

set(headers
    src/main.h
    src/image.h
    src/interactions.h
    src/intersections.h
    src/glslUtility.hpp
    src/pathtrace.h
    src/scene.h
    src/sceneStructs.h
    src/preview.h
    src/utilities.h
)

set(sources
    src/main.cpp
    src/stb.cpp
    src/tinygltf.cpp
    src/image.cpp
    src/glslUtility.cpp
    src/pathtrace.cu
    src/intersections.cu
    src/interactions.cu
    src/scene.cpp
    src/preview.cpp
    src/utilities.cpp
)

set(imgui_headers
    src/ImGui/imconfig.h
    src/ImGui/imgui.h
    src/ImGui/imgui_impl_glfw.h
    src/ImGui/imgui_impl_opengl3.h
    src/ImGui/imgui_impl_opengl3_loader.h
    src/ImGui/imgui_internal.h
    src/ImGui/imstb_rectpack.h
    src/ImGui/imstb_textedit.h
    src/ImGui/imstb_truetype.h
)

set(imgui_sources
    src/ImGui/imgui.cpp
    src/ImGui/imgui_demo.cpp
    src/ImGui/imgui_draw.cpp
    src/ImGui/imgui_impl_glfw.cpp
    src/ImGui/imgui_impl_opengl3.cpp
    src/ImGui/imgui_tables.cpp
    src/ImGui/imgui_widgets.cpp
)

list(SORT headers)
list(SORT sources)
list(SORT imgui_headers)
list(SORT imgui_sources)

source_group("Headers" FILES ${headers})
source_group("Sources" FILES ${sources})
source_group("ImGui\\Headers" FILES ${imgui_headers})
source_group("ImGui\\Sources" FILES ${imgui_sources})

#add_subdirectory(src/ImGui)
#add_subdirectory(stream_compaction)  # TODO: uncomment if using your stream compaction

add_executable(${CMAKE_PROJECT_NAME} ${sources} ${headers} ${imgui_sources} ${imgui_headers})
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
if(CMAKE_VERSION VERSION_LESS "3.23.0")
    set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES OFF)
elseif(CMAKE_VERSION VERSION_LESS "3.24.0")
    set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES all-major)
else()
    set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES native)
endif()

# Link OIDN libraries only in Release mode
target_link_libraries(${CMAKE_PROJECT_NAME}
    ${LIBRARIES}
    cudadevrt
    $<$<CONFIG:Release>:${OIDN_LIBRARIES}>
    #stream_compaction  # TODO: uncomment if using your stream compaction
)

# Add dependencies for copying OIDN libraries only in Release mode
add_custom_command(TARGET ${CMAKE_PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Copying OIDN libraries..."
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<$<CONFIG:Release>:${OIDN_LIBRARIES}> ${OIDN_DEST_DIR}
    COMMENT "Copying OIDN libraries in Release mode"
)

target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE "$<$<AND:$<CONFIG:Debug,RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo;-src-in-ptx>;--extended-lambda")
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${CMAKE_PROJECT_NAME})