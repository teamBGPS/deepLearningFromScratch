cmake_minimum_required (VERSION 3.8)

project(deepLearningFromScratch)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_executable (TrainDnn "trainDnn.cpp" "Layer.cpp" "Util.cpp" "UtilLayer.cpp")
if(CMAKE_COMPILER_IS_GNUCC)
    target_link_libraries(TrainDnn pthread)
endif(CMAKE_COMPILER_IS_GNUCC)

add_executable (PredictDnn "predictDnn.cpp" "Layer.cpp" "Util.cpp")
