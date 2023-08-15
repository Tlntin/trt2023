# We need cmake >= 3.8, since 3.8 introduced CUDA as a first class language
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(CustomPlugin LANGUAGES CXX CUDA)

set(ENABLED_SMS "-DENABLE_SM72 -DENABLE_SM75 -DENABLE_SM80 -DENABLE_SM86 -DENABLE_SM87 -DENABLE_SM89 -DENABLE_SM90")
# Enable all compile warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ENABLED_SMS} -O3 -Wall -Wno-long-long -pedantic -Wno-deprecated-declarations")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")
set(libname customplugin)
set(version _2023)
# Sets variable to a value if variable is unset.
macro(set_ifndef var val)
    if (NOT ${var})
        set(${var} ${val})
    endif ()
    message(STATUS "Configurable variable ${var} set to ${${var}}")
endmacro()

# -------- CONFIGURATION --------
set_ifndef(TRT_LIB /home/player/TensorRT-8.6.1.6/lib/)
set_ifndef(TRT_INCLUDE /home/player/TensorRT-8.6.1.6/include)
set_ifndef(TRT_INCLUDE_2 /usr/local/cuda-11.8/targets/x86_64-linux/include/)
set_ifndef(CUDA_LIB /usr/local/cuda-11.8)

# Find dependencies:
message("\nThe following variables are derived from the values of the previous variables unless provided explicitly:\n")


find_library(_NVINFER_LIB nvinfer HINTS ${TRT_LIB} PATH_SUFFIXES lib lib64)
#find_library(_CUDA_DRIVER_LIB NAMES cuda HINTS ${TRT_LIB} PATH_SUFFIXES lib lib64)
find_library(_CUBLAS_LIB NAMES cublas HINTS ${CUDA_LIB} PATH_SUFFIXES lib lib64)
find_library(_CUDART_LIB NAMES cudart HINTS ${CUDA_LIB} PATH_SUFFIXES lib lib64)
find_library(_CURAND_LIB NAMES curand HINTS ${CUDA_LIB} PATH_SUFFIXES lib lib64)

set_ifndef(NVINFER_LIB ${_NVINFER_LIB})
set_ifndef(CUBLAS_LIB ${_CUBLAS_LIB})
set_ifndef(CUDART_LIB ${_CUDART_LIB})
set_ifndef(CURAND_LIB ${_CURAND_LIB})

MESSAGE(STATUS ${CUBLAS_LIB})
MESSAGE(STATUS ${CUDART_LIB})
MESSAGE(STATUS ${CURAND_LIB})

# -------- BUILDING --------

# Add include directories
include_directories(${CUDA_INC_DIR} ${TRT_INCLUDE} ${TRT_INCLUDE_2} ${CMAKE_SOURCE_DIR})

# Define clip embedding library target
	add_library(${libname}${version} MODULE
		${CMAKE_SOURCE_DIR}/common/bboxUtils.h
                ${CMAKE_SOURCE_DIR}/common/bertCommon.h
                ${CMAKE_SOURCE_DIR}/common/checkMacrosPlugin.cpp
                ${CMAKE_SOURCE_DIR}/common/checkMacrosPlugin.h
                ${CMAKE_SOURCE_DIR}/common/common.cuh
                ${CMAKE_SOURCE_DIR}/common/cub_helper.h
                ${CMAKE_SOURCE_DIR}/common/cudaDriverWrapper.cpp
                ${CMAKE_SOURCE_DIR}/common/cudaDriverWrapper.h
                ${CMAKE_SOURCE_DIR}/common/dimsHelpers.h
                ${CMAKE_SOURCE_DIR}/common/half.h
                ${CMAKE_SOURCE_DIR}/common/kernel.cpp
                ${CMAKE_SOURCE_DIR}/common/kernel.h
                ${CMAKE_SOURCE_DIR}/common/mrcnn_config.h
                ${CMAKE_SOURCE_DIR}/common/nmsHelper.cpp
                ${CMAKE_SOURCE_DIR}/common/nmsUtils.h
                ${CMAKE_SOURCE_DIR}/common/plugin.cpp
                ${CMAKE_SOURCE_DIR}/common/plugin.h
                ${CMAKE_SOURCE_DIR}/common/reducedMathPlugin.cpp
                ${CMAKE_SOURCE_DIR}/common/serialize.hpp
                ${CMAKE_SOURCE_DIR}/common/kernels/allClassNMS.cu
                ${CMAKE_SOURCE_DIR}/common/kernels/bboxDeltas2Proposals.cu
                ${CMAKE_SOURCE_DIR}/common/kernels/common.cu
                ${CMAKE_SOURCE_DIR}/common/kernels/cropAndResizeKernel.cu
                ${CMAKE_SOURCE_DIR}/common/kernels/decodeBBoxes.cu
                ${CMAKE_SOURCE_DIR}/common/kernels/decodeBbox3DKernels.cu
                ${CMAKE_SOURCE_DIR}/common/kernels/detectionForward.cu
                ${CMAKE_SOURCE_DIR}/common/kernels/extractFgScores.cu
                ${CMAKE_SOURCE_DIR}/common/kernels/gatherTopDetections.cu
                ${CMAKE_SOURCE_DIR}/common/kernels/generateAnchors.cu
                ${CMAKE_SOURCE_DIR}/common/kernels/gridAnchorLayer.cu
                ${CMAKE_SOURCE_DIR}/common/kernels/lReLU.cu
                ${CMAKE_SOURCE_DIR}/common/kernels/maskRCNNKernels.cu
                ${CMAKE_SOURCE_DIR}/common/kernels/maskRCNNKernels.h
                ${CMAKE_SOURCE_DIR}/common/kernels/nmsLayer.cu
                ${CMAKE_SOURCE_DIR}/common/kernels/normalizeLayer.cu
                ${CMAKE_SOURCE_DIR}/common/kernels/permuteData.cu
                ${CMAKE_SOURCE_DIR}/common/kernels/pillarScatterKernels.cu
                ${CMAKE_SOURCE_DIR}/common/kernels/priorBoxLayer.cu
                ${CMAKE_SOURCE_DIR}/common/kernels/proposalKernel.cu
                ${CMAKE_SOURCE_DIR}/common/kernels/proposalsForward.cu
                ${CMAKE_SOURCE_DIR}/common/kernels/reducedMathPlugin.h
                ${CMAKE_SOURCE_DIR}/common/kernels/regionForward.cu
                ${CMAKE_SOURCE_DIR}/common/kernels/reorgForward.cu
                ${CMAKE_SOURCE_DIR}/common/kernels/roiPooling.cu
                ${CMAKE_SOURCE_DIR}/common/kernels/rproiInferenceFused.cu
                ${CMAKE_SOURCE_DIR}/common/kernels/saturate.h
                ${CMAKE_SOURCE_DIR}/common/kernels/sortScoresPerClass.cu
                ${CMAKE_SOURCE_DIR}/common/kernels/sortScoresPerImage.cu
                ${CMAKE_SOURCE_DIR}/common/kernels/voxelGeneratorKernels.cu
		#${CMAKE_SOURCE_DIR}/api/InferPlugin.cpp


		${CMAKE_SOURCE_DIR}/groupNormPlugin/CMakeLists.txt
                ${CMAKE_SOURCE_DIR}/groupNormPlugin/groupNormKernel.cu
                ${CMAKE_SOURCE_DIR}/groupNormPlugin/groupNormKernel.h
                ${CMAKE_SOURCE_DIR}/groupNormPlugin/groupNormPlugin.cpp
                ${CMAKE_SOURCE_DIR}/groupNormPlugin/groupNormPlugin.h
                ${CMAKE_SOURCE_DIR}/groupNormPlugin/groupNormPluginCommon.h

                ${CMAKE_SOURCE_DIR}/layerNormPlugin/layerNormKernel.cu
                ${CMAKE_SOURCE_DIR}/layerNormPlugin/layerNormKernel.h
                ${CMAKE_SOURCE_DIR}/layerNormPlugin/layerNormPlugin.cpp
                ${CMAKE_SOURCE_DIR}/layerNormPlugin/layerNormPlugin.h
                
                ${CMAKE_SOURCE_DIR}/multiHeadCrossAttentionPlugin/fmhca.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadCrossAttentionPlugin/fmhca.h
                ${CMAKE_SOURCE_DIR}/multiHeadCrossAttentionPlugin/fmhcaPlugin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadCrossAttentionPlugin/fmhcaPlugin.h
                ${CMAKE_SOURCE_DIR}/multiHeadCrossAttentionPlugin/fmha_cross_attention/include/commonDatatype.h
                ${CMAKE_SOURCE_DIR}/multiHeadCrossAttentionPlugin/fmha_cross_attention/include/fmha_cross_attention.h
                ${CMAKE_SOURCE_DIR}/multiHeadCrossAttentionPlugin/fmha_cross_attention/include/sharedCubinLoader.h
                ${CMAKE_SOURCE_DIR}/multiHeadCrossAttentionPlugin/fmha_cross_attention/src/fmha_mhca_fp16_128_128_sm75.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadCrossAttentionPlugin/fmha_cross_attention/src/fmha_mhca_fp16_128_128_sm80.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadCrossAttentionPlugin/fmha_cross_attention/src/fmha_mhca_fp16_128_128_sm86.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadCrossAttentionPlugin/fmha_cross_attention/src/fmha_mhca_fp16_128_128_sm89.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadCrossAttentionPlugin/fmha_cross_attention/src/fmha_mhca_fp16_128_256_sm80.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadCrossAttentionPlugin/fmha_cross_attention/src/fmha_mhca_fp16_128_256_sm86.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadCrossAttentionPlugin/fmha_cross_attention/src/fmha_mhca_fp16_128_256_sm89.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadCrossAttentionPlugin/fmha_cross_attention/src/fmha_mhca_fp16_128_64_sm75.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadCrossAttentionPlugin/fmha_cross_attention/src/fmha_mhca_fp16_128_64_sm80.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadCrossAttentionPlugin/fmha_cross_attention/src/fmha_mhca_fp16_128_64_sm86.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadCrossAttentionPlugin/fmha_cross_attention/src/fmha_mhca_fp16_128_64_sm89.cubin.cpp
                
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha.h
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmhaPlugin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmhaPlugin.h
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/include/commonDatatype.h
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/include/fmha_flash_attention.h
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/include/sharedCubinLoader.h
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_64_32_S_80_sm80.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_64_32_S_80_sm86.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_64_32_S_80_sm89.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_64_64_S_16_sm75.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_64_64_S_32_sm75.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_64_64_S_40_sm75.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_64_64_S_64_sm75.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_64_32_S_128_sm75.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_64_32_S_128_sm80.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_64_32_S_128_sm86.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_64_32_S_128_sm89.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_64_32_S_16_sm80.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_64_32_S_16_sm86.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_64_32_S_16_sm89.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_64_32_S_32_sm80.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_64_32_S_32_sm86.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_64_32_S_32_sm89.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_64_32_S_40_sm80.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_64_32_S_40_sm86.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_64_32_S_40_sm89.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_64_32_S_64_sm80.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_64_32_S_64_sm86.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_64_32_S_64_sm89.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_64_32_S_80_sm75.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_128_16_S_16_sm80.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_128_16_S_16_sm86.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_128_16_S_16_sm89.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_128_16_S_32_sm80.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_128_16_S_32_sm86.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_128_16_S_32_sm89.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_128_16_S_40_sm80.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_128_16_S_40_sm86.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_128_16_S_40_sm89.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_128_16_S_64_sm80.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_128_16_S_64_sm86.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_128_16_S_64_sm89.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_128_32_S_128_sm80.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_128_32_S_128_sm86.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_128_32_S_128_sm89.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_128_32_S_80_sm80.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_128_32_S_80_sm86.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_128_32_S_80_sm89.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_64_16_S_160_sm75.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_64_16_S_160_sm80.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_64_16_S_160_sm86.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_64_16_S_160_sm89.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_64_16_S_256_sm75.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_64_16_S_256_sm80.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_64_16_S_256_sm86.cubin.cpp
                ${CMAKE_SOURCE_DIR}/multiHeadFlashAttentionPlugin/fmha_flash_attention/src/fmha_v2_flash_attention_fp16_64_16_S_256_sm89.cubin.cpp
                
                ${CMAKE_SOURCE_DIR}/seqLen2SpatialPlugin/seqLen2SpatialKernel.cu
                ${CMAKE_SOURCE_DIR}/seqLen2SpatialPlugin/seqLen2SpatialKernel.h
                ${CMAKE_SOURCE_DIR}/seqLen2SpatialPlugin/seqLen2SpatialPlugin.cpp
                ${CMAKE_SOURCE_DIR}/seqLen2SpatialPlugin/seqLen2SpatialPlugin.h

                ${CMAKE_SOURCE_DIR}/splitGeLUPlugin/splitGeLUKernel.cu
                ${CMAKE_SOURCE_DIR}/splitGeLUPlugin/splitGeLUKernel.h
                ${CMAKE_SOURCE_DIR}/splitGeLUPlugin/splitGeLUPlugin.cpp
                ${CMAKE_SOURCE_DIR}/splitGeLUPlugin/splitGeLUPlugin.h
                )


target_include_directories(${libname}${version}
        PUBLIC ${CUB_ROOT_DIR}
        PUBLIC ${CUDA_INSTALL_DIR}/include
        PUBLIC ${TARGET_DIR}
        )

# Use C++11
target_compile_features(${libname}${version} PUBLIC cxx_std_11)

# Link TensorRT's nvinfer lib
target_link_libraries(${libname}${version} PRIVATE ${NVINFER_LIB} ${CUDART_LIB} ${CUBLAS_LIB} ${CURAND_LIB})
link_directories("/usr/local/cuda-11.8/lib64/")

# We need to explicitly state that we need all CUDA files
# to be built with -dc as the member functions will be called by
# other libraries and executables (in our case, Python inference scripts)
set_target_properties(${libname}${version} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        )
