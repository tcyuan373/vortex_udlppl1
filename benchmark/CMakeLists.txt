
# Client log tags
set(LOG_TAG_QUERIES_SENDING_START 10000)
set(LOG_TAG_QUERIES_SENDING_END 10001)
set(LOG_TAG_QUERIES_RESULT_CLIENT_RECEIVED 10100)

add_executable(build_hnsw_index perf_data/hnsw/build_hnsw_index.cpp)
target_include_directories(build_hnsw_index PRIVATE ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS})
target_link_libraries(build_hnsw_index Boost::python Boost::numpy ${Python_LIBRARIES})

add_executable(run_benchmark benchmark_client.cpp 
                              run_benchmark.cpp 
                              benchmark_dataset.cpp 
                              ${CMAKE_SOURCE_DIR}/vortex_udls/serialize_utils.cpp)
target_link_libraries(run_benchmark derecho::cascade)
target_compile_definitions(run_benchmark PRIVATE
    LOG_TAG_QUERIES_SENDING_START=${LOG_TAG_QUERIES_SENDING_START}
    LOG_TAG_QUERIES_SENDING_END=${LOG_TAG_QUERIES_SENDING_END}
    LOG_TAG_QUERIES_RESULT_CLIENT_RECEIVED=${LOG_TAG_QUERIES_RESULT_CLIENT_RECEIVED}
)