#!/bin/bash
rm CMakeCache.txt -f
rm CMakeFiles -rf
#set your eigen path here
cmake . -DEIGEN3_INCLUDE_DIR=/home/zeeeyang/2.researchs/lexicalized_treelstm/eigen_bak/include/eigen3
make BiTreeSentimentZhu -j 9
