#! /bin/sh

BUILD_DIR=$1
BUILD_DIR=${BUILD_DIR:=build}

echo ${BUILD_DIR}
${BUILD_DIR}/test/test01
${BUILD_DIR}/test/bench --validation-only
VEORUN_BIN=${BUILD_DIR}/veorun_tf python test/python/avgpoolgrad.py
