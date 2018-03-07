#!/bin/sh
LOGDIR = ./log
TOOLS=./caffe/build/tools
mkdir $LOGDIR

GLOG_logtostderr=0 GLOG_log_dir=$LOGDIR $TOOLS/caffe train --solver=solver_ResNet_50.prototxt --weights=ResNet-50-model.caffemodel
echo 'Done.'
