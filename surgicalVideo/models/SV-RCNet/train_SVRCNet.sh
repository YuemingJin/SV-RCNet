#!/bin/sh
LOGDIR = ./log
TOOLS=./caffe/build/tools
mkdir $LOGDIR

GLOG_logtostderr=0 GLOG_log_dir=$LOGDIR $TOOLS/caffe train --solver=SVRCNet_solver.prototxt --weights=resnet_workflow.caffemodel
echo 'Done.'
