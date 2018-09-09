#!/bin/sh
export CURRENTDIR=`pwd`
export PYTHONPATH=$PYTHONPATH:$CURRENTDIR

cd notebooks

jupyter lab