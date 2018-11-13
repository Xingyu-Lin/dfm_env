#!/usr/bin/env bash
#. ~/.bashrc
if [[ $(hostname) = *"psc"* ]] || [[ $(hostname) = *"compute-0"* ]]; then
    export LC_ALL=C.UTF-8
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/xlin3/.mujoco/mujoco200_macos/bin
fi

PATH=~/software/miniconda3/bin:$PATH
. activate her
#export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
