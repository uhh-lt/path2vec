#!/bin/bash

for i in ${1}/*.vec.gz
   do
	echo ${i}
	/projects/ltg/python3/bin/python3 wsd/graph_wsd_test_v2.py ${i}
   done