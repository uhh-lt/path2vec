#!/bin/bash

for i in *.tsv
do
    sort -n -t $'\t' -k 2 ${i} > ${i}.csv
done