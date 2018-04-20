#!/bin/sh

# Vector sizes
for vsize in 16 32 64 100 200 300
    do
	# Batch sizes
	for bsize in 10
	    do
	    # Learning rates
	    for lrate in 0.0005 0.001 0.002 0.004
		do
		    python3 embeddings.py jcn_semcor_thresh01.tsv.gz  ${vsize} ${bsize} ${lrate} synsets_vocab.json.gz
		    python3 embeddings.py jcn_brown_thresh01.tsv.gz  ${vsize} ${bsize} ${lrate} synsets_vocab.json.gz
		done
	    done
    done
