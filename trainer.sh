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
		    python3 embeddings.py --input_file jcn_semcor_thresh01.tsv.gz  --vsize ${vsize} --bsize ${bsize} --lrate ${lrate} --vocab_file synsets_vocab.json.gz --use_neighbors True --neighbor_count 3 --epochs 15
		    python3 embeddings.py --input_file jcn_brown_thresh01.tsv.gz --vsize ${vsize} --bsize ${bsize} --lrate ${lrate} --vocab_file synsets_vocab.json.gz --use_neighbors True --neighbor_count 3 --epochs 15
		done
	    done
    done
