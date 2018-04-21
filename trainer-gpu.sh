export CUDA_VISIBLE_DEVICES=$3
dataset=$1
lrate=$2

# Vector sizes
for vsize in 16 32 64 100 200 300
    do
	# Batch sizes
	for bsize in 10
	    do
		    python embeddings.py ${dataset} ${vsize} ${bsize} ${lrate} synsets_vocab.json.gz
	    done
 done
