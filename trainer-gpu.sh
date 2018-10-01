export CUDA_VISIBLE_DEVICES=$3
dataset=$1
vsize=$2
bsize=$4 

for lrate in 0.005 ; do
	python embeddings.py --input_file ${dataset} --vsize ${vsize} --bsize ${bsize} --lrate ${lrate} --vocab_file synsets_vocab.json.gz --use_neighbors True --neighbor_count 3 --epochs 15
done
