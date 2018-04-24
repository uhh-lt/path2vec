export CUDA_VISIBLE_DEVICES=$3
dataset=$1
vsize=$2
bsize=$4 

for lrate in 0.005 ; do
	python embeddings.py ${dataset} ${vsize} ${bsize} ${lrate} synsets_vocab.json.gz
done
