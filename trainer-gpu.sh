dataset=$1
vsize=$2
export CUDA_VISIBLE_DEVICES=$3

for lrate in 0.001 0.002 0.0005 0.004 ; do
	for bsize in 100 200 50 25 10 ; do
		python embeddings.py ${dataset} ${vsize} ${bsize} ${lrate} synsets_vocab.json.gz
	done
done
