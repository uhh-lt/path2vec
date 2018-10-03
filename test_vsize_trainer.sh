export CUDA_VISIBLE_DEVICES=0

for vsize in 50 100 200 300 400 500 600
	do
		python3 embeddings.py --input_file jcn-semcor-thresh01-near50.tsv.gz  --vsize ${vsize} --bsize 100 --lrate 0.001 --vocab_file synsets_vocab.json.gz --use_neighbors True --neighbor_count 3 --epochs 15
		python3 embeddings.py --input_file jcn-brown-thresh01-near50.tsv.gz --vsize ${vsize} --bsize 100 --lrate 0.001 --vocab_file synsets_vocab.json.gz --use_neighbors True --neighbor_count 3 --epochs 15
		python3 embeddings.py --input_file lch-thresh15-near50.tsv.gz  --vsize ${vsize} --bsize 100 --lrate 0.001 --vocab_file synsets_vocab.json.gz --use_neighbors True --neighbor_count 3 --epochs 15
	done
