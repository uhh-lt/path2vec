export CUDA_VISIBLE_DEVICES=0

for value in {1..10}
do
	python3 embeddings_pytorch.py --input_file jcn-semcor-thresh01-near50.tsv.gz  --vsize 300 --bsize 100 --lrate 0.001 --vocab_file synsets_vocab.json.gz --use_neighbors True --epochs 10 --name $value
	python3 embeddings_pytorch.py --input_file jcn-brown-thresh01-near50.tsv.gz --vsize 300 --bsize 100 --lrate 0.001 --vocab_file synsets_vocab.json.gz --use_neighbors True --epochs 10 --name $value
	python3 embeddings_pytorch.py --input_file lch-thresh15-near50.tsv.gz  --vsize 300 --bsize 100 --lrate 0.001 --vocab_file synsets_vocab.json.gz --use_neighbors True --epochs 10 --name $value
done
