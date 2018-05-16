dims=( 50 100 200 300 )
walks=(5 10 25)
contexts=(5 10 25)
epochs=(5 10 25)
for d in "${dims[@]}"; do
    for l in "${walks[@]}"; do
        for k in "${contexts[@]}"; do
            for e in "${epochs[@]}"; do
                ./node2vec -i:graph/wordnet.edgelist -o:emb/wordnet.$d.$l.$k.$e.T.emb -d:$d -l:$l -k:$k -e:$e -dr -v
            done
        done
    done
done
for d in "${dims[@]}"; do
    for l in "${walks[@]}"; do
        for k in "${contexts[@]}"; do
            for e in "${epochs[@]}"; do
                ./node2vec -i:graph/wordnet.edgelist -o:emb/wordnet.$d.$l.$k.$e.F.emb -d:$d -l:$l -k:$k -e:$e -v
            done
        done
    done
done