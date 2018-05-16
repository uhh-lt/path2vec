import pickle
import sys

inputfile, outputfile = sys.argv[1:]

with open('nodes.pkl', 'rb') as f:
    pkl = pickle.load(f)

with open(inputfile) as f, open(outputfile, "w") as f1:
    #copy first line with dimensions
    f1.write(f.readline())
    for line in f:
        node_index = line.split(" ")[0]
        replace_with = str(pkl.get(node_index))
        f1.write(replace_with + line[len(node_index):])
