import pickle

triples = pickle.load(open("trajectories/database.pkl", "rb"))

for t in triples:
    print(t[2])
