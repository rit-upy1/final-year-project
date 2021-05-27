import pickle

example = {1: "Halo", 2: "What is love"}

with open("dict.pickle", "wb") as dump_location:
    pickle.dump(example, dump_location)

with open("dict.pickle", "rb") as dump_location:
    value = pickle.load(dump_location)

print(value)