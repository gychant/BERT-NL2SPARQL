
import numpy as np
from tqdm import tqdm


def load_glove_model(glove_file, dim=300):
    print("Loading Glove Model")
    f = open(glove_file, 'r', encoding='utf-8')
    model = {}
    for line in tqdm(f):
        split_line = line.split()
        word = " ".join(split_line[0:len(split_line) - dim])
        embedding = np.array([float(val) for val in split_line[-dim:]])
        model[word] = embedding
    print("Done.\n" + str(len(model)) + " words loaded!")
    return model

def load_w2v_model(w2v_file, dim=300):
    print("Loading %s Model" % w2v_file)
    f = open(w2v_file, 'r', encoding='utf-8')
    model = {}
    i = 0
    for line in tqdm(f):
        if i == 0:  # the first line is the total_num dim
            i += 1
            continue
        split_line = line.split()
        word = " ".join(split_line[0:len(split_line) - dim])
        embedding = np.array([float(val) for val in split_line[-dim:]])
        model[word] = embedding
    print("Done.\n" + str(len(model)) + " words loaded!")
    return model


if __name__ == '__main__':
    # load_glove_model("glove/glove.42B.300d.txt", dim=300)  # glove.6B.300d.txt, 42B 840B
    load_w2v_model("word_vector/enwiki_20180420_300d.txt", dim=300)
