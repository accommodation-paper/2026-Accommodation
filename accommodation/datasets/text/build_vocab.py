

def build_vocab(texts):
    token2idx = {'<pad>':0, '<unk>':1}
    idx = 2
    for sentence in texts:
        for token in sentence.split():
            if token not in token2idx:
                token2idx[token] = idx
                idx += 1
    return token2idx