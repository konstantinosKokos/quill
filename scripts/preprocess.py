from Name.data.agda.reader import parse_dir
from Name.data.tokenization import tokenize_to_pps
import pickle


if __name__ == '__main__':
    files = [tokenize_to_pps(file) for file in parse_dir('../data/stdlib', strict=False, validate=True)]

    print(f'Tokenized {len(files)} files with {sum(len(file.hole_strings) for file in files)} holes.')
    with open('../data/stdlib-pp.p', 'wb') as f:
        pickle.dump(files, f)
