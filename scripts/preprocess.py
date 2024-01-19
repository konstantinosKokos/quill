from Name.data.agda.reader import parse_dir
from Name.data.tokenization import tokenize_file
import pickle


if __name__ == '__main__':
    files = [tokenize_file(file) for file in parse_dir('../data/stdlib', False)]

    print(f'Tokenized {len(files)} files with {sum(len(file.hole_asts) for file in files)} holes.')
    with open('../data/tokenized.p', 'wb') as f:
        pickle.dump(files, f)
