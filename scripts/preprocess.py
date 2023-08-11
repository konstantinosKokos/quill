from src.Name.data.agda.reader import parse_dir, enum_references
from src.Name.data.tokenization import tokenize_file
import pickle


samples = []
for i, file in enumerate(parse_dir('../data/beauty')):
    try:
        anonymous, _ = enum_references(file)
    except ValueError:
        continue
    name, scope, holes = tokenize_file(anonymous)
    if len(holes) != 0:
        samples.append((name, scope, holes))

with open('../data/tokenized.p', 'wb') as f:
    pickle.dump(samples, f)
