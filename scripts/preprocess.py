from src.Name.data.reader import parse_dir, enum_references
from src.Name.data.tokenization import tokenize_file
import pickle


samples = []
for i, file in enumerate(parse_dir('../stdlib', version='simplified')):
    anonymous = enum_references(file)
    scope, holes = tokenize_file(anonymous)
    if len(holes) != 0:
        samples.append((scope, holes))

with open('../data/tokenized_sim.p', 'wb') as f:
    pickle.dump(samples, f)
