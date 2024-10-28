import json
from quill.data.agda.reader import parse_file
from quill.nn.inference import Inferer

train_cfg = json.load(open('/home/kokos/Projects/nagda/data/config.json', 'r'))
dev_files = train_cfg['dev_files']

inferer = Inferer(train_cfg['model_config'], 'cuda')
inferer.load('/home/kokos/Projects/nagda/data/model0.pt', strict=True, map_location='cuda')
inferer.eval()

file: str = dev_files[13]
selected = inferer.select_premises(
    file=parse_file(f'/home/kokos/Projects/nagda/data/stdlib/{file}.json', validate=True))

