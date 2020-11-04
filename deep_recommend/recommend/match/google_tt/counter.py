import json
import pathlib


data_dir = "/home/xddz/data/two_tower_data"
data_dir = pathlib.Path(data_dir)
filenames = sorted([str(fn) for fn in data_dir.glob("*.csv")])[-7:]


counter = {}
global_ids = set()

def count(id):
    if id not in global_ids:
        global_ids.add(id)
        counter[id] = 1
    else:
        counter[id] += 1

for fn in filenames:
    print(fn)
    step = 0
    with open(fn, 'r') as f:
        for line in f:
            line_data = line.strip("\n").split(",")
            past_watches = line_data[2]
            seed_id = line_data[3]
            cand_id = line_data[12]
            for pw in past_watches.split('_')[:10]:
                count(pw)
            count(seed_id)
            count(cand_id)
            step += 1
            if step % 1000 == 0:
                print(f"complte: {step}")

print(len(global_ids))

counter = dict(sorted(counter.items(), key=lambda item:item[1], reverse=True))

with open(str(data_dir/'counter.json'), 'w') as f:
    f.write(json.dumps(counter, indent=4))
    