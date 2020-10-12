import pathlib



data_dir = "/home/xddz/data/two_tower_data"
data_dir = pathlib.Path(data_dir)
filenames = sorted([str(fn) for fn in data_dir.glob("*.csv")])[-7:]

counter = {}
for fn in filenames:
    with open(fn, 'r') as f:
        for line in f:
            line_data = line.strip("\n").split(",")
            past_watches = line_data[2]
            seed_id = line_data[3]
            cand_id = line_data[12]
            print(past_watches)
            print(seed_id)
            print(cand_id)
        break