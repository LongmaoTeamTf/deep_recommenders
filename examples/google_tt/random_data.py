import random

csv_header = [
    'label',
    'udid',
    'past_watches',
    'seed_id',
    'seed_category',
    'seed_tags',
    'seed_gap_time',
    'seed_duration_time',
    'seed_play_count',
    'seed_like_count',
    'seed_share_count',
    'seed_collect_count',
    'seed_reply_count',
    'cand_id',
    'cand_category',
    'cand_tags',
    'cand_gap_time',
    'cand_duration_time',
    'cand_play_count',
    'cand_like_count',
    'cand_share_count',
    'cand_collect_count',
    'cand_reply_count',
]

video_ids_hash_size = 5000
video_categories_hash_size = 20
video_tags_hash_size = 500

video_ids = [
    str(i)
    for i in range(video_ids_hash_size)
]

video_categories = [
    'cat{}'.format(i)
    for i in range(video_categories_hash_size)
]

video_tags = [
    'tag{}'.format(i)
    for i in range(video_tags_hash_size)
]

train_data_file_num = 10
per_file_batch_num = 10000
batch_size = 128
for i in range(train_data_file_num):
    with open(f'data/train_data_{i}.csv', 'w', encoding='utf-8') as f:
        f.write(','.join(csv_header)+'\n')
        for i in range(per_file_batch_num):
            label = [
                '1' if random.random() > 0.75 else '0'
                for _ in range(5)
            ]
            label = '_'.join(label)
            udid = '1000001'
            past_watches = random.choices(video_ids, k=random.randint(1, 10))
            past_watches = '_'.join(past_watches)
            seed_id = random.choice(video_ids)
            seed_category = random.choice(video_categories)
            seed_tags = random.choices(video_tags, k=random.randint(1, 5))
            seed_tags = '_'.join(seed_tags)
            seed_gap_time = str(random.randint(1, 100000))
            seed_duration_time = str(random.randint(1, 500))
            seed_play_count = str(random.randint(1, 2000))
            seed_like_count = str(random.randint(1, 500))
            seed_share_count = str(random.randint(1, 500))
            seed_collect_count = str(random.randint(1, 500))
            seed_reply_count = str(random.randint(1, 100))
            cand_id = random.choice(video_ids)
            cand_category = random.choice(video_categories)
            cand_tags = random.choices(video_tags, k=random.randint(1, 5))
            cand_tags = '_'.join(cand_tags)
            cand_gap_time = str(random.randint(1, 100000))
            cand_duration_time = str(random.randint(1, 500))
            cand_play_count = str(random.randint(1, 2000))
            cand_like_count = str(random.randint(1, 500))
            cand_share_count = str(random.randint(1, 500))
            cand_collect_count = str(random.randint(1, 500))
            cand_reply_count = str(random.randint(1, 100))

            line_list = [
                label, udid, past_watches, seed_id, seed_category, seed_tags, seed_gap_time, seed_duration_time,
                seed_play_count, seed_like_count, seed_share_count, seed_collect_count, seed_reply_count,
                cand_id, cand_category, cand_tags, cand_gap_time, cand_duration_time, cand_play_count,
                cand_like_count, cand_share_count, cand_collect_count, cand_reply_count
            ]

            f.write(','.join(line_list)+'\n')

