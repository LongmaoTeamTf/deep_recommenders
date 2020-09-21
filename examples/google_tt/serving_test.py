import json
import time
import requests

# left_data = {
#     'past_watches': [['12314'] * 30],
#     'seed_id': [['1111']],
#     'seed_category': [['音乐']],
#     'seed_tags': [['tag1', 'tag3']],
#     'seed_gap_time': [[1000]],
#     'seed_duration_time': [[2000]],
#     'seed_play_count': [[10000]],
#     'seed_like_count': [[200]],
#     'seed_share_count': [[20]],
#     'seed_collect_count': [[100]]
# }

# right_data = {
#     'cand_id': [['1111']]*500,
#     'cand_category': [['音乐']]*500,
#     'cand_tags': [['tag1', 'tag3']]*500,
#     'cand_gap_time': [[1000]]*500,
#     'cand_duration_time': [[2000]]*500,
#     'cand_play_count': [[10000]]*500,
#     'cand_like_count': [[200]]*500,
#     'cand_share_count': [[200]]*500,
#     'cand_collect_count': [[200]]*500
# }

left_data = {'past_watches': [['215891_215969_213241_215971_215849_215930_215902_215772_215612_209749_215901_215903_215366_215776_128415_215757_215490_214615_215190_214990_215777_2104_170828_155349_154290_155149_5614_179233_176368_170321_209965_209966_154751_209684_13885_215888_155885_167643_209963_166176_177235_153110_7746_168968_209686_10360_132634_138637_1742_213367_215968']], 'seed_id': [['154751']], 'seed_category': [['动画']], 'seed_tags': [['动画梦工厂_剧情_动画']], 'seed_gap_time': [[47212749.0]], 'seed_duration_time': [[466]], 'seed_play_count': [[22426]], 'seed_like_count': [[577]], 'seed_collect_count': [[66]], 'seed_share_count': [[472]]}


data = json.dumps({"signature": "serving_default", "inputs": left_data})
headers = {"content-type": "application/json"}

start = time.time()
json_response = requests.post('http://localhost:8501/v1/models/google_tt_query:predict', data=data, headers=headers)
stop = time.time()
print(stop-start)

print(json_response.text)