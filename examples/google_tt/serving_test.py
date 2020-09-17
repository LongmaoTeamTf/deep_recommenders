import json
import time
import requests

left_data = {
    'past_watches': [['12314'] * 30],
    'seed_id': [['1111']],
    'seed_category': [['音乐']],
    'seed_tags': [['tag1', 'tag3']],
    'seed_gap_time': [[1000]],
    'seed_duration_time': [[2000]],
    'seed_play_count': [[10000]],
    'seed_like_count': [[200]],
    'seed_share_count': [[20]],
    'seed_collect_count': [[100]]
}

right_data = {
    'cand_id': [['1111']],
    'cand_category': [['音乐']],
    'cand_tags': [['tag1', 'tag3']],
    'cand_gap_time': [[1000]],
    'cand_duration_time': [[2000]],
    'cand_play_count': [[10000]],
    'cand_like_count': [[200]],
    'cand_share_count': [[200]],
    'cand_collect_count': [[200]]
}


data = json.dumps({"signature": "serving_default", "inputs": right_data})
headers = {"content-type": "application/json"}

start = time.time()
json_response = requests.post('http://localhost:8501/v1/models/google_tt_candidate:predict', data=data, headers=headers)
stop = time.time()
print(stop-start)

print(json_response.text)