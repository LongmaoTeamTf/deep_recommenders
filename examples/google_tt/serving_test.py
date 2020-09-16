import json
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

right_columns = {
    'cand_id',
    'cand_category',
    'cand_tags',
    'cand_gap_time',
    'cand_duration_time',
    'cand_play_count',
    'cand_like_count',
    'cand_share_count',
    'cand_collect_count'
}


data = json.dumps({"signature": "serving_default", "inputs": left_data})
headers = {"content-type": "application/json"}

json_response = requests.post('http://localhost:8501/v1/models/google_tt_query:predict', data=data, headers=headers)

print(json_response.text)