'''
@Description: 
@version: 
@License: MIT
@Author: Wang Yao
@Date: 2020-03-26 16:55:51
@LastEditors: Wang Yao
@LastEditTime: 2020-03-26 17:42:40
'''
import os
import requests
from tqdm import tqdm
from contextlib import closing
from zipfile import ZipFile

filename = "xiaohuangji50w_fenciA.conv.zip"
url = "https://github.com/candlewill/Dialog_Corpus/raw/master/{}".format(filename)

with closing(requests.get(url, stream=True)) as response:
    chunk_size = 1024
    content_size = int(response.headers['content-length'])
    progress = tqdm(
        total=content_size, unit='B', unit_scale=True, desc=filename)
    with open(filename, "wb") as f:
       for chunk in response.iter_content(chunk_size=chunk_size):
           f.write(chunk)
           progress.update(chunk_size)

with ZipFile(filename, "r") as f:
    for filename in f.namelist(): 
        f.extract(filename, '.')
