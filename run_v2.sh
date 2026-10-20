#!/bin/bash
# 新真值集（两轮合并后）的采集：基础数据 + 多语言评测；ITAD 等 q_itad 那条队列跑完再抓，避免两个进程一起超限
cd ~/steamsales-guess
python3 src/crawler_extra.py truth_v2                > logs_truth_v2.txt 2>&1
python3 src/crawler_extra.py langs --cohort truth_v2 > logs_langs_truth_v2.txt 2>&1
while tmux has-session -t q_itad 2>/dev/null; do sleep 60; done
python3 src/crawler_extra.py itad --cohort truth_v2  > logs_itad_truth_v2.txt 2>&1
