#!/bin/bash
# 补充采集排队脚本（在服务器上跑）：Steam 和 ITAD 是两个网站、各自限速，分成两条队列并行
#   tmux new-session -d -s q_steam "./run_extra.sh steam"
#   tmux new-session -d -s q_itad  "./run_extra.sh itad"
# 中途断了重新运行即可：已缓存的请求会直接跳过
cd ~/steamsales-guess
if [ "$1" = steam ]; then
  python3 src/crawler_extra.py langs --cohort truth      > logs_langs_truth.txt 2>&1
  python3 src/crawler_extra.py truth_v2                  > logs_truth_v2.txt 2>&1
  python3 src/crawler_extra.py langs --cohort truth_v2   > logs_langs_truth_v2.txt 2>&1
  python3 src/crawler_extra.py langs --cohort train_2018 > logs_langs_train_2018.txt 2>&1
else
  python3 src/crawler_extra.py itad --cohort truth       > logs_itad_truth.txt 2>&1
  python3 src/crawler_extra.py itad --cohort truth_v2    > logs_itad_truth_v2.txt 2>&1
  python3 src/crawler_extra.py itad --cohort train_2018  > logs_itad_train_2018.txt 2>&1
fi
