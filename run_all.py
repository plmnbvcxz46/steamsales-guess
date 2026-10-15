# -*- coding: utf-8 -*-
"""
一条命令重跑"离线"部分 (run_all.py)：从 data/raw/ 的真实采集数据 -> ETL -> 建模 -> 作图 -> MySQL 转储 -> 测试。
不访问网络。要重新采集数据，见 README 第三节的 crawler.py 各子命令。

用法：python run_all.py            # 增量快照按水位线追加，其余全量重建
      python run_all.py --fresh    # 先删掉 data/steam_sales_dw.db，从零重建（快照表也重新装载）
"""
import sys
import subprocess
import pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parent
STEPS = [
    ["src/detect_policy_change.py"],
    ["src/prototype_explore.py"],
    ["src/etl_pipeline.py", "--demo-resume"],
    ["src/models.py"],
    ["src/visualize_all.py"],
    ["src/news_extract_audit.py"],
    ["src/export_mysql_dump.py"],
    ["tests/test_etl_and_models.py"],
]

if __name__ == "__main__":
    if "--fresh" in sys.argv:
        db = BASE_DIR / "data" / "steam_sales_dw.db"
        if db.exists():
            db.unlink()
    for step in STEPS:
        print(f"\n========== python {' '.join(step)} ==========", flush=True)
        r = subprocess.run([sys.executable] + step, cwd=BASE_DIR)
        if r.returncode != 0:
            sys.exit(f"步骤失败：{' '.join(step)}")
    print("\n全部完成。")
