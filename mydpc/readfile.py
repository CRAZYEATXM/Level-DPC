# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def readfile(
    file,
    dimensions=None,
    has_class=True,
    has_id=None,
    sep=None,
    skip_header=1,
):
  """自動過濾 ID 欄位 (支援智慧 ID 自動偵測)、自動偵測特徵維度與相容字串類別標籤的多維資料讀取函式

  參數：
  - file: 檔案路徑
  - dimensions: 特徵維度數量。若為 None (預設)，自動根據欄位推算
  - has_class: 最後一欄是否為類別標籤 (Class/Label)，預設為 True
  - has_id: 第一欄 (第 0 欄) 是否為流水號/ID。
            若設為 None (預設)，程式會自動偵測第 0 欄是否為連續流水號 (1..N 或 0..N-1)！
  - sep: 欄位分隔符號 (預設 None 自動判斷逗號、Tab 或空格)
  - skip_header: 跳過標頭行數 (預設 1)

  回傳：
  (distances, xxdist, ND, N, points, dataclass)
  """
  print('Loading the file ...')

  # 1. 使用 Pandas 讀取檔案
  engine = 'python' if sep is None else 'c'
  df = pd.read_csv(
      file,
      sep=sep,
      header=None,
      skiprows=skip_header if skip_header > 0 else None,
      engine=engine,
      encoding='UTF-8',
  )

  total_cols = df.shape[1]

  # 2. 智慧 ID 自動偵測：檢查第 0 欄是否為整數流水號 (1..N 或 0..N-1)
  if has_id is None:
    try:
      col0 = df.iloc[:, 0].to_numpy()
      col0_int = col0.astype(int)
      # 判斷是否為 1..N 或 0..N-1 的連續遞增序列
      is_seq_1 = np.array_equal(col0_int, np.arange(1, len(df) + 1))
      is_seq_0 = np.array_equal(col0_int, np.arange(0, len(df)))
      has_id = is_seq_1 or is_seq_0
    except Exception:
      has_id = False

  # 3. 確定特徵與標籤的起迄欄位索引
  start_col = 1 if has_id else 0  # 若有 ID，從第 1 欄開始；無 ID 則從第 0 欄開始
  end_col = (
      total_cols - 1 if has_class else total_cols
  )  # 若有 Class，最後一欄為標籤

  # 自動偵測特徵維度數量
  if dimensions is None:
    dimensions = end_col - start_col

  # 4. 自動切片：分離特徵座標 (points) 與 類別標籤 (dataclass)
  points = df.iloc[:, start_col : start_col + dimensions].to_numpy(
      dtype=float
  )

  if has_class:
    dataclass = df.iloc[:, end_col].astype(str).to_numpy()
  else:
    dataclass = None

  ND = len(points)

  # 5. 計算兩兩歐式距離矩陣
  dist = cdist(points, points, metric='euclidean')
  xxdist = np.unique(dist[dist > 0])
  N = len(xxdist)

  print(
      f'成功讀取資料！資料點數 (ND): {ND}，特徵維度: {points.shape[1]} 維'
      f" ({'偵測到第 0 欄為 ID 並已自動跳過' if has_id else '無 ID 欄位，完整讀取特徵'})。"
  )

  return (dist, xxdist, ND, N, points, dataclass)