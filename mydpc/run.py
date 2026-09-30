# -*- coding: utf-8 -*-
"""Density Peak Clustering (DPC) 改善演算法 - 三階段自動化分群模組

--------------------------------------------------
Step 1: 層級概念進行 Density Peak 搜尋與微群初始分割。
Step 2: 引入微群重心 (Centroid) 與自然鄰居距離 r，計算相似度矩陣 R 進行第二階段合併。
Step 3: 結合 Single-Linkage 階層式分群 (Hierarchical Clustering) 進行最終目標群數產出。
Step 4: 自動匯出點層級分群標籤檔與全域效能統計總表 (CSV 報表)。
"""

from collections import Counter, deque
import math
import os
import time
import numpy as np
from numpy.linalg import *
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .DCplot import DCplot
from .readfile import readfile
from .rhodelta import rhodelta


# ==============================================================================
# 0. 頂層入口與主演算法流程 (Main Entrypoint & Architecture Controller)
# ==============================================================================


def run(*args, **kwargs):
  """主執行入口 (Entrypoint)"""
  file = kwargs.get('fi')
  sep = kwargs.get('sep', ',')
  name = kwargs.get('name')
  t = kwargs.get('t', None)

  # 自動偵測維度
  dist, xxdist, ND, N, points, dataclass = readfile(file, sep=sep)
  (rho, delta, ordrho, dc, nneigh) = rhodelta(
      dist, xxdist, ND, N, percent=2.0
  )

  clusters = density_peak_clustering(
      dist,
      rho,
      delta,
      dc,
      nneigh,
      ordrho,
      ND,
      17,
      0.1,
      file,
      points,
      dataclass,
      name,
      t,
  )
  return clusters


def density_peak_clustering(
    dist,
    rho,
    delta,
    dc,
    nneigh,
    ordrho,
    ND,
    rhomin,
    deltamin,
    file,
    points,
    dataclass,
    name,
    t,
):
  """主演算法流程：三階段密度峰值分群 (DPC-based Three-stage Clustering)"""
  start_time = time.time()  # 記錄開始時間
  dirpath = os.getcwd()
  folder_path = os.path.join(dirpath, 'result_ph', '')
  file_name = os.path.basename(file)

  # ------------------------------------------------------------------
  # STEP 1: Peak 搜尋與初始微群切割 (High Merge Accuracy guaranteed)
  # ------------------------------------------------------------------
  peaks, peak_levels = findpeak(nneigh, delta)
  first_clusters_result, cluster_levels = first_clusters(
      nneigh, peaks, ND, peak_levels
  )

  # [修改] 傳入 step=1
  find_wrongly_merged(
      dataclass, first_clusters_result, folder_path, file_name, step=1
  )

  dpc_final_cluster = labelpoint(first_clusters_result, ND)
  step = 1
  DCplot(
      dist,
      points,
      ND,
      rho,
      delta,
      ordrho,
      dc,
      nneigh,
      17,
      0.1,
      dpc_final_cluster,
      file_name,
      dataclass,
      folder_path,
      name,
      str(step),
  )

  # ------------------------------------------------------------------
  # STEP 2: 基於相似度 R (結合重心距離與自然鄰居) 進行微群合併
  # ------------------------------------------------------------------
  second_clusters_result = External_Environment(
      rho, first_clusters_result, cluster_levels, dist, points
  )
  dpc_final_cluster = labelpoint(second_clusters_result, ND)

  # [修改] 傳入 step=2
  find_wrongly_merged(
      dataclass, second_clusters_result, folder_path, file_name, step=2
  )

  step = 2
  DCplot(
      dist,
      points,
      ND,
      rho,
      delta,
      ordrho,
      dc,
      nneigh,
      17,
      0.1,
      dpc_final_cluster,
      file_name,
      dataclass,
      folder_path,
      name,
      str(step),
  )

  # ------------------------------------------------------------------
  # STEP 3: 階層式分群 (Single-Linkage) 收斂至指定群數 t
  # ------------------------------------------------------------------
  if t is None:
    if dataclass is not None:
      # 自動算出真實標籤有幾種類別 (排除 Missing_Label)
      valid_labels = pd.Series(dataclass).dropna()
      valid_labels = valid_labels[valid_labels != 'Missing_Label']
      t = len(np.unique(valid_labels))
      if t == 0:
        t = 2  # 預設保底 2 群
    else:
      t = 2  # 預設保底 2 群

    print(f'📌 第三階段目標群數設定為: t = {t}')

  final_result = hier_clustering(second_clusters_result, dist, rho, ND, t)

  # 重新組織索引結構以供評估與畫圖
  index_dict = {}
  for index, value in enumerate(final_result):
    if value not in index_dict:
      index_dict[value] = []
    index_dict[value].append(index)

  result = [indices for indices in index_dict.values()]
  step = 3

  # [修改] 傳入 step=3 取得最終階段的 Merge/Split Accuracy
  final_ma, final_sa = find_wrongly_merged(
      dataclass, result, folder_path, file_name, step=3
  )

  DCplot(
      dist,
      points,
      ND,
      rho,
      delta,
      ordrho,
      dc,
      nneigh,
      17,
      0.1,
      final_result,
      file_name,
      dataclass,
      folder_path,
      name,
      str(step),
  )

  # ------------------------------------------------------------------
  # STEP 4: 自動匯出詳細點資料與全域效能統計總表
  # ------------------------------------------------------------------
  exec_time = time.time() - start_time
  export_results(
      file=file,
      points=points,
      dataclass=dataclass,
      final_result=final_result,
      folder_path=folder_path,
      exec_time=exec_time,
      merge_accuracy=final_ma,
      split_accuracy=final_sa,
  )

  return final_result


# ==============================================================================
# 1. 第一階段專用函式 (Step 1: Local Peak Identification & Initial Micro-clusters)
# ==============================================================================


def findpeak(nneigh, delta):
  """Step 1.1: 利用拓撲層級與 Delta 值的動態變化，自動尋找局部密度峰值點 (Peaks)。"""
  Lv = build_levels(nneigh)
  peaks = []
  peak_levels = {}

  if len(Lv) < 2:
    print('層級不足，無法執行檢查')
    return peaks, peak_levels

  for level in range(1, len(Lv)):
    for p in Lv[level]:
      if p in peaks:
        continue

      max_delta_in = None
      for prev_level in range(level):
        for q in Lv[prev_level]:
          if nneigh[q] == p:
            if max_delta_in is None or delta[q] > max_delta_in:
              max_delta_in = delta[q]

      if max_delta_in is None:
        continue

      target = nneigh[p]
      if target == -1:
        peaks.append(p)
        peak_levels[p] = level
        continue

      if max_delta_in < delta[p]:
        peaks.append(p)
        peak_levels[p] = level

  return peaks, peak_levels


def build_levels(nneigh):
  """根據點對點的近鄰指向關係 (nneigh) 建立拓撲層級結構 (DAG Leveling)。"""
  n = len(nneigh)
  incoming_count = np.zeros(n, dtype=int)
  for i in range(n):
    if nneigh[i] != -1:
      incoming_count[nneigh[i]] += 1

  Lv = []
  queue = deque(np.where(incoming_count == 0)[0])
  visited = set(queue)

  while queue:
    current_level = list(queue)
    Lv.append(current_level)
    queue.clear()

    for node in current_level:
      next_node = nneigh[node]
      if next_node != -1 and next_node not in visited:
        incoming_count[next_node] -= 1
        if incoming_count[next_node] == 0:
          queue.append(next_node)
          visited.add(next_node)

  return Lv


def first_clusters(nneigh, peaks, ND, peak_levels):
  """Step 1.2: 依據找到的 Peak 點，將其餘資料點順著近鄰指向路徑 (nneigh) 歸類至對應 Peak 形成初始微群。"""
  clusters = []
  cluster_levels = []
  finished_num = set()

  for peak in peaks:
    clusters.append([peak])
    cluster_levels.append(peak_levels.get(peak, -1))

  for i in range(ND):
    if i not in peaks and i not in finished_num:
      cluster = [i]
      target = int(nneigh[i])

      while target not in peaks:
        if target == -1:
          break
        cluster.append(target)
        target = nneigh[int(target)]

      if target in peaks:
        for j in range(len(clusters)):
          if target == clusters[j][0]:
            for point in cluster:
              if point not in clusters[j]:
                clusters[j].append(point)
            break

      finished_num.update(cluster)

  return clusters, cluster_levels


# ==============================================================================
# 2. 第二階段專用函式 (Step 2: Micro-cluster Merging with R Index) - O(1) 加速版
# ==============================================================================


def External_Environment(rho, first_clusters_result, cluster_levels, dist, points):
  """Step 2: 依據相似度 R 進行層級間的微群合併。"""
  Rmed = []
  Rnow = 1
  Rpre = len(first_clusters_result)
  k = 1

  while Rpre > Rnow:
    unique_levels = sorted(set(cluster_levels))
    if len(unique_levels) == 1:
      break

    clu_rho = calculate_cluster_density(rho, first_clusters_result)
    min_level = unique_levels[0]
    higher_levels = unique_levels[1:]
    calcluster_dist = cal_dist(first_clusters_result, dist)

    rank_matrix = precompute_rank_matrix(calcluster_dist)

    level_dict = {lvl: [] for lvl in unique_levels}
    for i, lvl in enumerate(cluster_levels):
      level_dict[lvl].append(i)

    Rvalue = []
    merged_indices = set()

    for i in level_dict[min_level]:
      R = float('inf')
      minRq = -1
      for level in higher_levels:
        for j in level_dict[level]:
          dist_center_A_B = calculate_centroid(
              points, first_clusters_result[i], first_clusters_result[j]
          )
          dist_peak_A_B = dist[
              first_clusters_result[i][0], first_clusters_result[j][0]
          ]

          r = int(max(rank_matrix[i, j], rank_matrix[j, i]))

          if r > len(level_dict[min_level]):
            continue

          R_temp = calculate_R(dist_center_A_B, dist_peak_A_B, clu_rho[i], r)

          if R_temp < R:
            R = R_temp
            minRq = j

      if minRq != -1:
        Rvalue.append(R)
        first_clusters_result[minRq].extend(first_clusters_result[i])
        merged_indices.add(i)

    Rmed.append(np.median(Rvalue) if Rvalue else 0)
    Rpre = len(first_clusters_result)

    to_delete = sorted(list(merged_indices), reverse=True)
    for i in to_delete:
      del cluster_levels[i]
      del first_clusters_result[i]

    if k != 1:
      Rnow = len(first_clusters_result)
    k += 1

  return first_clusters_result


def precompute_rank_matrix(calcluster_dist):
  """將微群間距離矩陣轉為排名矩陣 Rank Matrix。"""
  calcluster_dist = np.asarray(calcluster_dist)
  ranks = np.argsort(np.argsort(calcluster_dist, axis=1), axis=1)
  return ranks


def calculate_centroid(points, clusters1, clusters2):
  """計算兩個微群重心之間的歐式距離 (支援任意 N 維)"""
  c1 = np.mean(points[clusters1], axis=0)
  c2 = np.mean(points[clusters2], axis=0)
  centroid_dist = np.linalg.norm(c1 - c2)
  return centroid_dist


def calculate_R(dist_center_A_B, dist_peak_A_B, clu_rho_A, r, eps=1e-10):
  """計算兩微群之間的相似度指標 Log(R_AB)。"""
  log_rho = math.log(max(clu_rho_A, eps))
  log_dist_center = math.log(max(dist_center_A_B, eps))
  log_dist_peak = math.log(max(dist_peak_A_B, eps))
  log_r = math.log(max(r, eps))

  return log_rho + log_dist_center + log_dist_peak + log_r


def calculate_cluster_density(rho, clusters_result):
  """計算每個微群的平均密度。"""
  clu_rho = []
  for cluster in clusters_result:
    total_rho = sum(rho[idx] for idx in cluster)
    clu_rho.append(total_rho / len(cluster))
  return clu_rho


# ==============================================================================
# 3. 第三階段專用函式 (Step 3: Single-Linkage Hierarchical Convergence)
# ==============================================================================


def hier_clustering(second_clusters_result, dist, rho, ND, t):
  """Step 3: 使用 Single-Linkage 階層式分群將第二階段的微群收斂至指定的群數 t。"""
  calcluster_dist = []
  for i in range(len(second_clusters_result) - 1):
    for j in range(i + 1, len(second_clusters_result)):
      calcluster_dist.append(
          cal_clusters_distance(second_clusters_result, dist, i, j)
      )

  calcluster_dist = np.array(calcluster_dist)
  Z = linkage(calcluster_dist, 'single')
  hier_result = fcluster(Z, t=t, criterion='maxclust')

  final_cluster = hierlabelpoint(hier_result, second_clusters_result, ND)
  return final_cluster


def cal_clusters_distance(fcluster_result, dist, i, j):
  """計算兩個微群之間的 Single-Linkage 最短點對距離。"""
  c1 = fcluster_result[i]
  c2 = fcluster_result[j]
  min_dist = dist[c1[0], c2[0]]
  for ii in c1:
    for jj in c2:
      curr_dist = dist[ii, jj]
      if curr_dist < min_dist:
        min_dist = curr_dist
  return min_dist


def cal_dist(cluster, dist):
  """計算所有微群兩兩之間的距離矩陣。"""
  calcluster_dist = []
  for i in range(len(cluster)):
    calcluster = []
    for j in range(0, len(cluster)):
      calcluster.append(cal_clusters_distance(cluster, dist, i, j))
    calcluster_dist.append(calcluster)
  return np.array(calcluster_dist)


def hierlabelpoint(label, clusters_result, ND):
  """將微群層級的分群標籤映射回每個原始資料點。"""
  final_cluster = np.zeros(ND, dtype=int)
  for i in range(len(label)):
    for j in clusters_result[i]:
      final_cluster[j] = label[i]
  return final_cluster


# ==============================================================================
# 4. 評估與標籤轉換工具函式 (Evaluation Metrics & Utility Functions)
# ==============================================================================


def find_wrongly_merged(
    dataclass, first_clusters_result, save_path, file_name, step
):
  """分群結果品質診斷指標：

  - Merge Accuracy (MA): 防過度合併率
  - Split Accuracy (SA): 防過度拆分率
  """
  total_points = sum(len(cluster) for cluster in first_clusters_result)
  wrongly_merged_points = 0
  wrongly_split_points = 0

  for cluster in first_clusters_result:
    label_counts = Counter(dataclass[point] for point in cluster)
    if len(label_counts) > 1:
      wrongly_merged_points += sum(label_counts.values())

  true_clusters = {}
  for i, label in enumerate(dataclass):
    if label not in true_clusters:
      true_clusters[label] = []
    true_clusters[label].append(i)

  assigned_cluster = {}
  for cluster_id, cluster in enumerate(first_clusters_result):
    for point in cluster:
      assigned_cluster[point] = cluster_id

  for true_label, true_points in true_clusters.items():
    assigned_clusters = set(assigned_cluster[p] for p in true_points)
    if len(assigned_clusters) > 1:
      wrongly_split_points += len(true_points) - max(
          Counter(assigned_cluster[p] for p in true_points).values()
      )

  merge_accuracy = 1 - (wrongly_merged_points / total_points)
  split_accuracy = 1 - (wrongly_split_points / total_points)

  print(
      '=================================== accuracy'
      ' ============================================'
  )
  print(f'錯誤合併的點數: {wrongly_merged_points}')
  print(f'Merge Accuracy (防過度合併率): {merge_accuracy:.2%}')
  print(f'該合併卻未合併的點數: {wrongly_split_points}')
  print(f'Split Accuracy (防過度拆分率): {split_accuracy:.2%}')
  print(
      '=================================== accuracy'
      ' ============================================'
  )

  os.makedirs(save_path, exist_ok=True)
  output_file = os.path.join(save_path, 'accuracy_results.csv')

  # 格式化步驟名稱 (如 1 -> 'step1')
  step_str = f'step{step}' if isinstance(step, int) else str(step)

  # [修改] 將 'id': [name] 改為 'step': [step_str]
  df = pd.DataFrame({
      'step': [step_str],
      'file_name': [file_name],
      'Total Points': [total_points],
      'Wrongly Merged Points': [wrongly_merged_points],
      'Merge Accuracy': [merge_accuracy],
      'Wrongly Split Points': [wrongly_split_points],
      'Split Accuracy': [split_accuracy],
  })

  file_exists = os.path.exists(output_file)
  df.to_csv(output_file, mode='a', index=False, header=not file_exists)

  return merge_accuracy, split_accuracy


def labelpoint(clusters_result, ND):
  """將二維的 clusters_result 轉換為長度為 ND 的標籤陣列。"""
  final_cluster = np.zeros(ND, dtype=int)
  for i in range(len(clusters_result)):
    for j in clusters_result[i]:
      final_cluster[j] = i + 1
  return final_cluster


# ==============================================================================
# 5. 自動匯出報表工具函式 (Exporting Detailed CSV & Summary Reports)
# ==============================================================================


def map_clusters_to_labels(dataclass, final_result):
  """將數值型的預測群集 ID (例如 1, 2, 3) 依照多數決原則，映射回真實文字標籤 (例如 'Iris-setosa')"""
  df = pd.DataFrame({'true': dataclass, 'pred': final_result})

  cluster_map = {}
  for cluster_id in np.unique(final_result):
    if cluster_id <= 0:
      cluster_map[cluster_id] = 'Noise'
      continue

    subset = df[df['pred'] == cluster_id]
    if len(subset) > 0:
      most_common_label = subset['true'].mode()[0]
      cluster_map[cluster_id] = most_common_label

  print('📌 [多數決映射] 原始群集與文字類別對照表:', cluster_map)

  return [cluster_map.get(cid, 'Unknown') for cid in final_result]


def export_results(
    file,
    points,
    dataclass,
    final_result,
    folder_path,
    exec_time,
    merge_accuracy,
    split_accuracy,
):
  """自動產出兩層級分群報告 (包含 NaN 自動清洗與多數決類別映射)"""
  file_name = os.path.basename(file)
  base_name = os.path.splitext(file_name)[0]
  os.makedirs(folder_path, exist_ok=True)

  # 0. NaN 自動防護與資料預處理
  points_clean = np.nan_to_num(points, nan=0.0)

  if dataclass is None:
    dataclass_clean = np.array(['Missing_Label'] * len(points_clean))
  else:
    dataclass_series = pd.Series(dataclass).reset_index(drop=True)
    dataclass_clean = dataclass_series.fillna('Missing_Label').values

  final_result_series = pd.Series(final_result).reset_index(drop=True)
  final_result_clean = final_result_series.fillna(-1).values

  # 多數決映射
  predicted_class_labels = map_clusters_to_labels(
      dataclass_clean, final_result_clean
  )

  # 1. 匯出點層級資料表
  num_dims = points_clean.shape[1] if len(points_clean.shape) > 1 else 1
  feat_cols = [f'Dim_{i+1}' for i in range(num_dims)]

  df_points = pd.DataFrame(points_clean, columns=feat_cols)
  df_points['True_Class'] = dataclass_clean
  df_points['Predicted_Cluster'] = final_result_clean
  df_points['Predicted_Class'] = predicted_class_labels

  detail_path = os.path.join(folder_path, f'{base_name}_clustered_result.csv')
  df_points.to_csv(detail_path, index_label='Point_ID')
  print(f'已匯出點層級分群結果至: {detail_path}')

  # 2. 安全計算通用學術指標 (ARI, NMI)
  try:
    ari = adjusted_rand_score(dataclass_clean, final_result_clean)
    nmi = normalized_mutual_info_score(dataclass_clean, final_result_clean)
  except Exception as e:
    print(f'⚠️ 計算學術指標時跳過例外 ({e})，設為預設值 0.0000')
    ari, nmi = 0.0, 0.0

  num_clusters = len(np.unique(final_result_clean))

  summary_path = os.path.join(folder_path, 'clustering_summary.csv')
  summary_data = {
      'Dataset': [base_name],
      'Points_N': [len(points_clean)],
      'Dimension_D': [num_dims],
      'Predicted_Clusters': [num_clusters],
      'Merge_Accuracy': [f'{merge_accuracy:.2%}'],
      'Split_Accuracy': [f'{split_accuracy:.2%}'],
      'ARI': [f'{ari:.4f}'],
      'NMI': [f'{nmi:.4f}'],
      'Exec_Time_Sec': [f'{exec_time:.3f}'],
  }

  df_summary = pd.DataFrame(summary_data)
  file_exists = os.path.exists(summary_path)
  df_summary.to_csv(summary_path, mode='a', index=False, header=not file_exists)
  print(f'已更新全域分群統計總表至: {summary_path}')