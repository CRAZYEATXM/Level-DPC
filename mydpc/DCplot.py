# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd


def DCplot(
    dist,
    points,
    ND,
    rho,
    delta,
    ordrho,
    dc,
    nneigh,
    rhomin,
    deltamin,
    labels,
    file,
    true_labels,
    folder_path,
    name,
    step,
):
  """繪製數據點並使用不同顏色表示每個聚類 (僅支援 2 維資料繪圖，高維自動跳過)。"""

  # 1. 確保 points 為 2D NumPy 陣列結構 (N, D)
  points = np.atleast_2d(points)
  num_dims = points.shape[1]

  if num_dims != 2:
    print(
        f'ℹ️ 資料集維度為 {num_dims}D (特徵形狀: {points.shape})，非 2D 平面資料，已自動跳過'
        f' Step {step} 畫圖。'
    )
    return

  # 取出 2D 資料點 X 與 Y 座標
  X_plot = points[:, 0]
  Y_plot = points[:, 1]

  step_title = f'my step {step}'
  os.makedirs(folder_path, exist_ok=True)

  # 內部輔助函式：繪製特定的 Scatter 與 Cluster 內部連接線
  def _draw_sub_plot(ax, cluster_labels, title_text):
    # [關鍵修復] 使用 pd.factorize 將字串標籤轉為數字類別代碼 (如 0, 1, 2...)
    # 避免 Matplotlib 將字串 '2' 誤判為灰階顏色值而報錯
    numeric_labels, _ = pd.factorize(cluster_labels)

    ax.scatter(X_plot, Y_plot, c=numeric_labels, cmap='tab10', s=10)

    # 建立群組內點對點的連線
    lines = []
    for cluster_id in np.unique(cluster_labels):
      idx = np.where(cluster_labels == cluster_id)[0]
      sub_points = points[idx]

      if len(sub_points) > 1:
        lines.extend([
            (sub_points[i], sub_points[i + 1])
            for i in range(len(sub_points) - 1)
        ])

    if lines:
      line_collection = LineCollection(
          lines, colors='black', linewidths=0.5, alpha=0.5
      )
      ax.add_collection(line_collection)

    ax.set_title(title_text)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

  # 2. 繪製並儲存「True Labels」單獨畫布
  fig1, ax1 = plt.subplots(figsize=(6, 6))
  _draw_sub_plot(ax1, true_labels, 'True Labels')
  save_path1 = os.path.join(folder_path, f'{file}_True_Labels.png')
  plt.savefig(save_path1, dpi=300, format='PNG', bbox_inches='tight')
  plt.show()
  plt.close(fig1)

  # 3. 繪製並儲存「True Labels vs DPC Step Result」雙圖對比
  fig2, (ax2_1, ax2_2) = plt.subplots(1, 2, figsize=(12, 6))
  _draw_sub_plot(ax2_1, true_labels, 'True Labels')
  _draw_sub_plot(ax2_2, labels, step_title)
  plt.tight_layout()
  save_path2 = os.path.join(folder_path, f'{file}_Step_{step}_Compare.png')
  plt.savefig(save_path2, dpi=300, format='PNG', bbox_inches='tight')
  plt.show()
  plt.close(fig2)

  # 4. 繪製並儲存「DPC Step Result」單獨畫布
  fig3, ax3 = plt.subplots(figsize=(6, 6))
  _draw_sub_plot(ax3, labels, step_title)
  save_path3 = os.path.join(
      folder_path, f'{file}_{name}_Step_{step}_Result.png'
  )
  plt.savefig(save_path3, dpi=300, format='PNG', bbox_inches='tight')
  plt.show()
  plt.close(fig3)

  print(f'🖼️ 成功繪製並儲存 Step {step} 2D 分群視覺化圖片！')