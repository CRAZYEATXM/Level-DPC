import numpy as np


def rhodelta(dist, xxdist, ND, N, percent):
    """計算 Density Peak 分群所需的 local density (rho) 與 distance to higher density (delta)
      
    參數：
    - dist: ND x ND 的兩兩距離矩陣
    - xxdist: 所有不重複且大於 0 的距離陣列
    - ND: 資料點數量
    - N: 不重複距離的數量
    - percent: 計算截斷距離 (dc) 的百分比參數 (例: 2 代表前 2%)
      
    回傳：
    (rho, delta, ordrho, dc, nneigh)
    """
    print('Calculating rho and delta...')
    print(f'Average percentage of neighbours: {percent:5.6f}%')
      
    # 1. 計算截斷距離 Cutoff Distance (dc)
    position = int(round(N * percent / 100))
    position = min(max(position, 0), len(xxdist) - 1)  # 防呆：防止陣列越界
    dc = xxdist[position]
    print(f'Computing Rho with Gaussian kernel of radius (dc): {dc:12.6f}\n')
      
    # 2. 向量化計算 Gaussian Kernel 局部密度 Rho (省去雙重迴圈，速度提升數十倍)
    # exp(- (dist / dc)^2) 的全矩陣計算
    kernel_matrix = np.exp(-((dist / dc) ** 2))
    # 按列加總 (axis=1)，再扣掉對角線自身 (dist[i,i]=0 導致 exp(0)=1)
    rho = np.sum(kernel_matrix, axis=1) - 1.0
      
    # 3. 將資料點按 Rho 降冪排序 (密度最高的點索引在 ordrho[0])
    ordrho = np.argsort(-rho)
      
    # 4. 初始化 Delta 與 Nearest Neighbor (nneigh)
    delta = np.zeros(ND)
    nneigh = np.zeros(ND, dtype=int)
      
    # 密度最高的點沒有「比它更高密度的點」，預設設為 -1
    nneigh[ordrho[0]] = -1
      
    # 5. 優化計算 Delta 與最近高密度鄰居 (Vectorized Slicing)
    for ii in range(1, ND):
      curr_idx = ordrho[ii]
      # 取得所有密度比當前點高（排序在 ii 之前）的點之索引
      higher_density_indices = ordrho[:ii]
      
      # 計算當前點到所有更高密度點的距離
      dist_to_higher = dist[curr_idx, higher_density_indices]
      
      # 找出最小距離與對應的點索引
      min_pos = np.argmin(dist_to_higher)
      delta[curr_idx] = dist_to_higher[min_pos]
      nneigh[curr_idx] = higher_density_indices[min_pos]
      
    # 密度最高的點，其 Delta 定義為所有點中最大的 Delta 值
    delta[ordrho[0]] = np.max(delta)
      
    return (rho, delta, ordrho, dc, nneigh)

