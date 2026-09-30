防過度合併率 (Merge Accuracy, MA)核心意義：評估微群（Micro-cluster）的純度（確保小群內沒有混入異類）。審查邏輯（極度嚴苛）：採「零容忍」機制。只要某個小群內混入 >1 種真實標籤，程式會認定該群已遭污染，整群的所有資料點將全部計為錯誤合併點。

防過度拆分率 (Split Accuracy, SA)核心意義：評估真實類別的歸隊完整度（代表演算法真正的「實際錯分點數」）。審查邏輯：計算該類別中脫離大隊、未能成功合併的落單點數。

Merge Accuracy (MA)
Core Concept: Evaluates the purity of micro-clusters (ensuring no samples from different classes are mixed into a small cluster).
Evaluation Logic (Strict): Adopts a "zero-tolerance" mechanism. As long as a micro-cluster contains >1 ground-truth label, the cluster is considered polluted, and all data points in that cluster are counted as wrongly merged points.

Split Accuracy (SA)
Core Concept: Evaluates the completeness of ground-truth classes returning to their group (representing the algorithm's true "actual misclassified points").
Evaluation Logic: Calculates the number of stray points in that class that broke away from the main group and failed to merge successfully.
