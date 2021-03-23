# Pruning-Filters
# 剪枝--Pruning-Filters
# 下面介绍第二篇论文
Pruning Filters for Efficient ConvNets

主要思想：
由于CNN通常在不同的 Filter 和特征信道之间具有显着的冗余，论文中通过修剪 Filter 来减少CNN的计算成本。与在整个网络中修剪权重相比， Filter 修剪是一种自然结构化的修剪方法，不会引入稀疏性，因此不需要使用稀疏库或任何专用硬件。通过减少矩阵乘法的次数，修剪 Filter 的数量与加速度直接相关，这很容易针对目标加速进行调整。

