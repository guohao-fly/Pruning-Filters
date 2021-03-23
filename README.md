# Pruning-Filters
# 剪枝--Pruning-Filters
下面介绍第二篇论文：
Pruning Filters for Efficient ConvNets

主要思想：
由于CNN通常在不同的 Filter 和特征信道之间具有显着的冗余，论文中通过修剪 Filter 来减少CNN的计算成本。与在整个网络中修剪权重相比， Filter 修剪是一种自然结构化的修剪方法，不会引入稀疏性，因此不需要使用稀疏库或任何专用硬件。通过减少矩阵乘法的次数，修剪 Filter 的数量与加速度直接相关，这很容易针对目标加速进行调整。

![image](https://user-images.githubusercontent.com/80331072/112116080-dbbfe700-8bf4-11eb-8045-bac5bbc938c7.png)

如上图所示，删除一个 Filter 就能减少一个输出特征图，同时特征图对应的接下来卷积操作同样可以删除掉。

修剪Filter步骤：
1)计算 Filter 中所有权值的绝对值之和
2)根据求和大小排列 Filter
3)删除数值较小的 Filter （权重数值越小，代表权重的重要性越弱）
4)对删除之后的 Filter 重新组合，生成新的Filter矩阵

多层同时修剪：
作者给出了2中修剪思路：
1)独立修剪：修剪时每一层是独立的。
2)贪心修剪：修剪时考虑之前图层中删除的 Filter 。
两种方法的区别：独立修剪在计算（求权重绝对值之和）时不考虑上一层的修剪情况，所以计算时下图中的黄点仍然参与计算；贪心修剪计算时不计算已经修剪过的，即黄点不参与计算。
结果证明第二种方法的精度高一些。

![image](https://user-images.githubusercontent.com/80331072/112116392-36594300-8bf5-11eb-89cb-968c580db546.png)

![image](https://user-images.githubusercontent.com/80331072/112116453-453ff580-8bf5-11eb-9d1c-4d929aca1d47.png)
