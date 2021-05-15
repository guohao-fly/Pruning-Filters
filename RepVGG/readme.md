# RepVGG: Making VGG-style ConvNets Great Again
[RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/abs/2101.03697)
## Abstract
&ensp; 本文提出一种简单而强有力的CNN架构RepVGG，在推理阶段，它具有与VGG类似的架构，而在训练阶段，它则具有多分支架构体系，这种训练-推理解耦的架构设计源自一种称之为“重参数化(re-parameterization)”的技术。  
![image](https://user-images.githubusercontent.com/80331072/118351670-c54c6100-b58f-11eb-9ca1-015d65a59327.png)

## 优势
1.Fast：相比VGG，现有的多分支架构理论上具有更低的Flops，但推理速度并未更快。
2.Memory-economical：多分支结构是一种内存低效的架构，这是因为每个分支的结构都需要在Add/Concat之前保存，这会导致更大的峰值内存占用；而plain模型则具有更好的内存高效特征。  
3.Flexible：多分支结构会限制CNN的灵活性，比如ResBlock会约束两个分支的tensor具有相同的形状；与此同时，多分支结构对于模型剪枝不够友好。

## 基本结构
&ensp; 本文所设计的RepVGG则是受ResNet启发得到，尽管多分支结构对于推理不友好，但对于训练友好，作者将RepVGG设计为训练时的多分支，推理时单分支结构。作者参考ResNet的identity与1 × 1分支，设计了如下形式模块：  
![image](https://user-images.githubusercontent.com/80331072/118351780-6f2bed80-b590-11eb-96ef-9b5ab5f19243.png)  
其中，g(x),f(x) 分别对应为1×1,3×3卷积。  
![image](https://user-images.githubusercontent.com/80331072/118351828-b4501f80-b590-11eb-836e-a9e656c90e20.png)  
在训练阶段deploy=False，通过简单的堆叠上述模块构建CNN架构；而在推理阶段deploy=True，上述模块可以轻易转换为y=h(x)形式，且h(x)的参数可以通过线性组合方式从已训练好的模型中转换得到。  

## Re-param for Plain Inference-time Model
如何将三个分支合并呢？  
![image](https://user-images.githubusercontent.com/80331072/118352049-dc8c4e00-b591-11eb-8c3f-bc2261d325e8.png)  

![image](https://user-images.githubusercontent.com/80331072/118352090-1fe6bc80-b592-11eb-8f08-21acc89cb2c0.png)
由此，一个基本结构可以表示为：  
![image](https://user-images.githubusercontent.com/80331072/118352128-57edff80-b592-11eb-8abb-a9b5e99a138a.png)  
bn层函数，形式上为：  
![image](https://user-images.githubusercontent.com/80331072/118352187-9aafd780-b592-11eb-956a-d80b52bd40ef.png)  
首先将每个BN及其前一conv层转换为一个带有偏差向量的conv。设{W0,b0}为{W，µ，σ，γ，β}转换的核和偏置，有  
![image](https://user-images.githubusercontent.com/80331072/118352219-ce8afd00-b592-11eb-80b5-1cf1a4718b08.png)  
那就很容易验证了  
![image](https://user-images.githubusercontent.com/80331072/118352236-eb273500-b592-11eb-899c-a8e13090acef.png)  




