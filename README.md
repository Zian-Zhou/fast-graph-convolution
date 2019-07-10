# FastGCN

* ICLR2018 paper: ["FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling"](https://openreview.net/forum?id=rytstxWAW&noteId=ByU9EpGSf)

* 根据[github开源源码](https://github.com/matenure/FastGCN.git)实现

## 代码环境

基于tensorflow， 版本为1.13.1， python版本3.7.3，Win10上测试通过。
需要安装的相关模块有：

```
tensorflow
fire
scipy
numpy
networkx
```


## 文件目录说明

```
	--data:
		graph数据存放目录
	
	--models:
		模型存放目录
	
	--scripts:
		脚本示例目录

	--snapshot:
		模型训练结果存储目录，分目录为 model_ID:

		----pubmed_gcn_appr_default:
				表示在数据集pubmed上，采用gcn_appr模型，ID为default的训练模型结果：
				包含config.pkl、model_acc.ckpt.data、model_acc.ckpt.index、model_acc.ckpt.meta
				以及分目录finetune_model：微调训练新模型
				------20190704_20_08:
					按训练时间划分目录

	--utils:
		训练模型所需要的代码

	config.py:
		参数仓库

	finetune.py:
		微调模型代码

	main.py:
		主要训练代码
```


## 训练实例

### [For pubmed]  


* 原论文指标：0.880  


```
#训练模型：

python main.py main --model=gcn_appr --dataset=pubmed --ID=default --rank0=100 --rank1=100 --hidden1=16 --learning_rate=0.001
```

训练完后，模型将存储于./snapshot/pubmed_gcn_appr_default/ 目录下。对应模型acc为0.85200.  

```
#微调模型：

python finetune.py finetune --model_id=gcn_appr_default --lr_decay=1.1
```

微调训练完后，模型将存储于./snapshot/pubmed_gcn_appr_default/finetune_model/20190705_13_13/ 目录下，对应新的模型acc为0.88200.

* 上面示例采用的是双层GCN，如果采用Dense+单层GCN效果更佳，并且训练速度更快：acc=0.883 ——>> 0.89100

### [For Cora]  


* 原论文指标：0.850  


```
#训练模型：

python main.py main --model=dense_gcn_appr  --ID=h64 --rank0=100 --dataset=cora --hidden1=64
```


训练完成后，模型存储于./FastGCN-Mine/snapshot/cora_dense_gcn_appr_h64/ 目录下，对应模型acc为0.846.


```
#微调模型：

python finetune.py finetune --dataset_model_id=cora_dense_gcn_appr_h64 --lr_decay=0.8 --rank0=100
```


微调训练完后，模型存储于./FastGCN-Mine/snapshot/cora_dense_gcn_appr_h64/finetune_model/20190705_14_32/ 目录下，对应模型acc为0.850.  


* 双层GCN没有复现到论文相同效果。  


### [For Reddit]  


待完成  


## 改进模型

### 【dense+GCN】

* 1、全连接加单层GCN  

* 2、  
在Pubmed数据集上表现相比于双层GCN或者仅用全连接（MLP）效果都有所提升；  
在Cora数据集上表现不如双层GCN模型，但相比较MLP而言效果却有所提升。  

综合上面提到了所有模型的表现，猜测在Pubmed数据集上，全连接效果比GCN好；在Cora数据集上GCN比MLP好。实验也验证了这个猜测。

```
python main.py main --model=mlp --ID=compare --dataset=pubmed
python main.py main --model=dense_gcn_appr --ID=compare --dataset=pubmed
python main.py main --model=gcn_appr --ID=compare --dataset=pubmed

python main.py main --model=mlp --ID=compare --dataset=cora
python main.py main --model=dense_gcn_appr --ID=compare --dataset=cora
python main.py main --model=gcn_appr --ID=compare --dataset=cora
```

* 可能的原因：

```
可能是这两个数据集不同的特点所造成的——cora的节点特征维度大，但是特别稀疏，仅仅用自身的特征不足以判断类别，因此引入了周围节点信息的GCN模型效果会比仅仅利用自身信息的MLP模型效果好；而pubmed的节点特征维度不大，可能比较稠密，并且自身的特征足以判断类别，因此引入周围节点信息的GCN模型也带来了不必要的噪声，导致效果还不如MLP。
```

### 【mlp_gcn_highway_mix】

* 1、两个分支：  
mlp分支+GCN分支，主要区别就在于是否引入拓扑信息  

* 2、highway：  
两个分支通过一个门控函数做加权和，门控函数的设计同HighwayNetwork（或者说同GRU、LSTM）。  

* 3、设计动机：  
希望网络可以学习根据节点的特征判断是否引入周围节点的信息。做两个分支的目的也是在于显式的区分是否引入拓扑信息。  

* 4、效果：
Pubmed数据集：模型效果比较dense+GCN、双层GCN以及MLP都有所提升；  
Cora数据集：模型效果比较mlp有所提升，但是远不如dense+gcn、双层GCN。

```
python main.py main --model=mlp_gcn_highway_mix --ID=compare --dataset=pubmed

python main.py main --model=mlp_gcn_highway_mix --ID=compare --dataset=cora
```

* 5、可能原因：

```
可能还是由于数据集自身的分布情况造成的。如果数据集中既有特征丰富的节点也有特征稀疏的节点，那么可能这个网络模型中的门控函数能够学习到这种潜在规律；如果数据集没有这样的分布特点，网络模型可能在学习过程中会出现懒惰学习甚至不学习的倾向。
```
