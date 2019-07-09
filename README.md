# FastGCN

* ICLR2018 paper: ["FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling"](https://openreview.net/forum?id=rytstxWAW&noteId=ByU9EpGSf)

[toc]


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


