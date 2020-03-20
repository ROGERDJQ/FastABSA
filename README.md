# 基于fastNLP的Aspect-Level Sentiment Analysis实践 


## 介绍

代码为Aspect-Level Sentiment Analysis Via Convolution over Dependency Tree(CDT)的[fastNLP](https://github.com/fastnlp/fastNLP)实现，基于论文[Aspect-Level Sentiment Analysis Via Convolution over Dependency Tree](https://www.aclweb.org/anthology/D19-1569/)。文章结合lstm与GCN，利用Dependency Tree的结构信息，提高了在Aspect-Term Sentiment Analysis 任务上的结果。原始实现代码在[CDT_ABSA](https://github.com/sunkaikai/CDT_ABSA)。

## Requirements
   - [fastNLP 0.5.0](https://github.com/fastnlp/fastNLP) 
   - [fitlog](https://github.com/fastnlp/fitlog)
   - PyTorch

## Run
> 运行前建议安装并使用fitlog初始化文件夹，否则fast_gcn代码文件中的fitlog*语句可能会报错
### 数据准备
- 接受的数据格式为json，结构如：
```json
{"token": ["Boot","time",…],
"pos": ["NN","NN",…],
"head": [2, 4,…],
"deprel": ["compound","nsubj",…],
"aspects": [{
        "term": ["Boot","time"],
        "from": 0,
        "to": 2,
        "polarity": "positive"}]}
```
- 训练需要加载GloVe vectors [glove.840B.300d.zip](https://nlp.stanford.edu/projects/glove/),加载后，将位置赋给参数`glove_dir`，具体参看代码内说明。
### 训练
训练模型，运行：
```python
python fast_gcn.py --data_dir your/data/dir/ --glove_dir your/glove/dir
```
更多的参数设置参看代码内说明。
训练模型将会保存在`save_dir`。
## 结果
原论文共实验了四个数据集。在训练时也分别进行了实验。

<table>
   <tr>
      <td></td>
      <td colspan="2">Rest14</td>
      <td colspan="2">Laptop</td>
      <td colspan="2">Twitter</td>
      <td colspan="2">Rest16</td>
   </tr>
   <tr>
      <td>Model</td>
      <td>Acc</td>
      <td>F1</td>
      <td>Acc</td>
      <td>F1</td>
      <td>Acc</td>
      <td>F1</td>
      <td>Acc</td>
      <td>F1</td>
   </tr>
   <tr>
      <td>原论文</td>
      <td>82.30</td>
      <td>74.02</td>
      <td>77.19</td>
      <td>72.99</td>
      <td>74.66</td>
      <td>73.66</td>
      <td>85.58</td>
      <td>69.93</td>
   </tr>
   <tr>
      <td>本代码</td>
      <td>82.66</td>
      <td>74.11</td>
      <td>77.06</td>
      <td>73.27</td>
      <td>74.29</td>
      <td>72.78</td>
      <td>85.56</td>
      <td>66.11</td>
   </tr>
   <tr>
      <td></td>
   </tr>
</table>

## Disclaimer
建图时tree代码为原代码中实现，处理后数据文件与参数与原repo中结构相同，见[CDT_ABSA](https://github.com/sunkaikai/CDT_ABSA)。

## TODO

## References
- [Aspect-Level Sentiment Analysis Via Convolution over Dependency Tree](https://www.aclweb.org/anthology/D19-1569/), EMNLP 2019.
- [CDT_ABSA](https://github.com/sunkaikai/CDT_ABSA)
- [fastNLP 中文文档](https://fastnlp.readthedocs.io/zh/latest/)
- [fitlog 中文文档](https://fitlog.readthedocs.io/zh/latest/)
