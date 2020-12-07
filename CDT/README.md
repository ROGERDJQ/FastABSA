<!--
 * @Author: Roger
 * @Date: 2020-12-03 01:40:29
 * @LastEditors: Roger
 * @LastEditTime: 2020-12-07 13:35:30
 * @Description: file content
-->
# 基于fastNLP的Aspect-Level Sentiment Analysis实践 


## 介绍

代码为Aspect-Level Sentiment Analysis Via Convolution over Dependency Tree(CDT)的[fastNLP](https://github.com/fastnlp/fastNLP)实现，基于论文[Aspect-Level Sentiment Analysis Via Convolution over Dependency Tree](https://www.aclweb.org/anthology/D19-1569/)。文章结合lstm与GCN，利用Dependency Tree的结构信息，提高了在Aspect-Term Sentiment Analysis 任务上的结果。

## Requirements
   - [fastNLP 0.5.0](https://github.com/fastnlp/fastNLP) 
   - [fitlog](https://github.com/fastnlp/fitlog)
   - PyTorch

## Note
- > 我们建议运行前安装并使用fitlog，使用 命令行fitlog init xxx初始化文件夹后运行代码，否则代码文件中的fitlog.*语句可能会报错
- > 如果在调试时希望不运行fitlog相关代码，请保留入口处的 fitlog.debug()
- >已经将原CDT的数据集放在内部的dataset文件夹内，可在复现时使用。外部的Dataset内的数据使用了不同的预处理方法，在具体数据上可能与dataset内数据有诸多不同。
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
- 训练需要加载GloVe vectors [glove.840B.300d.zip](https://nlp.stanford.edu/projects/glove/)。fastNLP会自动加载常用的Glove 。
### 训练
训练模型，运行：
```python
python train.py --data_dir your/data/dir/ 
```
更多的参数设置参看代码内说明。
训练模型将会保存在`save_dir`。
## 结果
原论文共实验了四个数据集。在训练时分别进行了实验。

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
建图时tree代码为原代码中实现，处理后数据文件与参数与原repo中结构相同。

## TODO

## References
- [Aspect-Level Sentiment Analysis Via Convolution over Dependency Tree](https://www.aclweb.org/anthology/D19-1569/), EMNLP 2019.
- [CDT_ABSA](https://github.com/sunkaikai/CDT_ABSA)
- [fastNLP 中文文档](https://fastnlp.readthedocs.io/zh/latest/)
- [fitlog 中文文档](https://fitlog.readthedocs.io/zh/latest/)
