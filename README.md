<!--
 * @Author: Roger
 * @Date: 2020-12-07 13:27:14
 * @LastEditors  : Roger
 * @LastEditTime : 2020-12-20 09:54:44
 * @Description: file content
-->
# Code about ABSA
## 介绍
该仓库[https://github.com/ROGERDJQ/FastALSC](https://github.com/ROGERDJQ/FastALSC),是关于ABSA内Aspect-Level Sentiment Analysis(ALSC)任务的实现代码。在[https://gitee.com/ROGERDJQ/FastALSC](https://gitee.com/ROGERDJQ/FastALSC)同步更新。
## 进展
1. 已完成对论文[Aspect-Level Sentiment Analysis Via Convolution over Dependency Tree](https://www.aclweb.org/anthology/D19-1569/)的复现。可参看CDT中的详细说明。
## Note
ABSA 的模型通常有各自的预处理方法，不同的预处理方法对任务性能影响较大。为了尽量达到与原论文相近的效果，原始论文提出模型所使用的数据如果能在其仓库内找到，将会放置在对应模型文件夹的dataset文件夹内。