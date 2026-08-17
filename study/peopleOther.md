# Other People in AI

## Yangqing Jia

CEO of https://intentlab.ai/

https://www.linkedin.com/in/yangqing-jia/

## Jiarui Fang方佳瑞

https://github.com/feifeibear

- SGLang：LLM推理引擎发展新方向 (old, 2024): https://zhuanlan.zhihu.com/p/711378550

就像当年深度学习框架发展经历了
  cuda-convnet（2012）-> Caffe （2014）-> TensorFlow（2016）-> PyTorch（2017）
长达五年跨度若干的阶段，大模型推理框架一步到位也不太现实。vLLM有点像当年的Caffe阶段，比如下面几个方面
- 高性能：Caffe率先做到了在GPU上高性能运行，它写了大量cuda
  kernel，比如im2col后调用cuBLAS来优化卷积算子。同时很早支持数据并行方式多卡训练。其性能优势让很多人从Theano切换到Caffe，正如vLLM的Paged
  Attention打开了吞吐天花板。
- 开源影响力：工程和学术界大量model zoo都基于caffe开发，比如很多年的ImageNet比赛的模型都用Caffe，其中就包括CVPR 16' best
  paper Resnet。和vLLM现在的地位颇为相似。
- 学术机构维护：Caffe和vLLM都诞生于UCB实验室项目。Caffe的主程Yangqing Jia后来去Facebook，写了Caffe2后来合并进了PyTorch项目。
