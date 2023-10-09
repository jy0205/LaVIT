# LaVIT: Unified Language-Vision Pretraining in LLM with Dynamic Discrete Visual Tokenization
This is the official repository for the multi-modal large langauge model: **LaVIT**.

[[`arXiv`](https://arxiv.org/abs/2309.04669)] [[`BibTeX`](#Citing)]

## Introduction
We propose **LaVIT**, a new general-purpose multi-modal foundation model that inherits the successful learning paradigm of LLM: predicting the next image / text token in an auto-regressive manner. LaVIT introduce a well-designed visual tokenizer to translate the non-linguistic image into a sequence of discrete tokens like a foreign language that LLM can read. Hence, both images and texts can be handled simultaneously under the unified generative objective. For more technical details, please refer to our [paper](https://arxiv.org/abs/2309.04669).

<div align="center">
  <img src="assets/pipeline.png"/>
</div><br/>


After pre-training, LaVIT can serve as a multi-modal generalist to perform both multi-modal comprehension and generation without further fine-tuning. Specifically, it has the following capabilities
* read image contents and answer the questions.

<div align="center">
  <img src="assets/understanding.jpg"/>
</div><br/>

* Text-to-image creation.

<div align="center">
  <img src="assets/text2image.png"/>
</div><br/>

* Image synthesis via Multi-modal Prompt.

<div align="center">
  <img src="assets/multi_modal.png"/>
</div><br/>

Our model weights and codes will be released in the next few days.

## <a name="Citing"></a>Citation
Consider giving this repository a star and cite LaVIT in your publications if it helps your research.

```
@article{jin2023unified,
  title={Unified Language-Vision Pretraining in LLM with Dynamic Discrete Visual Tokenization},
  author={Jin, Yang and Xu, Kun and Xu, Kun and Chen, Liwei and Liao, Chao and Tan, Jianchao and Mu, Yadong and others},
  journal={arXiv preprint arXiv:2309.04669},
  year={2023}
}
