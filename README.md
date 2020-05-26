## Summarization论文汇总

### 预训练模型及其变种

(Transformer) Attention Is All You Need **[[pdf]](https://arxiv.org/abs/1706.03762)**

(BERT) BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding **[(pdf)](https://arxiv.org/abs/1810.04805)**

(T5) Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer **[[pdf]](https://arxiv.org/abs/1910.10683)** **[[code]](https://github.com/google-research/text-to-text-transfer-transformer)**

(Unilm) [NIPS 2019] Unified Language Model Pre-training for Natural Language Understanding and Generation **[[pdf]](https://arxiv.org/abs/1905.03197)**

(BART) BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension **[[pdf]](https://arxiv.org/abs/1910.13461)**

(MASS) [ICML 2019] MASS: Masked Sequence to Sequence Pre-training for Language Generation **[[pdf]](https://arxiv.org/abs/1905.02450)**

(ProphetNet) ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training **[[pdf]](https://arxiv.org/abs/2001.04063)**

(GPT) Improving Language Understanding by Generative Pre-Training **[[pdf]](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)**

(GPT 2) Language Models are Unsupervised Multitask Learners **[[pdf]](https://www.ceid.upatras.gr/webpages/faculty/zaro/teaching/alg-ds/PRESENTATIONS/PAPERS/2019-Radford-et-al_Language-Models-Are-Unsupervised-Multitask-%20Learners.pdf)**

(XLnet) XLNet: Generalized Autoregressive Pretraining for Language Understanding **[[pdf]](https://arxiv.org/abs/1906.08237)**

Fine-tune BERT for Extractive Summarization **[[pdf]](https://arxiv.org/abs/1903.10318)** **[[code]](https://github.com/nlpyang/BertSum
)** **[[notes]](https://docs.qq.com/doc/DVVBiaUF3b2tjWVFz)**

Text Summarization with Pretrained Encoders **[[pdf]](https://arxiv.org/abs/1908.08345)** **[[notes]](https://docs.qq.com/doc/DWG5xZ3pVV3lTTkVP)**

[CoNLL 2019] Pretraining-Based Natural Language Generation for Text Summarization **[[pdf]](https://arxiv.org/abs/1902.09243)** (文章总体为编解码器结构，编码端使用了BERT表示，解码端首先得到一个初步的摘要结果，再应用MASK规则预测每个位置的refine word（优化词），相当于一个重写过程)

[EMNLP 2016] Language as a Latent Variable: Discrete Generative Models for Sentence Compression **[[pdf]](https://arxiv.org/abs/1609.07317)** (先利用预先训练的模型辅助压缩句子，然后利用压缩后的句子还原原始句子。)

[ACL 2019] HIBERT: Document Level Pre-training of Hierarchical Bidirectional Transformers for Document **[[pdf]](https://www.aclweb.org/anthology/P19-1499/)** **[[notes]](https://docs.qq.com/doc/DWG5xZ3pVV3lTTkVP)**

### 结合外部知识

K-BERT: Enabling Language Representation with Knowledge Graph **[[pdf]](https://arxiv.org/abs/1909.07606)**

[SLT 2018] Abstractive Dialogue Summarization with Sentence-Gated Modeling Optimized by Dialogue Acts **[[pdf]](https://arxiv.org/abs/1809.05715)** **[[notes]](https://docs.qq.com/doc/DWG5xZ3pVV3lTTkVP)**

[ACL 2019] BiSET: Bi-directional Selective Encoding with Template for Abstractive Summarization **[[pdf]](https://www.aclweb.org/anthology/P19-1207/)** **[[notes]](https://docs.qq.com/doc/DWG5xZ3pVV3lTTkVP)**

[KDD 2019] Automatic Dialogue Summary Generation for Customer Service **[[pdf]](https://www.kdd.org/kdd2019/accepted-papers/view/automatic-dialogue-summary-generation-for-customer-service)** **[[notes]](https://docs.qq.com/doc/DY1VoUHROVXFlQ1BG)**

### 数据集

[ACL 2019] SAMSum Corpus: A Human-annotated Dialogue Dataset for Abstractive Summarization **[[pdf]](https://arxiv.org/abs/1911.12237)** **[[dataset]](https://arxiv.org/src/1911.12237v2/anc/corpus.7z)**

[ACL 2019] Multi-News: a Large-Scale Multi-Document Summarization Dataset and Abstractive Hierarchical Model **[[pdf]](https://arxiv.org/abs/1906.01749)** **[[code]](https://github.com/Alex-Fabbri/Multi-News)**

### PGN相关

(PGN) Get To The Point: Summarization with Pointer-Generator Networks **[[pdf]](https://arxiv.org/abs/1704.04368)**

(Transformer与PGN结合) [EMNLP 2019] Denoising based Sequence-to-Sequence Pre-training for Text Generation **[[pdf]](https://arxiv.org/abs/1908.08206)**

[EMNLP 2019] Improving Latent Alignment in Text Summarization by Generalizing the Pointer Generator **[[pdf]](https://www.aclweb.org/anthology/D19-1390/)** **[[code]](https://github.com/chin-gyou/generalized-PG)** **[[notes]](https://docs.qq.com/doc/DVVBiaUF3b2tjWVFz)**

[CIKM2018] Multi-Source Pointer Network for Product Title Summarization **[[pdf]](https://arxiv.org/abs/1808.06885)** **[[notes]](https://docs.qq.com/doc/DVVBiaUF3b2tjWVFz)**

### 图模型

[ACL 2018] Unsupervised Abstractive Meeting Summarization with Multi-Sentence Compression and Budgeted Submodular Maximization [**[code]**](https://bitbucket.org/dascim/acl2018_abssumm
) **[[pdf]](https://arxiv.org/abs/1805.05271)** [**[notes]**](https://docs.qq.com/doc/DQUR6d1BYU0Z5RFpB)

[ACL 2019] Sentence Centrality Revisited for Unsupervised Summarization **[[pdf]](https://arxiv.org/abs/1906.03508)** (该论文使用经过fine-tune的bert表示，并利用有向图模型刻画句子之间的位置关系。)

### 其他

[ACL 2019] Keep Meeting Summaries on Topic: Abstractive Multi-Modal Meeting Summarization **[[pdf]](https://www.aclweb.org/anthology/P19-1210/)** **[[notes]](https://docs.qq.com/doc/DWG5xZ3pVV3lTTkVP)**

[NAACL 2018] A Discourse-Aware Attention Model for Abstractive Summarization of Long Documents **[[pdf]](https://arxiv.org/abs/1804.05685)** **[[notes]](https://docs.qq.com/doc/DWG5xZ3pVV3lTTkVP)**

[EMNLP-IJCNLP 2019] Extractive Summarization of Long Documents by Combining Global and Local Context **[[pdf]](https://arxiv.org/abs/1909.08089)** **[[notes]](https://docs.qq.com/doc/DWG5xZ3pVV3lTTkVP)**

[ACL 2019] Scoring Sentence Singletons and Pairs for Abstractive Summarization **[[pdf]](https://arxiv.org/abs/1906.00077)** **[[notes]](https://docs.qq.com/doc/DWG5xZ3pVV3lTTkVP)**

[AAAI 2020] Two-Level Transformer and Auxiliary Coherence Modeling for Improved Text Segmentation **[[pdf]](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-GlavasG.7898.pdf)** (该文章对连贯性进行显式的建模，提高了文本分割的准确度)

[ACL 2019] Improving the Similarity Measure of Determinantal Point Processes for Extractive Multi-Document Summarization **[[pdf]](https://arxiv.org/abs/1906.00072)** (文章有两项创新点，1）使用DPP（行列式点过程）来表征摘要的质量和多样性，2）在计算句子相似度时使用胶囊网络，从更全面的角度考虑句子之间的语义相关程度)

[NAACL 2018] Guiding Generation for Abstractive Text Summarization Based on Key Information Guide Network **[[pdf]](https://www.aclweb.org/anthology/N18-2009/)** (文章提出抽取式和生成式相结合的方案，通过抽取式的方案获取关键词汇，用这些词汇指导摘要的生成)

BottleSum: Unsupervised and Self-supervised Sentence Summarization using the Information Bottleneck Principle **[[pdf]](https://arxiv.org/abs/1909.07405)** **[[notes]](https://docs.qq.com/doc/DQW9seVJnR0RoeWJ5)** (文中提出的模型包含一个无监督的句子压缩方法以及一个自监督的生成式方法，首先应用信息瓶颈原理对每个句子进行无监督的压缩，而后用压缩结果作为自监督的标签对语言模型进行微调，赋予模型生成的能力。)

[EMNLP 2018] Bottom-Up Abstractive Summarization **[[pdf]](https://arxiv.org/abs/1808.10792)** **[[notes]](https://docs.qq.com/doc/DY1VoUHROVXFlQ1BG)**

A More Abstractive Summarization Model **[[pdf]](https://arxiv.org/abs/2002.10959)** **[[notes]](https://docs.qq.com/doc/DY1VoUHROVXFlQ1BG)**

End-to-End Abstractive Summarization for Meetings **[[pdf]](https://arxiv.org/abs/2004.02016)**

[NAACL 2018] Ranking Sentences for Extractive Summarization with Reinforcement Learning **[[pdf]](https://www.aclweb.org/anthology/N18-1158/)** **[[code]](https://github.com/shashiongithub/Refresh)**

[ACL 2019] Self-Supervised Learning for Contextualized Extractive Summarization **[[pdf]](https://www.aclweb.org/anthology/P19-1214/)** **[[code]](https://github.com/hongwang600/Summarization)** **[[notes]](https://docs.qq.com/doc/DVVBiaUF3b2tjWVFz)**
