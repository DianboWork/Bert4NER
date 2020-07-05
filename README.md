In this work,  I used the [BERT](https://arxiv.org/abs/1810.04805) as the encoder to solve the English Named Entity Recognition (NER). 

What's the difference between this work and previous work? For BERT, it uses wordpiece tokenization, which means one word may break into several pieces (sub-word units). Then for NER, how to find the corresponding class label for these sub-word units is a problem. There are some discussions, such as  https://github.com/google-research/bert/issues/560 https://github.com/huggingface/transformers/issues/323 and https://github.com/huggingface/transformers/issues/64.  In this work, the average pooling over sub-word units is used to get the word representation, which is a more natural way and avoids the introduction of  the new tag ("X").
