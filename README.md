# Implementing the PLSA Algorithm

#### Writing the PLSA Algorithm:

The main data structures involved in the implementation of this EM algorithm are three matrices: 
1. T (topics by words): this is the set of parameters characterizing topic content that we denoted by &theta;<sub>i</sub>'s. Each element is the probability of a particular word in a particular topic. 

2. D (documents by topics): this is the set of parameters modeling the coverage of topics in each document, which we denoted by p<sub>ij</sub>'s. Each element is the probability of a particular topic is covered in a particular document. 

3. Z (hidden variables):  For every document, we need one Z which represents the probability that each word in the document has been generated from a particular topic, so for any document, this is a "word-by-topic" matrix, encoding p(Z|w) for a particular document. Z is the matrix that we compute in the E-step (based on matrices T and D, which represent our parameters). Note that we need to compute a different Z for each document, so we need to allocate a matrix Z for every document. If we do so, the M-step is simply to use all these Z matrices together with word counts in each document to re-estimate all the parameters, i.e., updating matrices T and D based on Z. Thus at a high level, this is what's happening in the algorithm: 
    * T and D are initialized. 
    * E-step computes all Z's based on T and D. 
    * M-step uses all Z's to update T and D. 
    * We iterate until the likelihood doesn't change much when we would use T and D as our output. Note that Zs are also very useful (can you imagine some applications of Zs?).



#### Resources:
[1]	[Cheng’s note](http://sifaka.cs.uiuc.edu/czhai/pub/em-note.pdf) on the EM algorithm  
[2]	[Chase Geigle’s note](http://times.cs.uiuc.edu/course/598f16/notes/em-algorithm.pdf) on the EM algorithm, which includes a derivation of the EM algorithm (see section 4), and  
[3]	[Qiaozhu Mei’s note](http://times.cs.uiuc.edu/course/598f16/plsa-note.pdf) on the EM algorithm for PLSA, which includes a different derivation of the EM algorithm.
