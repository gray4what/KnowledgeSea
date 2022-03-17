##################
DL
##################

Pytorch and Tensorflow experience

*********
ML Models
*********
1. LR
2. SVM
3. Decision Tree
4. Random forest
5. Bagging and boosting
6. Gradient boost
7. XGBoost

**********
DNN Models
**********
1. CNN
2. RNN
3. Self-attention

   * positional encoding: each position has a unique positional vector e^i (hand-crafted)
   * vs. CNN: self-attention **receptive field** size is **learnable** compare to CNN whose size is fixed.
   * vs. RNN: RNN **non-parallel**. Bi-RNN is similar, but Uni-direction only learn previous states.
   * Speech application:
      #. speech vector sequence is very long, use **trucked** self-attention?

.. image:: ./_images/self_attn_1.png
  :width: 400
  :align: center

.. image:: ./_images/self_attn_2.png
  :width: 400
  :align: center

.. image:: ./_images/self_attn_3.png
  :width: 400
  :align: center

.. image:: ./_images/self_attn_4.png
  :width: 400
  :align: center

.. image:: ./_images/self_attn_5.png
  :width: 400
  :align: center

4. Mult-head self-attention

.. image:: ./_images/mult_head.png
  :width: 400
  :align: center

5. Transformer 

  .. image:: ./_images/seq2seq-encoder.png 
    :width: 400
    :align: center
    :alt: Transformer encoder 

  #. encoder: self-attention, residual, layer normal, positional encoding.
  #. decoder: plus cross-attention, Uni-direction
  
  .. image:: ./_images/seq2seq_transformer.png 
    :width: 400
    :align: center
    :alt: Transformer

  #. **self-attention** + **positional encoding** + **cross-attention**

    .. image:: ./_images/seq2seq_cross_attn.png
      :width: 400
      :align: center
      :alt: cross-attention 
  
6. BERT
    #. use encoder of Transformer: bi-direction
    #. random mask token, guess the masked **token**. 
7. GPT-2, GPT-3
    #. use decoder of Transformer: Uni-direction
    #. predict next **sentence**  
8. Wav2vec
    #. self-supervised. The objective is a contrastive loss that requires distinguishing a true future audio sample from negatives.
    #. *Solution*: Lower the dimensionality of the speech sample through an “encoder network”, and then use a context network to predict the next values.
    #. encoder net  
        *  5 conv layer.  30ms shift 10ms
    #. context net 
        * 9 conv layer.  receptive field = 210ms/frame 
9. Conformer
    #. Add convolutional layer for local feature.
        * SpecAug -> conv subsampling -> linear -> Dropout -> Conformer Block 
        * feed forward -> Mult-head attention -> conv module -> feed forward  -> Layer Norm 
        * residual in every module 

Speech Models
=============

1. Hybrid Models
2. END-to-END Models
3. CTC
    #. independent output. 
        * Pros: streaming, beam search.
        * Cons: no contextual info. training with all possible combinations.(high computation)
4. CTC + WFST: Decoding method.  Add lattice. Use LM and Lexicon to constrain CTC output.
5. RNN-T
    #. Add another RNN layer onto CTC output, sent the hidden state to next node. The hidden state only relies on training context, which equal to a LM. 
6. Neural Transducer
    #. Selected a window of feature vector for input. applied Attention.
7. MoChA
    #. Dynamic window size upon on Neural Transducer.   Add a yes/no parameter to decide if need to stop expand window. 


**********
Training
**********

Gradient vanishing and exploding
=================================

1. Avoid vanishing: Initializing, active function(leaky relu), batch normalization , early stop
2. Avoid exploding: Gradient clipping, pooling 
3. Avoid over-fitting: Regularization ,dropout, early stopping.
4. Layer Norm: normalize single feature vector for mean and std.(along Hidden size) Vertical
5. Batch Norm: normalize all training samples in a batch. (along Batch dimension) Horizontal  

FineTuning
==========


1. Transfer learning vs FineTuning
2. Steps:

  1) Remove last linear layer
  2) Froze previous layers and train the initial parameters with few epoches
  3) Unfreeze all layers, keep Training
