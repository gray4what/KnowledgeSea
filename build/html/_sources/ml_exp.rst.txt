##################
DL
##################

Pytorch and Tensorflow experience

*********
ML Models
*********

LR  
=====

*1.multi-variant*

.. image:: ./_images/ml_lr_0.png
  :width: 400
  :align: center

.. image:: ./_images/ml_lr.png
  :width: 400
  :align: center

.. image:: ./_images/ml_lr_1.png
  :width: 400
  :align: center

.. image:: ./_images/ml_lr_2.png
  :width: 400
  :align: center

.. image:: ./_images/ml_lr_3.png
  :width: 400
  :align: center

Decision Tree
=============
   
   #. able to classifiction or regression 
   #. select node.  ranking gain by entropy. Classifiction and Regression Tree(CART: gini).  
   #. Classifiction cost function (CART)

   .. math:: 
      
    J(k, t_k) = \frac{m_{left}}{m} G_{left} + \frac{m_{right}}{m} G_{right}

  where: :math:`k` is a single feature, :math:`t_k` is a threshold, :math:`G_{left/right}` is impurity,   :math:`m_{left/right}` is the number of instances
  
   #. Regression cost function (CART)

   .. math::
     J(k, t_k) = \frac{m_{left}}{m} MSE_{left} + \frac{m_{right}}{m} MSE_{right}

  where:

  .. math::

    MSE_{node} =\sum_{i\in node}^{}(\hat y_{node} - y^{(i)})^2

    y_{node} = \frac{1}{m_{node}} \sum_{i \in node}^{}(y^{i})

  #. Pruning. (avoid Overfitting)
  Prior pruning: early stop creating node 
  Post pruning: new COST = cost + n * |num of leaf|,  min new COST. 

Random forest
=============
Random ->  random samples + random features(no replacement)

Bootstraping: sampling is performed with replacement. 

Bagging: select n% of samples with replacement to create classifiers


Bayes 
=========

.. math:: 
  P(A | B) = \frac{P(B | A) P(A)}{P(B)}

#. Max likelihood estimate (MLE):  Given data the MLE for the parameter p is the value of p that maximizes the likelihood P(data |p). 
That is, the MLE is the value of p for which the data is most likely.

#. Ockham's Razor: highest p(h) is most likely. 

#. Naive Bayes

ex: D is an email with N words, h+ for trash, h- for none-trash. 

.. math::
  P(h+|D) = P(h+) * P(D|h+) / P(D)

P(h+) is prior probability, P(D|h+1) is condition probability. 

.. math::
  P(D|h+) = P(d1, d2,..., dn | h+) -> P(d1|h+) * P(d2|d1, h+) * P(d3| d2, d1, h+)...

*Navie Bayes: d1, d2 ... dn they are independent*. Hence:

.. math::
  P(D|h+) = P(d1, d2,..., dn | h+) = P(d1|h+) * P(d2|d1, h+) * P(d3| d2, d1, h+)... \\
          = P(d1|h+) * P(d2|h+) * P(d3|h+) * ...

**TODO: Bayes spelling check implementation 

XGBoost
========
*Decision tree* + *object function constrain* + *Tylor series( 1st and 2nd derivation), saddle point*. 

Trick: iterate leafs instead of samples. 

#. object function constrain: assume new added model(function) will reduce the loss

.. image:: ./_images/xgboost-1.png
  :width: 400
  :align: center

#. Tylor series for objective function 

.. image:: ./_images/xgboost-2.png
  :width: 400
  :align: center

#. iterate on leafs instead of samples 

.. image:: ./_images/xgboost-3.png
  :width: 400
  :align: center

#. 1st derivation -> minimum

.. image:: ./_images/xgboost-4.png
  :width: 400
  :align: center

#. looking for minimum object score 

.. image:: ./_images/xgboost-5.png
  :width: 400
  :align: center

#. Overall: use `Gain` to evaluate 

.. image:: ./_images/xgboost-6.png
  :width: 400
  :align: center

#. Popular parameters:
 - learning rate
 - tree 
  - max_depth
  - min_child_weight
  - subsample: select sample 
  - colsample_bytree: select feature
  - gamma: more leafs more penalty 
 - Regularization
  - lambda
  - alpha 

Adaboost
========
Adaptive boosting: boost weak classifier weight. classifier can be Decision tree, KNN, etc. 

#. Initializing sample weight, 1/N
#. train weak classifer: if the sample classifed correctly, its weight reduced in next training sample. Or, the weight increased. 
   -> iterate the training sample(update weight) + classifier
#. Combine all weak classifier as an strong classifier:  high accuracy classifier has higher weight.   


SVM
===

.. image:: ./_images/svm-kernel.png
  :width: 400
  :align: center

KNN
====

.. code-block:: python 

    def get_neighbors(train, test_row, num_neighbors):
      distances = list()
      for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
      distances.sort(key=lambda tup: tup[1])
      neighbors = list()
      for i in range(num_neighbors):
        neighbors.append(distances[i][0])
      return neighbors
    
    # calculate the Euclidean distance between two vectors
    def euclidean_distance(row1, row2):
      distance = 0.0
      for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
      return sqrt(distance)

    # Make a classification prediction with neighbors
    def predict_classification(train, test_row, num_neighbors):
      neighbors = get_neighbors(train, test_row, num_neighbors)
      output_values = [row[-1] for row in neighbors]
      prediction = max(set(output_values), key=output_values.count)
      return prediction


1. Bagging and boosting
2. Gradient boost

************
Optimizition 
************
* Gradient descent
  
  *step 1*. (Randomly) Pick an initial values :math:` w^0, b^0 `

  *step 2*. Compute :math:`\frac{\partial L}{ \partial w} |_{w=w^0}`

  .. math:: 
      
    w^1 \gets w^0 - \eta\frac{\partial L}{ \partial w} |_{w=w^0, b=b^0}
    b^1 \gets b^0 - \eta\frac{\partial L}{ \partial b} |_{w=w^0, b=b^0}

  *step 3*. Update :math:` w and b ` iteratively.


***************
ML improvement  
***************

.. image:: ./_images/ml_impr.png
  :width: 400
  :align: center

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


Overfitting
============
1. Less parameters, sharing parameters(CNN?)
2. Less feature
3. Early stopping
4. Regularization
5. Dropout

Optimizing Fails 
================
1. critical point -> local minima vs saddle point 

  .. image:: ./_images/hassian.png
    :width: 400
    :align: center

  * If Hassian matrix :math: `H > 0` (all eigen values > 0 )-> Local minima
  * If Hassian matrix :math: `H < 0` (all eigen values < 0 )-> Local maxima 
  * If eigen values of :math: `H < 0 or > 0` -> Saddle point

*To escape the saddle point and decrease the loss, Update the parameter along the directions of negative eigen vector.*

2. Batch size:

  * Smaller batch size has better performance.  the small is relative to the GPU. with parallel, small and large batch size have same speed for one update.
  * Smaller batch better for Optimizing and Generalization. 

3. Momentum
  
  Momentum + Gradient descent: Movement not just based on gradient, but previous movement.

  .. image:: ./_images/momentum.png
    :width: 400
    :align: center

4. Training stuck: loss didn't decrease doesn't mean around a critical point.  need calculate the norm of gradient to check critical point. 
5. Learning rate scheduler:
  * learning rate decay.  keep decreasing. 
  * warm up.   increase than decrease.  (ResNet, Transformer)

Summary:
  .. image:: ./_images/optims.png
    :width: 400
    :align: center

Root Mean Square(RMSProp): by considering the magnitude of gradient.

Momentum: by considering the direction of gradient

Adam = RMSProp + Momentum.