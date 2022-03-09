####################
Speaker Verification 
####################


Models
===========

#. ECAPA-TDNN: pre-trained model from speechbrain:  https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
    * 1-Dimensional Squeeze-Excitation Res2Block
        a. Squeeze: H*W*C  -> 1 * 1 * C (by global average pooling). more receptive field.
        b. Excitation: after 1 * 1 * C, add a FC layer for different channel
        c. Res2Block: input + output of SE-block
    * Multi-layer feature aggregation and summation
    * Channel- and context-dependent statistics pooling
        a. global context: mean, std
            attn = troch.cat([x, mean, std], dim=1)

Evaluation Metrics
==================

#. Equal Error Rate: This is the rate used to determine the threshold value for a system when its false acceptance rate (FAR) and false rejection rate (FRR) are equal. 

    * false acceptance rate and false rejection rate are equal

#. Minimum Detection Cost:  Compared to equal error-rate, which assigns equal weight to false negatives and false positives, this error-rate is usually used to assess performance in settings where achieving a low false positive rate is more important than achieving a low false negative rate. 

    * The DCF is defined as a weighted sum of the probabilities of type I and type II errors at a given threshold \
    * To avoid ambiguity, we mention here that we will use the following parameters: C_Miss = 1, C_FalseAlarm = 1, and P_Target = 0.05 \
    *  We follow the procedure outlined in Sec 3.1 of the NIST 2018 Speaker Recognition Evaluation Plan, for the AfV trials

Loss function
=============
#. AAM-softmax
    * Angular Softmax: cos(margin, i)
    * AM-softmax: cos(theta) - m.  
    * TODO: add figure 
