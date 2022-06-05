#######################
ASR
#######################


****************
Traditional ASR
****************

GMM + HMM

-----------------------
Gaussian Mixture Model 
-----------------------

.. math:: 

    Pr(x) = \sum_{k=1}^{K}\pi\mathcal{N}(x | \mu_k, \Sigma_k)
    
    where

    \sum_{k=1}^{K} = 1, 0 \le \pi_k \le 1 


* Loss function is the negative log likelihood:
  
.. math:: 

    -logPr(x| \pi, \mu, \Sigma) = -\sum_{i=1}^{n}log\left\{ \sum_{k=1}^{K}\pi_k \mathcal{N}(x|\mu_k, \Sigma_k)  \right\}


hard to optimize directly:
    - sum over the components appears inside the log, thus coupling all the parameters.


* solution: iterative!  by Expectation-Maximization (EM) 

    - Given the observations :math:`x_i, i = 1, 2, ..., n`
    - Each :math:`x_i` is associated with a latent variable :math:`z_i = (z_{i1}, ... z_{ik})`
    - Given the complete data :math:`(x,z) = (x_i, z_i), i = 1, 2, ..., n`
        - we can estimate the parameters by maximizing the complete data log likelihood:

.. math::
    logPr(x| \pi, \mu, \Sigma) = \sum_{i=1}^{n}\sum_{k=1}^{k}Z_{ik}\left\{log \pi_k + log\mathcal{N}(x_i|\mu_k, \Sigma_k)  \right\}  
            

The latent varibale parameter :math:`Z_{ik}` represents the contribution of k-th Gaussian to :math:`x_i`


Optimizing
-----------

* initial with :math:`\mu_0, \theta_0, I, \pi_0`
* Update equations at the k-th iteration:
    - E-step: Give parameters, compute:
  
.. math:: 
    r_{ik} \triangleq E(Z_{ik}) = \frac{\pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k))}{\sum_{k=1}^{K} \pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k))}

*************************************************************

    - M-step: Maximize the expected complete log likelihood:
  
.. math:: 
    E[logPr(x, z | \pi, \mu, \Sigma)]] = \sum_{i=1}^{n}\sum_{k=1}^{k}r_{ik}\left\{log \pi_k + log\mathcal{N}(x_i|\mu_k, \Sigma_k)  \right\}  


By updating the parameters:

.. math:: 

    \pi_{k+1} = \frac{\sum_i^{n} r_{ik} }{n}, \mu_{k+1} = \frac{\sum_i^{n} r_{ik} x_i }{r_{ik}}, \Sigma_{k+1} = \frac{\sum_i r_{ik}(x_i -u_k)(x_i -u_k)^T}{\sum_i r_{ik}}

-- iterate till likelihood converges

-- Converges to local optimum of the log likelihood

! It may not converge to the global optimum


-----------------------
Hidden Markov Model
-----------------------

.. _HMM: https://github.com/fengdu78/lihang-code/blob/master/%E7%AC%AC10%E7%AB%A0%20%E9%9A%90%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E6%A8%A1%E5%9E%8B/10.HMM.ipynb

.. _HMM_detail: chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/viewer.html?pdfurl=https%3A%2F%2Fwww.cs.ubc.ca%2F~murphyk%2FSoftware%2FHMM%2FE6820-L10-ASR-seq.pdf&clen=196738&chunk=true

HMM example: HMM_ 

HMM detail: HMM_detail_


*********************
Text to Speech(TTS)
*********************

.. _TTSbook: https://github.com/cnlinxi/book-text-to-speech/blob/main/text_to_speech.pdf

TTS book: TTSbook_