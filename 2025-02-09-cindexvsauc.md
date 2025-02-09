---
layout: default
title: "On the equivalence of C-index and AUROC"
---

# On the equivalence of C-index and AUROC
**By Marek Oerlemans**  
_February 2025_

## Introduction

Recently, I (re-)discovered the relationship between the C-index and the area under the receiver operator characteristic curve (AUC or AUROC for short). I've been working with binary classification data for quite some time, and calculating the AUC to measure model performance is quite standard. Not so long ago I started reading some papers doing survival analysis and I found the C-index. I had heard of it and its definition. The plain language version of the definition is that it's the probability that the model score for a positive sample is greater than that of a negative sample (in a binary setting). This seems rather disconnected from the idea of an area under a curve made up of the true positive rate (TPR) on the y-axis and the false positive rate (FPR) on the x-axis.

Although this fact has been known for a very long time—there are published articles ([here](https://pubs.rsna.org/doi/10.1148/radiology.143.1.7063747)) and many Q&A or blog posts (e.g. [here](https://stats.stackexchange.com/questions/437490/what-is-the-relationship-between-the-harrels-c-and-the-auc), [here](https://stats.stackexchange.com/questions/272314/how-does-auc-of-roc-equal-concordance-probability), [here](https://stats.stackexchange.com/questions/190216/why-is-roc-auc-equivalent-to-the-probability-that-two-randomly-selected-samples), and [here](https://stats.stackexchange.com/questions/190216/why-is-roc-auc-equivalent-to-the-probability-that-two-randomly-selected-samples)) on the subject—I still had to wrap my head around how to work it out mathematically. In this post, I'll try to explain and follow that thought process, and hopefully you'll learn something from it. Along the way I'll also explain various concepts of mathematics and statistics that I use.

## Definitions

To start proving that the two quantities are equivalent, we first have to define them mathematically. Assume that we have some model dataset with $N$ samples: $\{(x_i, y_i)\}_{i=1}^N$. We have built a prediction model $f$ based on this set and define its output as $\hat{y}_i = f(x_i)$. The output of the model could be in $[0,1]$ (typical for a binary classifier), but we’ll keep it general and assume $\hat{y}_i \in (-\infty, \infty)$.

### C-index
As we discussed, the concordance index (C-index) is informally defined as the probability that the score of a positive sample is higher than the score of a negative sample. In mathematical notation:

$$
\text{C-index} = \mathbb{P}(\hat{y}_i > \hat{y}_j \mid y_i > y_j) 
= \mathbb{P}(\hat{y}_i > \hat{y}_j \mid y_i = 1, y_j=0).
$$

### AUROC
To define the AUROC we first recall:

- **True Positive Rate (TPR)**:

  
  $\text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \mathbb{P}(\hat{y}_i > t \,\mid\, y_i = 1).$

- **False Positive Rate (FPR)**:

  $\text{FPR} = \frac{\text{FP}}{\text{TN} + \text{FP}} = \mathbb{P}(\hat{y}_i > t \,\mid\, y_i = 0).$

Both depend on the chosen threshold \(t\). The ROC curve is a plot of (FPR, TPR) pairs at different thresholds \(t\), as seen in the figure below.

![An ROC curve. The orange line denotes the output of a classification model; the blue line is the line a random chance model would get. [Source](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/).](rocplot.png)

The area under the ROC curve is abbreviated as AUC or AUROC, taking values in $[0.5,1]$. A perfect classifier achieves 1.0, random guessing yields 0.5. Formally:

$$
\text{AUC} = \int_0^1 \text{TPR}(\text{FPR}) d(\text{FPR}).
$$

## Equivalence between the C-index and the AUROC

Now to show their equivalence. Starting from:

$$
\text{AUC} = \int_0^1 \text{TPR}\bigl(\text{FPR}\bigr) d\text{FPR}.
$$

We do a change of variable so that $u = \text{FPR}(t)$. In more detail, we note:

> **Derivative of the FPR**  
> $$
> \begin{aligned}
> \frac{d}{dt}\text{FPR}(t) &= \frac{d}{dt}\mathbb{P}(\hat{y} > t \mid y=0) \\
> &= p(t \mid y=0),
> \end{aligned}
> $$  
> where $p(t \mid y=0)$ is the probability density function (pdf) of the model score for negative samples.


Hence, we can rewrite:

$$
\text{AUC} 
= \int_{-\infty}^{\infty} \text{TPR}(t) p(t \mid y=0) dt.
$$

Recalling $\text{TPR}(t) = \mathbb{P}(\hat{y}>t \mid y=1)$, we get:

$$
\text{AUC} 
= \int_{-\infty}^{\infty} \Bigl[\mathbb{P}(\hat{y} > t \mid y=1)\Bigr] p(t \mid y=0) dt 
= \int_{-\infty}^{\infty} \int_{t}^{\infty} p(s \mid y=1) ds  p(t \mid y=0) dt
= \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} \mathbb{1}(s > t) p(t \mid y=0) p(s \mid y=1) dt ds.
$$

Where we used the indicator function $1\(\cdot\)$, which equals one when the argument is satisfied and zero otherwise.

Recognizing this last integral is exactly:

$$
\mathbb{P}(\hat{y}_i > \hat{y}_j \mid y_i = 1, y_j = 0),
$$

which is the definition of the C-index. Thus:

$$
\text{AUC} = \text{C-index}.
$$

## Closing words

Although this exercise might be trivial for many, I found it useful to work out once. Hopefully you’ve learned something as well. Not only is it well known that the AUC and C-index are equivalent, but another well-known connection is to the [Mann–Whitney U test](https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test), a nonparametric test to see if values from one group tend to be greater than values from another group. We'll leave figuring this connection out as an exercise to the reader. Hint: use the definition of the c-index and write the U statistic as the sum of an indicator function over the positive and negative samples.

PS I'm still getting the hang of writing maths in markdown here on github pages. Sorry for weird formatting
