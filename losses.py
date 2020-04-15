# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 17:02:14 2020

@author: rfuchs
"""

"""
From umbertogriffo focal_loss_keras
Define our custom loss function.
"""

from keras import backend as K
import tensorflow as tf
import numpy as np

import dill


def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return binary_focal_loss_fixed


def categorical_focal_loss(gamma=2., alpha=.25):
    """
    Softmax version of focal loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        return K.sum(loss, axis=1)

    return categorical_focal_loss_fixed



def focal_loss(labels, logits, alpha, gamma):
  """Compute the focal loss between `logits` and the ground truth `labels`.
  Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
  where pt is the probability of being classified to the true class.
  pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
  Args:
    labels: A float32 tensor of size [batch, num_classes].
    logits: A float32 tensor of size [batch, num_classes].
    alpha: A float32 tensor of size [batch_size]
      specifying per-example weight for balanced cross entropy.
    gamma: A float32 scalar modulating loss from hard and easy examples.
  Returns:
    focal_loss: A float32 scalar representing normalized total loss.
  """
  with tf.name_scope('focal_loss'):
    logits = tf.cast(logits, dtype=tf.float32)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels, logits=logits)

    # positive_label_mask = tf.equal(labels, 1.0)
    # probs = tf.sigmoid(logits)
    # probs_gt = tf.where(positive_label_mask, probs, 1.0 - probs)
    # # With gamma < 1, the implementation could produce NaN during back prop.
    # modulator = tf.pow(1.0 - probs_gt, gamma)

    # A numerically stable implementation of modulator.
    if gamma == 0.0:
      modulator = 1.0
    else:
      modulator = tf.exp(-gamma * labels * logits - gamma * tf.math.log1p(tf.exp(-1.0 * logits)))

    loss = modulator * cross_entropy

    weighted_loss = alpha * loss
    focal_loss = tf.reduce_sum(weighted_loss)
    # Normalize by the total number of positive samples.
    focal_loss /= tf.reduce_sum(labels)
  return focal_loss


def CB_loss(samples_per_cls, loss_type = 'focal', beta = 0.999, gamma = 2.0):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      samples_per_cls: A python list of size [no_of_classes].
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    
    def class_balanced_focal_loss(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """
        
        num_classes = y_pred.shape[1] 
        num_classes = tf.cast(num_classes, dtype=tf.float32)
        
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * num_classes
                    
        weights = tf.cast(weights, dtype=tf.float32)
        weights = tf.expand_dims(weights, 0)

        weights = tf.tile(weights, [tf.shape(y_true)[0], 1]) * y_true

        # Before
        weights = tf.reduce_sum(weights, axis=1)
        weights = tf.expand_dims(weights, 1)
        weights = tf.tile(weights, [1, num_classes])
                        
        if loss_type == 'softmax':
          loss = tf.losses.softmax_cross_entropy(
              y_true, y_pred, weights=tf.reduce_mean(weights, axis=1))
          loss = tf.reduce_mean(loss)
        
        elif loss_type == 'focal':
          loss = focal_loss(y_true, y_pred, weights, gamma)
          
        return loss
    
    return class_balanced_focal_loss


def CB_loss_old(labels, logits, samples_per_cls = [14,  3432, 1, 70, 13121, 230, 205, 1956, 4972],\
                loss_type = 'focal', beta = 0.9999, gamma = 2.0):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    
    num_classes = logits.shape[1] 
    
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * num_classes
    
    
    one_hot_labels = tf.one_hot(labels, num_classes)

    weights = tf.cast(weights, dtype=tf.float32)
    weights = tf.expand_dims(weights, 0)
    weights = tf.tile(weights, [tf.shape(one_hot_labels)[0], 1]) * one_hot_labels
    weights = tf.reduce_sum(weights, axis=1)
    weights = tf.expand_dims(weights, 1)
    weights = tf.tile(weights, [1, num_classes])
    
    if loss_type == 'softmax':
      loss = tf.losses.softmax_cross_entropy(
          one_hot_labels, logits, weights=tf.reduce_mean(weights, axis=1))
      loss = tf.reduce_mean(loss)
    
    elif loss_type == 'focal':
      loss = categorical_focal_loss(one_hot_labels, logits, weights, gamma)
      
    return loss


if __name__ == '__main__':

    # Test serialization of nested functions
    bin_inner = dill.loads(dill.dumps(binary_focal_loss(gamma=2., alpha=.25)))
    print(bin_inner)

    cat_inner = dill.loads(dill.dumps(categorical_focal_loss(gamma=2., alpha=.25)))
    print(cat_inner)
    
#model = load_model(fname, custom_objects={'loss_max': loss_max}) 