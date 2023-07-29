"""
Module related to the Neural  Alignment metrics
"""

import numpy as np
import tensorflow as tf
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression


EPSILON = 1e-9


def spearman_correlation(heatmaps_a, heatmaps_b):
    """
    Computes the Spearman correlation between two sets of heatmaps.

    Parameters
    ----------
    heatmaps_a
        First set of heatmaps.
        Expected shape (N, W, H).
    heatmaps_b
        Second set of heatmaps.
        Expected shape (N, W, H).

    Returns
    -------
    spearman_correlations
        Array of Spearman correlation score between the two sets of heatmaps.
    """
    assert heatmaps_a.shape == heatmaps_b.shape, "The two sets of heatmaps must" \
                                                 "have the same shape."
    assert len(heatmaps_a.shape) == 3, "The two sets of heatmaps must have shape (N, W, H)."

    scores = []

    heatmaps_a = tf.cast(heatmaps_a, tf.float32).numpy()
    heatmaps_b = tf.cast(heatmaps_b, tf.float32).numpy()

    for ha, hb in zip(heatmaps_a, heatmaps_b):
        rho, _ = spearmanr(ha.flatten(), hb.flatten())
        scores.append(rho)

    return np.array(scores)


def intersection_over_union(heatmaps_a, heatmaps_b, percentile = 10):
    """
    Computes the Intersection over Union (IoU) between two sets of heatmaps.
    Use a percentile threshold to binarize the heatmaps.

    Parameters
    ----------
    heatmaps_a
        First set of heatmaps.
        Expected shape (N, W, H).
    heatmaps_b
        Second set of heatmaps.
        Expected shape (N, W, H).

    Returns
    -------
    ious_scores
        Array of IoU scores between the two sets of heatmaps.
    """
    assert heatmaps_a.shape == heatmaps_b.shape, "The two sets of heatmaps must" \
                                                 "have the same shape."
    assert len(heatmaps_a.shape) == 3, "The two sets of heatmaps must have shape (N, W, H)."

    scores = []

    heatmaps_a = tf.cast(heatmaps_a, tf.float32).numpy()
    heatmaps_b = tf.cast(heatmaps_b, tf.float32).numpy()

    for ha, hb in zip(heatmaps_a, heatmaps_b):
        ha = (ha > np.percentile(ha, 100-percentile, (0, 1))).astype(np.float32)
        hb = (hb > np.percentile(hb, 100-percentile, (0, 1))).astype(np.float32)

        iou_inter = np.sum(np.logical_and(ha, hb))
        iou_union = np.sum(np.logical_or(ha, hb))

        iou_score = iou_inter / (iou_union + EPSILON)

        scores.append(iou_score)

    return np.array(scores)


def dice(heatmaps_a, heatmaps_b, percentile = 10):
    """
    Computes the Sorensen-Dice score between two sets of heatmaps.
    Use a percentile threshold to binarize the heatmaps.

    Parameters
    ----------
    heatmaps_a
        First set of heatmaps.
        Expected shape (N, W, H).
    heatmaps_b
        Second set of heatmaps.
        Expected shape (N, W, H).

    Returns
    -------
    dice_scores
        Array of dice scores between the two sets of heatmaps.
    """
    scores = []

    heatmaps_a = tf.cast(heatmaps_a, tf.float32).numpy()
    heatmaps_b = tf.cast(heatmaps_b, tf.float32).numpy()

    for ha, hb in zip(heatmaps_a, heatmaps_b):
        ha = (ha > np.percentile(ha, 100-percentile, (0, 1))).astype(np.float32)
        hb = (hb > np.percentile(hb, 100-percentile, (0, 1))).astype(np.float32)

        dice_score = 2.0 * np.sum(ha * hb) / (np.sum(ha + hb) + EPSILON)

        scores.append(dice_score)

    return np.array(scores)


def tf_pearson_loss(x, y, axis=-1):
    """
    Compute the Pearson's correlation as the centered cosine similarity

    Parameters
    ----------
    x   :   tf.Tensor
            First tensor
    y   :   tf.Tensor   
            Second tensor
    axis:   int 
            Axis along which to compute the correlation
    
    Returns     
    ------- 
    correlations:   tf.Tensor
                    Pearson's correlation between x and y

    """
    x_means = tf.reduce_mean(x, axis, keepdims=True)
    y_means = tf.reduce_mean(y, axis, keepdims=True)

    x_centered = x - x_means
    y_centered = y - y_means

    inner_products = tf.reduce_sum(x_centered * y_centered, axis)
    norms = tf.sqrt(tf.reduce_sum(x_centered**2, axis)) * \
            tf.sqrt(tf.reduce_sum(y_centered**2, axis))

    correlations = inner_products / norms
    
    return correlations

def brain_score( X_train, Y_train, X_test, Y_test, n_components=25,reducer='median',correlation_fn='pearson'):
    """
    Compute the brain score as the Pearson's correlation between the predicted 

    Parameters  
    ----------
    X_train :   tf.Tensor
                Training data
    Y_train :   tf.Tensor
                Training labels
    X_test  :   tf.Tensor   
                Test data  
    Y_test  :   tf.Tensor
                Test labels
    n_components    :   int
                        Number of components for the PLS regression
    reduction   :   str
                    Reduction method to aggregate the predicted scores. Options are 'mean' and 'median'. Default is 'median'
                    
    correlation :   str 
                    Correlation method to compute the brain score. Options are 'pearson' and 'spearman'. Default is 'pearson'


    Returns     
    ------- 
    score   :   float
                Brain score
    pls_kernel  :   tf.Tensor               
                    PLS kernel  
    Y_pred  :   tf.Tensor   
                Predicted labels

    """

    pls_reg = PLSRegression(25, scale=False, tol=1e-4)
    pls_reg.fit(X_train, Y_train)

    pls_kernel = tf.cast(pls_reg.coef_, tf.float32)
    Y_test = tf.cast(Y_test, tf.float32)
    X_test = tf.cast(X_test,tf.float32)
    Y_pred = tf.matmul(X_test, pls_kernel)
    if correlation_fn == 'pearson':
        correlations = tf_pearson_loss(
            tf.transpose(Y_pred),
            tf.transpose(Y_test)
        )
    elif correlation_fn == 'spearman': 
        correlations = spearmanr_sim(y_pred,y_test,reducers=[1])
    else:
        raise ValueError('Unknown correlation metric')
    if reducer == 'median':
        score = np.median(correlations)
    else:   
        score = np.mean(correlations)
    

    return score, pls_kernel,Y_pred

def saliency_score(y_test,x_test,y_pred,activation_model,pls_kernel):
    with tf.GradientTape(persistent=True) as tape:
      tape.watch(x_test)
      A_test = activation_model(x_test)
      A_test = tf.cast(A_test,tf.float32)
      y_pred = tf.matmul(A_test.reshape(17*17,-1) , pls_kernel)
      loss = tf.abs(y_pred - y_test)
   
    saliency = tape.gradient(loss, x_test).numpy()
    return saliency

def spearmanr_sim(explanations1, explanations2, reducers = [1, 4, 16],visualize=False):
  sims = {k: [] for k in reducers}

  if explanations1.shape[-1] != 1:
    explanations1 = explanations1[:,:,:,None]
  if explanations2.shape[-1] != 1:
    explanations2 = explanations2[:,:,:,None]

  explanations1 = tf.cast(explanations1, tf.float32).numpy()
  explanations2 = tf.cast(explanations2, tf.float32).numpy()

  for reducer in reducers:
    sz = int(explanations1.shape[1] / reducer)
    explanations1_resize = tf.image.resize(explanations1, (sz, sz)).numpy()
    explanations2_resize = tf.image.resize(explanations2, (sz, sz)).numpy()
    
    if visualize:
      plt.subplot(1, 2, 1)
      show(explanations1_resize[0])
      plt.subplot(1, 2, 2)
      show(explanations2_resize[0])
      plt.show()
    
    for x1, x2 in zip(explanations1_resize, explanations2_resize):
        rho, _ = spearmanr(x1.flatten(), x2.flatten())
        sims[reducer].append(rho)

  return sims


def highest_off_diagonal(correlation_matrix):
    if correlation_matrix.shape[0] != correlation_matrix.shape[1]:
        raise ValueError("Matrix is not square!")
    
    n = correlation_matrix.shape[0]
    
    # Initialize with negative infinity to ensure we find the maximum value outside diagonal
    max_value = -float('inf')
    
    for i in range(n):
        for j in range(n):
            if i != j and correlation_matrix[i,j] > max_value:
                max_value = correlation_matrix[i,j]
                
    return max_value

def ceiling_score(neural_array, time_a=100,time_b=150, correlation_fn ='pearson'):
    """
    Compute the ceiling score as the Spearman's correlation between the predicted
    and the ground truth labels

    Parameters
    ----------

    neural_array    :   np.array
                        Array of neural responses               
    time_a          :   int 
                        Start time                      
    time_b          :   int                                                                                                                 
                        End time   
    
            

    Returns 
    ------- 

    score           :   float
                        Ceiling score

                    
                        
    """
    N = neural_array.shape[0]
    R = neural_array.shape[-1]
    correlations = np.zeros((N,R,R))
    # Compute the minimum and maximum activity across all neurons and images
    min_activity = neural_array[:,:,:,time_a:time_b].mean(axis=3).min()
    max_activity = neural_array[:,:,:,time_a:time_b].mean(axis=3).max()
    for n in range(N):
        for i in range(R-1):
            # Take an anchor neurontake the average activity over the time window and flip the array to match the orientation of the images
            # Then Normalize 
            anchor_post = np.flipud(neural_array[n,:,:,time_a:time_b,i].mean(axis=2))-min_activity/(max_activity-min_activity)
            for j in range(R-1):
                #To avoid computing the same correlation twice
                if i>=j:
                    continue
                # Take a tarjet neuron, take the average activity over the time window and flip the array to match the orientation of the images
                neural_post = np.flipud(neural_array[n,:,:,time_a:time_b,j].mean(axis=2))-min_activity/(max_activity-min_activity)
                
                # make sure to use the same correlation function as the one used to compute the brain score. 
                if correlation_fn  == 'spearman':
                    score = spearmanr_sim(anchor_post, neural_post, reducers=[1, 4])[1]
                elif correlation_fn == 'pearson':
                    score = tf_pearson_loss(anchor_post,neural_post)
                else:
                    print('Please select a valid correlation function')
                    return
                # Since there is a grid of locations, compute the mean across all the locations
                correlations[n,i,j]=  np.mean(score)
            
    ceiling = []
    # Computing the ceiling score as the maximum correlation between the anchor neuron and the other neurons
    # then taking the mean across all the images.
    for n in range(N):
      dist = highest_off_diagonal(correlations[n,:])
      ceiling.append(dist)
    return np.mean(ceiling)