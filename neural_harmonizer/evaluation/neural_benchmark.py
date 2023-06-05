from metrics import brain_score, salieny_score,spearmanr_sim
import numpy as np



def resizing_activity(activity_array,grid_size=17):
    """
    Resize the activity array to the grid size
    
    Parameters  
    ----------
    activity_array  :   np.ndarray
                        Array of neural activity
    grid_size       :   int
                        Size of the grid
    
    Returns     
    ------- 
    resized_activity_array  :   np.ndarray
                                Resized array of neural activity
    """
    take = activity_array.shape[-1]
    X_features = np.zeros((14,grid_size,grid_size,take))
    size = activity_array.shape[1]   
    if size <=grid_size:
        for k in range(14):
            for ac in range(activity_array.shape[-1]-3):
                X_features[k,:,:,ac:ac+3] = tf.image.resize(activity_array[k,:,:,ac:ac+3],(grid_size,grid_size))
                #activity_array=  np.resize(activity_array,(14,grid_size,grid_size,take))
    else:
        size = activity_array.shape[1]
        indexes = [int(i*size/float(grid_size)) for i in range(grid_size+1)]
        for c1 in range(len(indexes)-1):
            for c2 in range(len(indexes)-1):
                X_features[:,c1,c2,:] = activity_array[:,indexes[c1]:indexes[c1+1],indexes[c2]:indexes[c2+1],:take].mean(axis=1).mean(axis=1)
    for i in range(X_features.shape[0]):
        X_features[i] -= X_features[i].min()
        X_features[i] /= X_features[i].max()

    return X_features
    
def score_features(X_features, Y,grid_size=17):
    """
    Compute the brain score and saliency score for the neural network
    
    Parameters  
    ----------
    X_features  :   np.ndarray
                    Array of neural activity
    Y           :   np.ndarray
                    Array of labels

    Returns     
    ------- 
    brain_score :   float
                    Brain score
    saliency_score   :   float
                        Saliency score
    """
    scores =[]
    dice_act_4 = []
    dice_act_1 =[]

    for i in range(14):
        out = i
        rest = [j for j in range(14) if j!=i]
        h,w = X_features.shape[1], X_features.shape[2]
        X_train = X_features[rest].reshape(13*grid_size**2,-1)
        X_test = X_features[out].reshape(1*grid_size**2,-1)
        Y_train = Y[rest].reshape(13*grid_size**2,-1)
        Y_test = Y[out].reshape(1*grid_size**2,-1)
        score,pls_kernel,y_pred= brain_score( X_train, Y_train, X_test, Y_test)
        gt = Y[out].mean(axis=2).reshape(1,grid_size,grid_size)
        activity_saliency = y_pred.numpy().reshape(grid_size,grid_size,30).mean(axis=2)
        dice_activity = spearmanr_sim(activity_saliency.reshape(1,grid_size,grid_size),gt)

        scores.append(score)
        dice_act_4.append(dice_activity[4][0])
        dice_act_1.append(dice_activity[1][0])

    final_score = [np.array(scores).mean(),np.array(scores).std()]
    final_dice_act_4 = [np.array(dice_act_4).mean(),np.array(dice_act_4).std()]
    final_dice_act_1 = [np.array(dice_act_1).mean(),np.array(dice_act_1).std()]
    return final_score, final_dice_act_4, final_dice_act_1

def score_model(activation_model,X,Y,grid_size=17):
    """
    Compute the brain score and saliency score for the neural network
    
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

    Returns     
    ------- 
    brain_score :   float
                    Brain score
    saliency_score   :   float
                        Saliency score
    """

    activity_array = activation_model.predict(X)

    assert len(activity_array)>3, "The activation model should have at least 3 layers"
    if activity_array.shape[-1]>600:
        a_shape = activity_array.flatten().shape
        if a_shape%3==0:
            activity_array = np.average(activity_array.reshape(-1,3),axis=1).reshape(a_shape[0],a_shape[1],a_shape[2],-1)
        elif a_shape%4==0:
            activity_array = np.average(activity_array.reshape(-1,4),axis=1).reshape(a_shape[0],a_shape[1],a_shape[2],-1)
        else: 
            raise ValueError("Cannot resize incompatible shapes")  
    X_features = resizing_activity(activity_array,grid_size=grid_size)
    final_score, final_dice_act_4, final_dice_act_1 = score_features(X_features, Y,grid_size=grid_size)
    
    return final_score, final_dice_act_4, final_dice_act_1