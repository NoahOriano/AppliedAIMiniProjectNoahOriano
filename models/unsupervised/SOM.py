import numpy as np
import numpy.matlib
import math
import time
import matplotlib.pyplot as plt



def getEuclideanDistance(single_point,array):
    nrows, ncols, nfeatures=array.shape[0],array.shape[1], array.shape[2]
    points=array.reshape((nrows*ncols,nfeatures))
                         
    dist = (points - single_point)**2
    dist = np.sum(dist, axis=1)
    dist = np.sqrt(dist)

    dist=dist.reshape((nrows,ncols))
    return dist

def GET_SOM (dispRes, trainingData, ndim=10, nepochs=10, eta0=0.1, etadecay=0.05, sgm0=20, sgmdecay=0.05, showMode=0):
    nfeatures=trainingData.shape[1]
    ntrainingvectors=trainingData.shape[0]
    
    nrows = ndim
    ncols = ndim
    
    mu, sigma = 0, 0.1
    numpy.random.seed(int(time.time()))
    som = np.random.normal(mu, sigma, (nrows,ncols,nfeatures))

    if showMode >= 1:
        fig, ax=plt.subplots(nrows=nrows, ncols=ncols, figsize=(15,15))
        # Title the plot
        fig.suptitle('SOM features BEFORE training with gridsize ' + str(dispRes[0]), fontsize=20)
        
        for k in range(nrows):
            for l in range (ncols):
                A=som[k,l,:].reshape((dispRes[0],dispRes[1]))
                ax[k,l].imshow(A,cmap="plasma")
                ax[k,l].set_yticks([])
                ax[k,l].set_xticks([])   
    
    #Generate coordinate system
    x,y=np.meshgrid(range(ncols),range(nrows))
    
    
    for t in range (1,nepochs+1):
        #Compute the learning rate for the current epoch
        eta = eta0 * math.exp(-t*etadecay)
        
        #Compute the variance of the Gaussian (Neighbourhood) function for the ucrrent epoch
        sgm = sgm0 * math.exp(-t*sgmdecay)
        
        #Consider the width of the Gaussian function as 3 sigma
        width = math.ceil(sgm*3)

        # Because this is before the EPOCH, this statemnt is the halfway point. 
        # Ex, with two epochs, after the first epoch t = 2, 1 + (2+1)//2 = 2, so this will be the halfway point
        if t == 1 + (nepochs+1)//2:
            # Display the SOM map after the first half of the epochs
            if(showMode >= 1):
                fig, ax=plt.subplots(nrows=nrows, ncols=ncols, figsize=(15,15))
                fig.suptitle('SOM features HALFWAY through training', fontsize=20)
                for k in range(nrows):
                    for l in range (ncols):
                        A=som[k,l,:].reshape((dispRes[0],dispRes[1]))
                        ax[k,l].imshow(A,cmap="plasma")
                        ax[k,l].set_yticks([])
                    ax[k,l].set_xticks([])
        
        for ntraining in range(ntrainingvectors):
            trainingVector = trainingData[ntraining,:]
            
            # Compute the Euclidean distance between the training vector and
            # each neuron in the SOM map
            dist = getEuclideanDistance(trainingVector, som)
       
            # Find 2D coordinates of the Best Matching Unit (bmu)
            bmurow, bmucol =np.unravel_index(np.argmin(dist, axis=None), dist.shape)
            
            
            #Generate a Gaussian function centered on the location of the bmu
            g = np.exp(-((np.power(x - bmucol,2)) + (np.power(y - bmurow,2))) / (2*sgm*sgm))

            #Determine the boundary of the local neighbourhood
            fromrow = max(0,bmurow - width)
            torow   = min(bmurow + width,nrows)
            fromcol = max(0,bmucol - width)
            tocol   = min(bmucol + width,ncols)

            
            #Get the neighbouring neurons and determine the size of the neighbourhood
            neighbourNeurons = som[fromrow:torow,fromcol:tocol,:]
            sz = neighbourNeurons.shape
            
            #Transform the training vector and the Gaussian function into 
            # multi-dimensional to facilitate the computation of the neuron weights update
            T = np.matlib.repmat(trainingVector,sz[0]*sz[1],1).reshape((sz[0],sz[1],nfeatures));                   
            G = np.dstack([g[fromrow:torow,fromcol:tocol]]*nfeatures)

            # Update the weights of the neurons that are in the neighbourhood of the bmu
            neighbourNeurons = neighbourNeurons + eta * G * (T - neighbourNeurons)

            
            #Put the new weights of the BMU neighbouring neurons back to the
            #entire SOM map
            som[fromrow:torow,fromcol:tocol,:] = neighbourNeurons

    if showMode >= 1:
        fig, ax=plt.subplots(nrows=nrows, ncols=ncols, figsize=(15,15))
        fig.suptitle('SOM features AFTER training', fontsize=20)
        for k in range(nrows):
            for l in range (ncols):
                A=som[k,l,:].reshape((dispRes[0],dispRes[1]))
                ax[k,l].imshow(A,cmap="plasma")
                ax[k,l].set_yticks([])
            ax[k,l].set_xticks([])   
    return som
    
#verification of correctness on the training set:

def SOM_Test (trainingData, som_, classes, grid_, ConfusionMatrix, ndim=60):
    nfeatures=trainingData.shape[1]
    ntrainingvectors=trainingData.shape[0]
    
    nrows = ndim
    ncols = ndim
    
    nclasses=np.max(classes)

    som_cl=np.zeros((ndim,ndim,nclasses+1))
    
    
    for ntraining in range(ntrainingvectors):
        trainingVector = trainingData[ntraining,:]
        class_of_sample= classes[ntraining]    
        # Compute the Euclidean distance between the training vector and
        # each neuron in the SOM map
        dist = getEuclideanDistance(trainingVector, som_)
       
        # Find 2D coordinates of the Best Matching Unit (bmu)
        bmurow, bmucol =np.unravel_index(np.argmin(dist, axis=None), dist.shape) ;
        
        
        som_cl[bmurow, bmucol,class_of_sample]=som_cl[bmurow, bmucol,class_of_sample]+1
    
    
    
    for i in range (nrows):
        for j in range (ncols):
            grid_[i,j]=np.argmax(som_cl[i,j,:])

    for ntraining in range(ntrainingvectors):
        trainingVector = trainingData[ntraining,:]
        class_of_sample= classes[ntraining]  
        # Compute the Euclidean distance between the training vector and
        # each neuron in the SOM map
        dist = getEuclideanDistance(trainingVector, som_)
       
        # Find 2D coordinates of the Best Matching Unit (bmu)
        bmurow, bmucol =np.unravel_index(np.argmin(dist, axis=None), dist.shape) ;
        
        predicted=np.argmax(som_cl[bmurow, bmucol,:])
        if (predicted > ConfusionMatrix.shape[0]):
            print('Error: SOM_Test: predicted class is greater than the number of classes')
            predicted=ConfusionMatrix.shape[0]
        if (class_of_sample > ConfusionMatrix.shape[0]):
            print('Error: SOM_Test: class of sample is greater than the number of classes')
            class_of_sample=ConfusionMatrix.shape[0]
        ConfusionMatrix[class_of_sample-1, predicted-1]=ConfusionMatrix[class_of_sample-1, predicted-1]+1
        
    return grid_, ConfusionMatrix
    

class SOM:
    def train(X_train, y_train):
        # Limit the samples to 1500
        if len(X_train) > 1500:
            X_train = X_train[:1500]
            y_train = y_train[:1500]
        model = GET_SOM ([20,20], X_train, ndim=10, nepochs=100, eta0=1, etadecay=0.03, sgm0=20, sgmdecay=0.05, showMode=0)
        return model
    
    def predict_and_evaluate(model, X_test, y_test):
        # Convert labels to integers
        if(type(y_test) == str):
            # Find unique labels and map them to integers
            unique_labels = np.unique(y_test)
            label_map = {label: i for i, label in enumerate(unique_labels)}
            y_test = np.array([label_map[label] for label in y_test])
        else:
            y_test = np.array([int(label) for label in y_test])
        num_classes = len(np.unique(y_test))
        grid_= np.zeros((20,20))
        ConfusionMatrix= np.zeros((num_classes,num_classes))
        model = SOM_Test(X_test, model, y_test, grid_, ConfusionMatrix, ndim=10)
        return ConfusionMatrix

        