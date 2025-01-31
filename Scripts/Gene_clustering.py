import numpy as np
import os 
import re
import concurrent.futures
from multiprocessing import shared_memory
from tslearn.piecewise import OneD_SymbolicAggregateApproximation
from tslearn.piecewise import SymbolicAggregateApproximation
from Scripts.pwm_functions import process_batch
from tck import TCK
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import somoclu as som #This works only on linux systems
import pycatch22 as catch22 #This only works on linux systems
from scipy.spatial.distance import pdist, squareform
from itertools import combinations
from sklearn.gaussian_process import GaussianProcessRegressor as GP
from sklearn.gaussian_process.kernels import RBF
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score

matplotlib.use('Agg')

#Fourth attempt / Using De Bruijn graphs and phylogenetic inference

### You should remember that this procedure starts with finding the SAX representation of all binding profiles

#Converting the SAX representations to a string

def SAX_to_String(profile: np.ndarray, word_length: int = 30) -> str:

    '''
    This function accepts a SAX represented TF binding profile. The function converts each site (total number of sites: word_length) to
    a string of length profile.shape[0], in our case equal to the number of TF factors i.e. 145. It outputs a string of length 145 * word_length,
    corresponding to the concatenation of strings per site. 
    '''

    s=''
    for index in range(word_length):
        #This is the computation of a string per position of profile for all TFs
        s+=(profile[:,index]+65).astype(np.int8).tobytes().decode('ascii') #65 here for ASCII capital letters.
    return s

#These are the per site string representations of the TF binding profiles for all S. cerevisae genes
SAX_strings={f'Gene {profile}': SAX_to_String(profile=repr[profile]) for profile in range(repr.shape[0])}


#I could try to use some form of jaccard index // OR even cosine similarity(?)

def compute_Jaccard(sequence1: str, sequence2: str, k: int)-> float:
    '''
    The function accepts two sequences, sequence1 and sequence2, representing the cumulative binding profile of TFs in the upstream area of a certain gene as input. 
    The function outputs the jaccard index score based on the similarity of the two strings based on the common set of k-mers of that length.
    '''

    #Find all k-mer instances within the two sequences
    kmers1=[sequence1[index: index+k] for index in range(len(sequence1)-k+1)]
    kmers2=[sequence2[index: index+k] for index in range(len(sequence2)-k+1)]

    kmers1, kmers2 = set(kmers1), set(kmers2)

    return len(kmers1 & kmers2) / (len(kmers1) + len(kmers2) - len(kmers1 & kmers2) )


distance_Jaccard=np.zeros((repr.shape[0], repr.shape[0]))

#Generating a distance matrix based on the Jaccard distances
for i in range(repr.shape[0]):
    for j in range(i, repr.shape[0]):
        distance_Jaccard[i,j] = compute_Jaccard(sequence1=SAX_strings[f'Gene {i}'], sequence2=SAX_strings[f'Gene {j}'], k=6)

distance_Jaccard=np.load("Jaccard.npy")

sym_jaccard=distance_Jaccard + distance_Jaccard.T - np.diag(distance_Jaccard.diagonal())


#Try to align the sequences using multiple sequence alignment because, why not! First, i need to create a FASTA file

with open(os.getcwd()+'\\scripts\\SAX_strings.fa','x') as f:
    for gene, string in SAX_strings.items():
        f.write(f'>{gene}\n{string}\n')



#This is stuff related to de-bruijn graph interpretation

import math
from functools import reduce
from scipy.special import gammaln
import networkx as nx

def debruijn_adjacency(representation: str, order: int = 5) -> np.ndarray:

    '''
    This function accepts as input a SAX cumulative representation of TF binding.
    The function outputs an adjacency matrix corresponding to the De-Bruijn graph of the cyclic representation of the word.

    Adapted from: De Bruijn entropy and string similarity
    '''

    #We first find the kth-order patterns that exist within the input representation
    kmers=[]
    for index in range(len(representation)-order+1):
        kmers+=[ representation[index: index+order] ]

    #This is to make the word cyclic and make sure that we build a cyclic eulerian graph
    for i in range(order): 
        kmers+=[kmers[-1][1:]+kmers[0][i]]

    #Since the kmers are added sequentialy, taking the neighboring entries results in constructing the De-Bruijn graph.
    edges = [(kmers[i], kmers[i+1]) for i in range(len(kmers)) if i+1 < len(kmers)]

    #A utility that numbers unique patterns to indices to help with adjacency matrix construction
    node_index={kmer: num for num, kmer in enumerate(np.unique(kmers).tolist())}

    #Initialization of the adjacency matrix
    adjacency = np.zeros( (len(node_index), len(node_index)) )

    #Adding 1 for each ordered pair, i.e. each edge of the de-Bruijn graph
    for outgoing, incoming in edges:
        adjacency[node_index[outgoing], node_index[incoming]]+=1

    return adjacency

def calculate_entropy(A: np.ndarray) -> float:

    #This is the greatest common divisor of the flattened input matrix
    d=reduce(math.gcd, A.flatten().astype(int))

    #Find the divisors of d
    candidates = np.arange(1, int(np.sqrt(d)) + 1)
    divisors = candidates[d % candidates == 0]

    entropy = 0

    for divisor in divisors:
        
        #Normalize based on divisor
        Acur= A/divisor
        degrees=np.sum(Acur, axis=1)

        #This is the Laplacian of matrix Acur
        L=np.diag( degrees ) - Acur

        #I only need to take the determinant once because all diagonal cofactors are equal
        L_minor = np.delete(np.delete(L, 0, axis=0), 0, axis=1)  # Remove row i and column i
        logt=np.linalg.slogdet(L_minor)  #this calculates the number of directed spanning trees 

        #This just calculates all the numbers up to divisor that are prime (gcd: 1) with divisor
        totient = np.sum([math.gcd(divisor, i) == 1 for i in range(1, divisor + 1)])

        #This is the upper part of eqn. 2 from article
        log_upper = np.log(totient) + logt+ np.sum( gammaln(degrees) )

        #This is the lower part of eqn.2 from article
        log_lower = np.log(divisor) + np.sum( gammaln(Acur[Acur>0] + 1 ) )

        entropy += log_upper - log_lower
        
    return entropy

def debruijn_coupled_adjacency(representation1: str, representation2: str, order: int = 5) -> np.ndarray:

    '''
    This function accepts as input a SAX cumulative representation of TF binding.
    The function outputs an adjacency matrix corresponding to the De-Bruijn graph of the cyclic representation of the word.

    Adapted from: De Bruijn entropy and string similarity
    '''

    #We first find the kth-order patterns that exist within the input representation
    kmers1=[]
    for index in range(len(representation1)-order+1):
        kmers1+=[ representation1[index: index+order] ]

    kmers2=[]
    for index in range(len(representation2)-order+1):
        kmers2+=[ representation2[index: index+order] ]

    #This is to make the word cyclic and make sure that we build a cyclic eulerian graph
    for i in range(order): 
        kmers1+=[kmers1[-1][1:]+kmers1[0][i]]
        kmers2+=[kmers2[-1][1:]+kmers2[0][i]]

    common_kmers = set(kmers1) | set(kmers2)
    #Since the kmers are added sequentialy, taking the neighboring entries results in constructing the De-Bruijn graph.
    edges1 = [(kmers1[i], kmers1[i+1]) for i in range(len(kmers1)) if i+1 < len(kmers1)]
    edges2 = [(kmers2[i], kmers2[i+1]) for i in range(len(kmers2)) if i+1 < len(kmers2)]

    #A utility that numbers unique patterns to indices to help with adjacency matrix construction
    node_index={kmer: num for num, kmer in enumerate(common_kmers)}

    #Initialization of the adjacency matrix
    adjacency1 = np.zeros( (len(node_index), len(node_index)) )
    adjacency2 = np.zeros( (len(node_index), len(node_index)) )

    #Adding 1 for each ordered pair, i.e. each edge of the de-Bruijn graph
    for outgoing, incoming in edges1:
        adjacency1[node_index[outgoing], node_index[incoming]]+=1
    
    for outgoing, incoming in edges2:
        adjacency2[node_index[outgoing], node_index[incoming]]+=1

    return adjacency1,adjacency2

def calculate_relative_entropy(A1: np.ndarray, A2: np.ndarray) -> float:

    '''
    This function accepts the eulerian quivers of two strings and produces the relative entropy between the two.
    The function calculates the component-wise eulerian entropy for combined string. Could also be used to find the common
    patterns between the two strings.
    '''

    #Transformation of input adjacencies
    Af = A1 - A2 
    Af[Af<0]=0 #Clipping the values to non-negative 
    Ab = A2 - A1
    Ab[Ab<0]=0

    A = Af + Ab.T

    #This is to find all disconnected euleriean quivers from the combined adjacency
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    
    # A list containing the eulerian quivers
    sccs = list(nx.strongly_connected_components(G))
    sccs = sorted(sccs, key=lambda x: min(x))  # Sorted for consistency

    entropy = 0

    for quiver in sccs:
        
        #Finding the component-wise quiver 
        Ac= A[list(quiver),:] 

        #Degenerate case of a eulerian quiver with a single node
        if Ac.shape[0] == 1:
            #In that case, there is no entropy
            entropy+=0
            continue

        #This is to clean any columns containing zero entries // Makes matrix square
        Ac = Ac[:,np.any(Ac!=0, axis=0)]       

        #This is the greatest common divisor of the flattened input matrix
        d=reduce(math.gcd, Ac.flatten().astype(int))

        #Find the divisors of d
        candidates = np.arange(1, int(np.sqrt(d)) + 1)
        divisors = candidates[d % candidates == 0]

        for divisor in divisors:
            
            #Normalize based on divisor
            Acur= Ac/divisor
            degrees=np.sum(Acur, axis=1)

            #This is the Laplacian of matrix Acur
            L=np.diag( degrees ) - Acur

            #I only need to take the determinant once because all diagonal cofactors are equal
            L_minor = np.delete(np.delete(L, 0, axis=0), 0, axis=1)  # Remove row i and column i
            _,logt=np.linalg.slogdet(L_minor)  #this calculates the number of directed spanning trees 

            #This condition is important, since this means that there are no directed graphs with those nodes
            #Thus we skip calculations
            if np.isneginf(logt):
                continue

            #This just calculates all the numbers up to divisor that are prime (gcd: 1) with divisor
            totient = np.sum([math.gcd(divisor, i) == 1 for i in range(1, divisor + 1)])

            #This is the upper part of eqn. 2 from article
            log_upper = np.log(totient) + logt + np.sum( gammaln(degrees) )

            #This is the lower part of eqn.2 from article
            log_lower = np.log(divisor) + np.sum( gammaln(Acur[Acur>0] + 1 ) )

            entropy += log_upper - log_lower
        
    return entropy

#Based on the paper the maximaly informative order of k=4 // log_{8}(4350)

debruijn_entr=np.zeros( (100, 100) )
for i in range(100):
    for j in range(i,100):
        A1, A2 = debruijn_coupled_adjacency(SAX_strings[f'Gene {i}'], SAX_strings[f'Gene {j}'], order=4)
        debruijn_entr[i,j] = calculate_relative_entropy(A1=A1, A2=A2)


A1, A2 = debruijn_coupled_adjacency(SAX_strings[f'Gene 22'], SAX_strings[f'Gene 24'], order=4)
calculate_relative_entropy(A1=A1, A2=A2)


deb = debruijn_entr + debruijn_entr.T
 
visualize_clustermap(deb, 'average', 'deby')



#A small utility in order to more cleanly define the directory of a file 
def generate_path(folder: str, file: str)-> str:

    '''
    This function accepts as input a system folder and a file name thought to exist within said folder.
    The function returns a full path to the desired file. If the path is not valid, it returns an error.
    '''

    full_dir=os.path.join(os.path.expanduser("~"), folder, file)
    return full_dir if os.path.exists(full_dir) else f"File {file} does not exist in Folder {folder}"

#Import the generated gene IDs
with open(rf'{generate_path("Documents","Gene_Profiles.npy")}',"rb") as f: 
    result=np.load(f, allow_pickle=True)


### First clustering attempt / SAX representation ###

#Z-score normalization of the Gene Profiles
mu=np.mean(result,axis=-1, keepdims=True)
std=np.std(result, axis=-1, keepdims=True)
normalized=(result-mu)/std

#Initialization of constructors responsible for SAX and 1D-SAX encoding of the input sequences
n_sax_symbols = 8
n_segments=30
sax = SymbolicAggregateApproximation(n_segments=n_segments,
                                     alphabet_size_avg=n_sax_symbols,)


n_seg=30
one_d_sax = OneD_SymbolicAggregateApproximation(
    n_segments=n_seg,
    alphabet_size_avg=10,
    alphabet_size_slope=10,
    sigma_l=np.sqrt(0.03/(500/n_seg))) #This was based on annotation from the paper of 1D-SAX


## Transformation of the original gene profiles to their corresponding representations ##

#SAX 
repr=[]
for index in range(normalized.shape[0]):
    repr.append(sax.fit_transform(normalized[index,:,:,np.newaxis]))

#This converts the final array to the appropriate format 
repr=np.array(repr)
repr=np.squeeze(repr) 


#1D-SAX
one_d_repr=[]
for index in range(normalized.shape[0]):
    one_d_repr.append(one_d_sax.fit_transform(normalized[index,:,:,np.newaxis]))

one_d_repr=np.array(one_d_repr)
one_d_repr=np.squeeze(one_d_repr)

## Estimation of a distance matrix between SAX / 1D-SAX representations ##

#Distance Profile calculation 

def get_distance_matrix(TF: int, random_entries: np.array, repr: np.ndarray = repr, subset: int = 30, type: str = 'sax' ) -> np.ndarray:
    '''

    * This function accepts as input a particular Transcription Factor index. Besides that, it accepts the type of representation.
    * It returns an upper triangular matrix of dimensions (N_genes x N_genes) containing the pairwise similarity of TF binding profiles as
    recorded by the lower bound euclidean distance estimation provided by the encoding.

    Note: The function can be adapted to calculate a distance matrix for a subset of representations, the number of which can be tuned using the subset parameter (default: 30). So as to select the same indices per TF index, a random_entries array must also
    be provided in order to specify which indices should be used in distance matrix estimation.

    '''

    ## Commented lines correspond to parameters necessary for total distance matrix estimation ## 

    shm = shared_memory.SharedMemory(create=True, size=repr.nbytes)
    
    # Create a Numpy array using the shared memory
    shared_repr = np.ndarray(repr.shape, dtype=repr.dtype, buffer=shm.buf)
    np.copyto(shared_repr, repr)  # Copy data into shared memory

    #distances=np.zeros((repr.shape[0], repr.shape[0]))
    distances=np.zeros((subset,subset))
    batch_size=10

    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:

        futures = []

        if type=='sax':
    
            for row in range(subset): #repr.shape[0]

                batch = []

                for column in range(row, subset): #repr.shape[0]
                    
                    batch.append((random_entries[row], random_entries[column], shared_repr[:,TF,:]))

                    if len(batch) == batch_size:
                        futures.append(executor.submit(process_batch, normalized[0,:,:,np.newaxis], batch))
                        batch = []

                if batch:  # Submit any remaining pairs
                    futures.append(executor.submit(process_batch, normalized[0,:,:,np.newaxis], batch))
                
                for process in concurrent.futures.as_completed(futures):
                    
                    results=process.result()

                    for row, column, value in results:
                        distances[np.where(random_entries==row)[0][0], np.where(random_entries==column)[0][0]]=value

        elif type=='one_d_sax':  

            for row in range(subset): #repr.shape[0]

                batch = []

                for column in range(row, subset): #repr.shape[0]
                    
                    batch.append((random_entries[row], random_entries[column], shared_repr[:,TF,:]))

                    if len(batch) == batch_size:
                        futures.append(executor.submit(process_batch, normalized[0,:,:,np.newaxis], batch, 'one_d_sax'))
                        batch = []

                if batch:  # Submit any remaining pairs
                    futures.append(executor.submit(process_batch, normalized[0,:,:,np.newaxis], batch, 'one_d_sax'))
                
                for process in concurrent.futures.as_completed(futures):
                    
                    results=process.result()

                    for row, column, value in results:
                        distances[np.where(random_entries==row)[0][0], np.where(random_entries==column)[0][0]]=value
                               
    shm.close()  # Close the shared memory
    shm.unlink()  # Remove the shared memory block
   
    return distances

#Construction of the total distance matrix // A total distance matrix HAS NOT been calculated as it takes an exorbitant amount of time

dist=[]

for TF in range(145):
    #Note that if one wanted to calculate the distance profile for a subset of Gene Profiles, here one should use random_entries and subset parameters
    dist.append( get_distance_matrix(TF, repr=one_d_repr, type='one_d_sax') )

distances=np.array(dist)

#This returns a distance matrix of dimensions (N_genes, N_genes). It is an upper triangular matrix. Each entry is the mean euclidean distance
#lower bound amongst all TF indices.
dis_mat=np.mean(distances,axis=0)

#Just an alternative summarization using the euclidean distance of euclidean lower bounds per distance matrix position
dis_mat=np.linalg.norm(distances, ord=2, axis=0)

#Making the upper triangular matrix symmetric // This is the final distance matrix that can then be used for either visualization and clustering
sym_di=dis_mat + dis_mat.T - np.diag(dis_mat.diagonal())

# Using the aforementioned format on a subset of Gene Profiles 

#These are just two different methods to import ACS coexpression data in two different operating systems
with open(r"C:\Users\nwnta\Documents\Gene_Coexp\Total_matrix.npy",'rb') as f: #Windows
    coexp=np.load(f, allow_pickle=True)

with open(r"/mnt/c/Users/nwntas/Downloads/Total_matrix.npy",'rb') as f: #Linux
    coexp=np.load(f, allow_pickle=True)


#Two different initializations of random Gene Profile indices, based on the accepted format for get_distance_matrix
random_inds=np.concatenate( (np.argsort(coexp[6321,:])[-30:], np.random.choice(np.arange(result.shape[0]), replace=False, size=20)) )
random_inds=np.concatenate( (np.argsort(coexp[6321,:])[-30:],  np.argsort(coexp[6321,:])[:50]) )

#Estimation of the total distance profile for the subset of Gene Profiles
dist=[]
for TF in range(145):
    dist.append( get_distance_matrix(TF,repr=one_d_repr,random_entries=random_inds, subset=80, type='one_d_sax') )

distances=np.array(dist)
dis_mat=np.mean(distances,axis=0)
sym_di=dis_mat + dis_mat.T - np.diag(dis_mat.diagonal())



### Second clustering attempt /TCK kernel ###

#Initialization  of the ensemble-based model with 200 Gaussian mixture models, each with 40 different clusters
model=TCK.TCK(G=200, C=40)

#a=result[random_inds,:,:]
#noise = np.random.normal(0, 1e-2, a.shape) #This code is only used when one wants to employ the model using a subset of Gene Profiles
#a=a+noise
#b=np.moveaxis(a,1,2)

#Running the model // Note that I here symbolizes the rounds of EM optimization. Only low values work, else problems with convergence.
K=model.fit(result, I=2)  

#This computes the distance matrix // Check the corresponding documentation for more info
res=model.predict('tr-tr')



### Third clustering attempt /SOM ###

#Extraction of the Catch22 feature set per TF biding profile per Gene Profile
data=np.array([ [ catch22.catch22_all(result[ind,row,:])["values"] for row in range(result.shape[1]) ] for ind in range(result.shape[0])], dtype=np.float32)

def generate_corr(profile: np.ndarray) -> np.ndarray:

    '''
    * This function accepts as input a Gene Profile.
    * This function returns an upper triangular correlation matrix of dimensions (22, 22) corresponding to input Gene Profile.
    '''
    
    #Initialization of a SOM grid with 100 prototype vectors
    s=som.Somoclu(n_columns=10, n_rows=10)
    s.train(profile)

    #This contains a numpy array that includes all component planes of the trained SOM
    sm=s.codebook

    #Initialization of the resulting correlation matrix
    n_features=profile.shape[1]
    corr_mat=np.zeros( shape=(n_features,n_features) )

    for row in range(n_features):

        #Flattening a component plane to a vector
        feat1=sm[:,:,row].flatten()
        #Normalizing the component plane
        feat1 = ( feat1-np.mean(feat1) ) / np.std(feat1)

        for column in range(row, n_features):
            
            #Enforcing the same flattening and normalization scheme on another component plane
            feat2 = sm[:,:,column].flatten()
            feat2 = ( feat2-np.mean(feat2) ) / np.std(feat2)
            
            #Filling the matrix with the pearson correlation of the flattened component planes
            corr_mat[row, column]=np.corrcoef(feat1, feat2)[0,1]
    
    return corr_mat

#This is the resulting 3D array that contains all correlation matrices per Gene Profile
dists=np.array([generate_corr(data[gene]) for gene in range(data.shape[0])])



#This is just a way to load the result of the aforementioned procedure given that this computation has already taken place 
# // Quite intensive
with open(r"C:\Users\nwnta\Documents\dists.npy","rb") as f: 
    dists=np.load(f,allow_pickle=True)


#Re-arranging the correlation matrices and using them to compute their pairwise distance
arr_reshaped = dists.reshape(dists.shape[0], -1)
frobenius_norm_vector = pdist(arr_reshaped, metric='euclidean')
frobenius_norm_matrix = squareform(frobenius_norm_vector)

#Optional: Normalization of the resulting distance matrix
mu,std=np.mean(frobenius_norm_matrix.flatten()), np.std(frobenius_norm_matrix.flatten())
nr=(frobenius_norm_matrix-mu)/std




### Validation of SOM clusters ###

#Loading of 3D coordinate Data
coords=np.loadtxt(r"C:\Users\nwnta\Documents\3D_gene_location.csv",delimiter=',', dtype=object)


#Formating my two datasets (coords and dists) in a way that they are comparable with each other by containing the same entries

#Indices refer to the Gene Profiles that should be retained from the original dataset
indices=[]

#These are the entries of 3D coordinates that should be deleted since they do not correspond to a known Gene Profile
to_del=[]

c=0
for gene in coords[:,0]:
    c+=1
    if m:=[cand for cand in index.keys() if re.search(rf"{gene}", cand)]:
        res=min(m)
        coords[c-1,0] = res
        indices+=[ index[res] ] #The index dictonary used here derives from the coexpression_setup script
    else: 
        to_del+=[c-1]

#These are the formatted datasets
dists=dists[indices]
coords = np.delete(coords, to_del, axis=0)

#This is the Hierarchical clustering of the SOM distance matrix using the ward metric
links=linkage(frobenius_norm_vector, method='ward')

#Bayesian Optimization to find the split that gives the best silhouetter score 

#Prior data for optimal ward cut-off estimation using a Gaussian Process
X_train, y_train=[],[]
for num in np.linspace(1, 5, 5):
    X_train+=[num]
    labels=fcluster(links, t=num, criterion='distance') 
    y_train+=[silhouette_score(frobenius_norm_matrix, labels=labels, metric='precomputed')]

X_train, Y_train=np.array(X_train).reshape(-1,1), np.array(y_train).reshape(-1,1)

#Defining the necessary functions for Bayesian Optimization
def UCB(mean: np.array, std: np.array, coef: float)-> np.array:
    return mean + coef * std

def calculate_state(model,test_data, ucb_coef=8):
    y_mean,y_std=model.predict(X=test_data,return_std=True)
    return UCB(y_mean,y_std,ucb_coef)

def training(train_data:np.array, known_values: np.array, x_test:np.array, cycles: int = 2, scale: float = 1)-> tuple:
    #Defining the RBF kernel 
    kernel=RBF(length_scale=scale)
    #Initialization of the Gaussian Process Regressor 
    gp=GP(kernel=kernel)

    #Fitting the known data
    gp.fit(X=train_data,y=known_values)
    ucb=calculate_state(gp,x_test,ucb_coef=8)
    
    #Finding the maximum point to be considered based on the UCB criterion
    next_point=x_test[np.argmax(ucb)]

    if cycles!=0:

        decay=lambda x: 0.5 * np.exp(-0.5 * x)

        for cycle in range(cycles):
            
            print(f"Cycle: {cycle}, {next_point}")
    
            
            #Update the training data
            train_data=np.append(train_data, next_point.reshape(1,-1),axis=0)

            labels=fcluster(links, t=next_point, criterion='distance')
            new_val=[[silhouette_score(frobenius_norm_matrix, labels=labels, metric='precomputed')*100_000]]
            known_values=np.concatenate((known_values,new_val))

            if known_values.max()>new_val:
                break

            #Fit the new GP
            gp.fit(X=train_data,y=known_values.reshape(-1,1))

            #Set a new test set 
            dT=decay(cycle)
            x_test=np.linspace(next_point-dT, next_point+dT, 100).reshape(-1,1)

            #Calculate the conditional parameters for my all points that belong to my test space
            ucb=calculate_state(gp,x_test,ucb_coef=8)

            next_point=x_test[np.argmax(ucb)]
        
        return known_values.max()/100_000, train_data[known_values.argmax()]
    else:
        return "A cycle number is required"
    
training(X_train, Y_train*100_000, np.linspace(1, 5, 5).reshape(-1,1), cycles=5, scale=0.35)


#t value found using Gaussian Process Optimization // Results in 2974 clusters for the ACS case
labs=fcluster(links, t=2.57110839, criterion='distance')  #Baseline clusters estimation


#Estimation based on ACS coexpression
links_coexp=linkage(coexp, metric='euclidean', method='ward')
labs_coexp=fcluster(links_coexp, t=2479, criterion='maxclust') 

#Estimation of the distances to chromosomal origin
origin_dist=coords[:,7].astype(np.float32)
#origin_dist= ( origin_dist - np.min(origin_dist)) / (np.max(origin_dist) - np.min(origin_dist)) #Normalization for some reason, I dont think i need it
pairwise_distances = pdist(origin_dist[:, None], metric='euclidean')

origin_links=linkage(pairwise_distances, method='ward')
#The number of t here corresponds to the number of clusters defined by the optimal threshold for the gene profiles 
origin_labs=fcluster( origin_links, t=2479, criterion='maxclust') 


#Inter_gene distances
positions=coords[:, 4:7].astype(np.float32)
position_links = linkage(positions, method='ward', metric='euclidean')
position_labs = fcluster(position_links, t=2479, criterion='maxclust')

#Comparison of clustering attempts based on different qualities
adjusted_rand_score(labs, labs_coexp) #What i find is 0.0030396222165263356 for the ACS case // This means that there is no congruence between the two clusterings
adjusted_rand_score(labs, origin_labs) #I find a value that is really close to zero // 
adjusted_rand_score(labs, position_labs) #I find 0.001105155419977364 for the intergene distance // This means that similarly to all other tests there is no congruence between the two clusterings



### Visualizations ###

#TCK kernel // heatmap
fig =  plt.figure(figsize=(6,6))
h = plt.imshow(res)
plt.title("TCK matrix (sorted)")
plt.colorbar(h)
plt.xlabel("MTS class")
plt.ylabel("MTS class")
plt.show()

#TCK kernel // kernelPCA
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components=2, kernel='precomputed')
embeddings_pca = kpca.fit_transform(res)
fig =  plt.figure(figsize=(8,6))
plt.scatter(embeddings_pca[:,0], embeddings_pca[:,1], c=[1]*30 + [0]*49, s=10, cmap='tab20')
plt.title("Kernel PCA embeddings")
plt.show()



# SOMs
matplotlib.use('Agg')

sns.clustermap(data=frobenius_norm_matrix, method='average', cmap='viridis_r')
#plt.xticks(fontsize=3)
plt.savefig("trial_cluster_10")
plt.close()


def visualize_clustermap(distance_matrix: np.ndarray, link: str, output_name: str):

    '''
    This function is responsible for visualizing the clustered distance matrix for a pre-defined distance metric.
    The function accepts a symmetric matrix as input. It performs hierarchical clustering using a defined linkage parameter.
    It returns a figure in the currect directory.
    '''
    #Defining the hierarchical clustering
    D=squareform(distance_matrix)
    linkage_matrix = linkage(D, method=link)
    
    #Plotting the clustering
    sns.clustermap(distance_matrix, row_linkage=linkage_matrix, col_linkage=linkage_matrix, cmap="viridis_r")

    plt.savefig(f"{output_name}.png")
    plt.close()


### This is just a PCA visualization trial // Could be deleted ### 
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D

mu=np.mean(sym_di,axis=-1, keepdims=True)
std=np.std(sym_di, axis=-1, keepdims=True)
norm_ds=(sym_di-mu)/std


pca = PCA(n_components=2)
tr=pca.fit_transform(norm_ds)

#Loading projection of my data
loads=pca.components_.T * np.sqrt(pca.explained_variance_)
t=norm_ds @ loads

labs=np.array([1]*30 + [0]*50)

plt.scatter(x=t[np.where(labs==1)[0],0], y=t[np.where(labs==1)[0],1], edgecolors='k', label="Over",cmap='viridis')
plt.scatter(x=t[np.where(labs==0)[0],0], y=t[np.where(labs==0)[0],1], edgecolors='k', label="Under", cmap='viridis')
sns.despine()
plt.legend()
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)")
plt.savefig("test")
plt.close()


#This is just to stack the summaries of the genes in an array of dimensions N_genes X N_TFs
gene_summaries=np.empty((result.shape[1]))
for id in range(result.shape[0]):
    gene_summaries=np.vstack((gene_summaries, cutoff_summary(result[id,:,:], cutoff=0.7)))


### Relationship between mean ACS score and TF binding preferences ###

with open(r"C:\Users\nwnta\Documents\output.bed") as f:
    table=[line.rstrip().split("\t") for line in f]

table=np.array(table)

#A total of 97/145 (~67%) TF have data related to their binding affinity to yeast genes // motifs here is imported from pwm_functions 
common_TFs=list(set(motifs.keys()) & set(np.unique(table[:,6])))

binding_table={TF: np.unique(table[table[:,6]==TF][:,4]) for TF in common_TFs}

indexed_table=dict(map(lambda x: (x[0], [index[gene] for gene in x[1]]),binding_table.items()))


observations={k:np.mean([coexp[c] for c in combinations(v,2)])  for k,v in indexed_table.items()}

input_dict = {k:(len(v), observations[k]) for k,v in indexed_table.items()}

#Creating bootstrapped datasets for each TF gene set
def ACS_bootstrap(gene_set_size: int, iter: int, obs: float) -> float:
    
    '''
    This function uses bootstrap to compute the p-value corresponding to the mean ACS score for the gene targets of a TF.
    * The function accepts as input the size of the gene target set, the number of iterations for bootstrap, and the observed ACS value for the true gene set.
    * It outputs the corresponding p-value.
    '''
    
    distr=[]
    for _ in range(iter):
        #Select a random set of genes equal in size to the original set
        random_genes=np.random.choice(np.arange(start=0, stop=6692), size=gene_set_size, replace=True)
        #Get the mean ACS score for the random set of genes
        distr+=[np.mean([coexp[c] for c in combinations(random_genes,2)])]

    distr = np.array(distr)
    #Compute the resulting p-value
    return (np.where(distr>=obs)[0].shape[0] + 1) /(iter + 1) 

#Computing the bootstrap p-values for all TFs using a total of 1000 iterations
boot_pvals={k: ACS_bootstrap(gene_set_size=v[0], iter=1000, obs=v[1]) for k,v in input_dict.items()}

import json

#Since the computation is very intensive, the pre-computed p-values can be loaded for further analysis
with open(r"C:\Users\nwnta\Documents\TF_pvals.json", 'w') as f:
    json.dump(boot_pvals, f)

#Correcting the p-values for multiple testing using FDR
from scipy import stats
boot_pvals=np.array(list(boot_pvals.values()))
adj_ps=stats.false_discovery_control(boot_pvals)


#Distinguishing between significant and non-significant corrected p-values and plotting the results
fig, ax =plt.subplots()
not_sig=np.where(adj_ps>0.05)[0]

x,y = list(map(lambda x: x[0], input_dict.values())), list(map(lambda x: x[1], input_dict.values()))

#A scatter plot comparing gene set size legnth with mean ACS value and significance
ax.scatter(x,
           y, 
           c=[ 'green' if not TF in not_sig  else 'red' for TF in range(adj_ps.shape[0]) ],
           edgecolors='k')

ax.set_xlabel("Number of target genes")
ax.set_ylabel("Mean ACS value")

sns.despine()
handles = [
    plt.Line2D([], [], marker='o', color='w', markerfacecolor='green', markersize=8, label='Significant (Padj$\leq$0.05)'),
    plt.Line2D([], [], marker='o', color='w', markerfacecolor='red', markersize=8, label='Not Significant (Padj>0.05)')
]
ax.legend(handles=handles)
plt.savefig('adj.png')