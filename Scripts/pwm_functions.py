import numpy as np
from typing import Dict

def score_interval(interval: str, pwm: np.ndarray)-> np.float64:

    '''
    This function accepts a nucleotide sequebce that should be equal in size to the number of columns of the target PWM. 
    It scores the aforementioned sequence by summing PWM score values per position based on the nucleotide present.
    The function also normalizes the calculated score by the max possible score as defined by the consensus motif.
    '''

    assert len(interval)==pwm.shape[1], "Mismatch between sequence length and PWM length"

    #Set up a dictionary that corresponds to the nucleotide arrangement of all PWMs
    pwm_labels={"A": 0, "C": 1, "G": 2, "T": 3}

    #project each nucleotide to the corresponding PWM line entry
    edited_interval=list(map(lambda x: pwm_labels[x], interval))

    #Defining the max possible score of the PWM sequence
    max_score=np.sum(np.max(pwm,axis=0))

    #return the sum of the PWM entries
    return np.sum( pwm[ edited_interval, np.arange(pwm.shape[1]) ] ) /max_score


def TF_profile( TF: np.ndarray, genes: Dict[str,str]) -> np.ndarray:

    '''
    This function accepts a PWM as input along with all target gene sequences.
    It scores the provided PWM across the totality of the target region of each gene,
    and returns a matrix of dimensions N_genes X N_intervals, where N_intervals is the total number 
    of subsequences scored based on PWM length.

    Note: As some target sequences may differ in length due to chromosomal coordinates, those aberrant 
    sequences have been padded with NaN values so as no to disturb matrix dimensionality
    '''

    #Setting up the Kmer length based on input PWM
    kmer=TF.shape[1]
    #Initializing the resulting matrix // Since most sequences are of length 500, the value is hardcoded
    TF_profile=np.empty(shape=(1, 500-kmer+1))

    for gene in genes.values():

        #Calculation of PWM scoring per gene sequence 
        gene_profile=np.array([])

        for init in range(0,len(gene)-kmer+1):
            gene_profile=np.append( gene_profile, score_interval(gene[init: init+kmer], TF) )

        #Padding sequences that differ from the expected length
        if gene_profile.shape[0]!=TF_profile.shape[1]:
            #Defining how extensive the padding should be
            pad_size=TF_profile.shape[1]-gene_profile.shape[0]
            gene_profile=np.pad(gene_profile, (0, pad_size), constant_values=np.nan)

        #Appending PWM scoring to the total TF profile
        TF_profile=np.vstack((TF_profile, gene_profile))
    
    return TF_profile[1:,:]
