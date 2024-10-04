import numpy as np
import re
import concurrent.futures
from Scripts.pwm_functions import gene_profile
import os

### TF PWM dictionary set-up ###

motifs=dict()
with open(os.getcwd()+"\\Data\\target_motifs.txt","r") as f:

    for line in f:
        
        #If I encounter a blank line, meaning that I am switching TF, then append to the dictionary

        if not len(line.strip()):
             
             PWM=PWM[4:].reshape(4,-1)
             motifs[motif_name]=PWM
             continue
        
        #If you find Motif in the line, that means that it is a header line

        if m:=re.search(r"Motif \d{1,}\s(\w+)\s", line):
            PWM=np.empty(shape=(4,))
            motif_name=m.group(1)
            continue

        #Else, it is just a line that contains data for the last istance of header
      
        pwm_line=np.array(line.strip().split("\t")[1:]).astype(float)
        PWM=np.append(PWM,pwm_line)


### Setting up a dictionary containing the selected upstream intervals for each gene ###

sequences=dict()
with open(os.getcwd()+"\\Data\\sequences1.fa","r") as f: #sequences1.fa contains sequences of length 500 + max{len(PWM)}
    
    for line in f:

        #If it is a header line, define the name

        if m:=re.search(r">([-a-zA-Z0-9]+)\([+-]\)", line):
            name=m.group(1)
            continue

        #Else, it is a sequence that is appended to the last known gene name

        sequences[name]=line.strip()


### Final result ###

#Setting up an array to wrap the results
total_profile=np.empty(len(motifs), dtype=object)

#Set up a pool executor for concurrent execution
with concurrent.futures.ProcessPoolExecutor() as executor:
    #Create all processes
    control={executor.submit(gene_profile, pwm, sequences):index for index, pwm in enumerate(motifs.values())}

    #Save the processes as completed // Retain the original index/TF identity
    counter=0
    for process in concurrent.futures.as_completed(list(control.keys())):
        counter+=1
        idx=control[process]
        total_profile[idx]=process.result()
        print(f"Finished {counter/len(motifs)*100:.2f} % of PWMs")

with open(r"C:\Users\nwnta\Documents\TF_binding.npy","rb") as f:
    result=np.load(f, allow_pickle=True)