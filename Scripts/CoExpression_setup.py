import numpy as np
import re
import os

sequences=dict()
with open(os.getcwd()+"\\Data\\sequences1.fa","r") as f: #sequences1.fa contains sequences of length 500 + max{len(PWM)}
    
    for line in f:

        #If it is a header line, define the name

        if m:=re.search(r">([-a-zA-Z0-9]+)\([+-]\)", line):
            name=m.group(1)
            continue

        #Else, it is a sequence that is appended to the last known gene name

        sequences[name]=line.strip()


index={gene: ind for  ind,gene in enumerate(sequences)}


coexp_mat=np.empty( ( len(index), len(index) ) )

path=r"C:\Users\nwnta\Documents\Gene_Coexp"

unknown=[]
for file in os.listdir(path=path):

    with open(path+f"\\{file}") as f:

        for line in f:

            if m:=re.search("(?P<flag>Q)?(\d+)?\t(?P<gene>[a-zA-Z0-9-]+)(\t.*)?\t(?P<cor>\d+\.\d+)?", line.strip()):

                m=m.groupdict()

                if not m['gene'] in index.keys():
                    unknown.append(m['gene'])
                    continue

                if m["flag"]:
                    row_ind=index[m['gene']]
                    continue
            
                coexp_mat[row_ind, index[m["gene"]] ] = float(m['cor'])

            continue

np.save(r"C:\Users\nwnta\Documents\Gene_Coexp\Total_matrix", coexp_mat)