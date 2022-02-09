from calzone import Calzone
from featureio import FeatureIO
from pineapple.contrib.components.data_factories.matrix_factory import MatrixFactory as mf

from pineapple.core import experiment_context
import datetime
experiment_context.reset_context(0, "x", "x", datetime.datetime(2021, 2, 2,0,0))
import argparse
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import os

def make_input_df(matrix, df_healthies, title):
    sample_names = [f"g2_{i}" for i in range(matrix.shape[0])]
    df = pd.DataFrame(matrix.x[:,:,0]/matrix.x[:,:,1]).T
    df.columns = sample_names
    df = df.fillna(".")
    df_met = pd.concat([df_healthies, df], axis=1)
    print(df_met.head())
    print(df_met.shape)
    
    #with open(f"./{title}_tissue_met_input.tsv", "w") as fout:
    #    df_met.to_csv(fout, sep="\t", index=False)
    print("Make input Complete")
    return df_met
    
    
def split_by_chr(input_df, title):
    os.system(f"mkdir -p {title}_met_chr_inputs")
    for i in range(1,23):
        df_chr = input_df[input_df["chr"]==f"chr{i}"]
        with open(f"./{title}_met_chr_inputs/{title}_chr{i}.tsv","w") as fout:
            df_chr.to_csv(fout, index=False, sep="\t")
            print(f"Chr{i} complete.")
    return
    
    
def main():
    p = argparse.ArgumentParser()

    p.add_argument("--training_classes", dest="training_classes", required=True, nargs="+", action="extend")
    p.add_argument("--out_prefix", dest="out_prefix", required=True)
    args = p.parse_args()
	
    df_healthies = pd.read_csv("./healthy_plasma_met_input.csv", sep=",",header=0)
	
    mat_tissue = mf.create_from_training_classes(args.training_classes,
                                     		'per_cpg_hmcfc_min_5_cpg_dense/v1',
                                     		class_labels=np.repeat(1,len(args.training_classes)))
                                          		
    df_met = make_input_df(mat_tissue, df_healthies, args.out_prefix)
    
    split_by_chr(df_met, args.out_prefix)
    
    
if __name__=="__main__":
	main()
