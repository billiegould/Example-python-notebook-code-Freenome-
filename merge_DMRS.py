import pandas as pd
import os
from argparse import ArgumentParser
import glob

p = ArgumentParser()
p.add_argument("--in_dir", dest="in_dir", required=True)
p.add_argument("--out_bed", dest="out_bed", required=True)
args = p.parse_args()

#files = os.system(f"ls ./{args.in_dir}/*_DMRs.txt")
files = glob.glob(f"{args.in_dir}/*_DMRs.txt")
print(f"FILES: {files}")
dfs = []
for file in files:
    df = pd.read_csv(file, sep="\t", header=None)
    print(df.head())
    df = df[df.iloc[:,3]<0.001] # q-value
    df = df[df.iloc[:,4]<=-0.2]     #mean meth diff healthy - cancer
    df = df[df.iloc[:,8]<=0.25] # healthy methylation rate
    dfs.append(df)

df_all = pd.concat(dfs)
df_all.columns = ["| chr |"," start","| stop |","q-value","| mean methylation difference |"," #CpGs |"," p (MWU)"," | p (2D KS) ","| mean g1 ","| mean g2"]
#df_all = df_all.apply(pd.to_numeric, args=('coerce',))
df_all_sorted = df_all.sort_values(by="q-value")
print(df_all_sorted.shape)
print(df_all_sorted.head())

with open(f"./{args.out_bed}","w") as fout:
    df_all_sorted.to_csv(fout, index=False, header=None, sep="\t")
