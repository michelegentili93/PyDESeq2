# %%

import os
import pickle as pkl
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from pydeseq2.utils import load_example_data


counts_df = pd.read_table("/Users/rloeb/PyDESeq2/datasets/TCGA-BRCA_raw_RNAseq.tsv", index_col=0)
genes_to_keep = counts_df.columns[counts_df.sum(axis=0) >= 10]
counts_df = counts_df[genes_to_keep]
metadata = pd.read_table("/Users/rloeb/PyDESeq2/datasets/TCGA-BRCA_clinical.tsv", index_col=0)


path_lengths = '/Users/rloeb/PyDESeq2/datasets/recount3_gene_bp_length.parquet'
gene_lenghts = pd.read_parquet(path_lengths)
#sdrop gene_id from gene_lenghts nutiindex
gene_lenghts = gene_lenghts.reset_index().set_index("gene_name")["bp_length"]
# remove gene_name version number
gene_lenghts.index = gene_lenghts.index.str.split(".").str[0]
# take mean length per gene (should be the same for all versions)
gene_lenghts = gene_lenghts.groupby(gene_lenghts.index).mean()


counts_df.index = counts_df.index.str.split(".").str[0]
# Take most variable gene for duplicates
variances = counts_df.var(axis=1)
# Take most variable duplicate
counts_df = counts_df.iloc[variances.reset_index().groupby('gene').idxmax()[0]]

# Take counts df genes that are in gene_lenghts
counts_df = counts_df.loc[counts_df.index.intersection(gene_lenghts.index)]

# FOR DEBUGGING
counts_df = counts_df.sample(500, axis=0)




# %%


import seaborn as sns
import matplotlib.pyplot as plt

# Plot vst against gene length
counts = (1+counts_df).T.stack().reset_index()
counts.columns = ["sample", "gene", "counts"]
counts = counts.join(gene_lenghts, on="gene")
# Use log x scale
sns.scatterplot(x="bp_length", y="counts", data=counts)
plt.xscale('log')
plt.yscale('log')



# %%

import seaborn as sns
import matplotlib.pyplot as plt

# Plot vst against gene length
counts = (1+counts_df.mean(axis=1)).to_frame().join(gene_lenghts)
# Use log x scale
sns.scatterplot(x="bp_length", y=0, data=counts)
plt.xscale('log')
plt.yscale('log')







# %%






dds = DeseqDataSet(
    counts=counts_df.T,
    metadata=metadata,
    design_factors="primary_diagnosis", # random metadata colummn with no Nan
    refit_cooks=True,
    n_cpus=8,
)


# Compute vst
dds.vst(fit_type="mean")


# %%
vst = pd.DataFrame(
    dds.layers["vst_counts"],
    index = dds.obs_names,
    columns = dds.var_names
)


# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Plot vst against gene length
vst_df = vst.stack().reset_index()
vst_df.columns = ["sample", "gene", "vst"]
vst_df = vst_df.join(gene_lenghts, on="gene")
# Use log x scale
sns.scatterplot(x="bp_length", y="vst", data=vst_df)
plt.xscale('log')

# %%
# Spearman cor
from scipy.stats import spearmanr
spearmanr(vst_df["bp_length"], vst_df["vst"])
# %%

#dds.deseq2()

dispersions = dds.varm["dispersions"]
genes = dds.var.index
disp_df = pd.DataFrame(dispersions, index=genes, columns=["dispersion"])


#import ipdb; ipdb.set_trace();

# join gene_lenghts with disp_df on gene and gene_name
disp_df_2 = pd.concat([disp_df, gene_lenghts], axis=1, join="inner")


# %%

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
fig, ax = plt.subplots(figsize=(10, 10))
sns.scatterplot(x="bp_length", y="dispersion", data=disp_df_2, ax=ax)
ax.set_xlabel("Gene length (bp)")
ax.set_ylabel("Dispersion")
#plt.xscale('log')
plt.show()



# %%
dds.plot_dispersions()


# %%

from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(disp_df_2['bp_length'], disp_df_2['dispersion'])
#get me the residuals from the regression
residuals = disp_df_2['dispersion'] - (disp_df_2['bp_length'] * slope + intercept)
#plot the residuals
fig, ax = plt.subplots(figsize=(10, 10))
sns.scatterplot(x="bp_length", y="dispersion", data=disp_df_2, ax=ax)
ax.set_xlabel("Gene length (bp)")
ax.set_ylabel("Dispersion")
plt.plot(disp_df_2['bp_length'], residuals, 'o', color='blue')
plt.show()

                                                         
