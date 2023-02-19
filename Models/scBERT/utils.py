import scanpy as sc, numpy as np, pandas as pd, anndata as ad

def transfer_gene2vec_as_weight(emb_loc:str,np_loc:str=None):
    # transfer gene2vec as weight
    # currently use https://github.com/jingcheng-du/Gene2vec.git
    #in:
    # emb_loc: gene2vec model location
    # np_loc: numpy file location
    try:
        from gensim.models import Word2Vec, KeyedVectors
        gene2vec_model = KeyedVectors.load_word2vec_format(emb_loc)
        print(f"Loaded gene2vec model with {len(gene2vec_model.index_to_key)} genes")
        if np_loc is not None:
            # save as npy
            np.save(np_loc, gene2vec_model.vectors)
            print(f"Saved gene2vec model as {np_loc}")
        return gene2vec_model
    except:
        raise ValueError ("Please install gensim to use gene2vec and use embedding file in  https://github.com/jingcheng-du/Gene2vec.git ")

def pre_process_sc_raw(raw_data_loc:str,name_ref:list,preprocessed_loc:str):
    """
    from single cell raw data to preprocessed data
    in:
    raw_data_loc: raw data location './data/your_raw_data.h5ad'
    name_ref: gene name reference
    preprocessed_loc: preprocessed data location './data/your_preprocessed_data.h5ad'
    
    """
    print(f"read raw data from {raw_data_loc}")
    from scipy import sparse
    data = sc.read_h5ad(raw_data_loc)
    gene_nb = len(name_ref)
    counts = sparse.lil_matrix((data.X.shape[0],gene_nb),dtype=np.float32)
    obj = data.var_names.tolist()
    print(f"get gene name from reference and transfer to sparse matrix")
    for i in range(len(name_ref)):
        if i % 2000 == 0: print(f"processing {i}/{len(name_ref)}")
        if name_ref[i] in obj:
            loc = obj.index(name_ref[i])
            counts[:,i] = data.X[:,loc]
    counts = counts.tocsr()
    print(f"convert to anndata..")
    new = ad.AnnData(X=counts)
    new.var_names = name_ref
    new.obs_names = data.obs_names
    new.obs = data.obs
    new.uns = {'log1p': {'base': 2}} #data.uns
    print(f"pre-process..")
    sc.pp.filter_cells(new, min_genes=200)
    sc.pp.normalize_total(new, target_sum=1e4)
    sc.pp.log1p(new, base=2)
    if preprocessed_loc is not None:
        print(f"save preprocessed data to {preprocessed_loc}")
        new.write(preprocessed_loc)
    return new
