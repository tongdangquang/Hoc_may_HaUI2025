from . import fcm
from . import pcm
from . import pfcm
from validity import validity
import numpy as np
import pandas as pd

algos={'fcm':fcm, 'pcm':pcm, 'pfcm':pfcm}

def use(name, data, max_iter, num_of_clus):
    if name == 'fcm':
        fcm_object=fcm.fcm(data=data, max_iter=max_iter, num_of_clus=num_of_clus)
        fcm_object.fit()
        return fcm_object
    elif name=='pcm':
        fcm_object=fcm.fcm(data=data, max_iter=max_iter, num_of_clus=num_of_clus)
        fcm_object.fit()
        pcm_object=pcm.pcm(data=data, max_iter=max_iter, num_of_clus=num_of_clus, init_typical=fcm_object.u, init_centroid=fcm_object.v)
        pcm_object.fit()
        return pcm_object
    else:
        fcm_object=fcm.fcm(data=data, max_iter=max_iter, num_of_clus=num_of_clus)
        fcm_object.fit()
        pcm_object=pcm.pcm(data=data, max_iter=max_iter, num_of_clus=num_of_clus, init_typical=fcm_object.u, init_centroid=fcm_object.v)
        pcm_object.fit()
        pfcm_object=pfcm.pfcm(data=data, max_iter=max_iter, num_of_clus=num_of_clus,init_typical=pcm_object.t, init_centroid=fcm_object.v, init_membership=fcm_object.u, a=2)
        pfcm_object.fit()

        return pfcm_object


def validity2(data: np.ndarray, membership: np.ndarray, time, iteration, algo_name='UNKNOWN'):
    labels = np.argmax(membership, axis=1)
    result_df = pd.DataFrame([[
        algo_name.upper(),
        iteration,
        round(time, 5),
        round(validity.davies_bouldin(data, labels=labels), 5),
        round(validity.partition_coefficient(membership), 5),
        round(validity.dunn_fast(data, labels=labels), 5)
    ]], columns=["Algo", "Iters", "Time", "DB", "PC", "DI"])
    return result_df
