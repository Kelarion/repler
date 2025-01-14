


def dot_product(x,y):
    """Assume features are in first axis"""
    return np.einsum('k...i,k...j->...ij', x, y)


def centered_kernel_alignment(K1,K2):

    K1_ = K1 - np.nanmean(K1,-2,keepdims=True) - np.nanmean(K1,-1,keepdims=True) + np.nanmean(K1,(-1,-2),keepdims=True)
    K2_ = K2 - np.nanmean(K2,-2,keepdims=True) - np.nanmean(K2,-1,keepdims=True) + np.nanmean(K2,(-1,-2),keepdims=True)
    denom = np.sqrt(np.nansum((K1_**2),(-1,-2))*np.nansum((K2_**2),(-1,-2)))

    return np.nansum((K1_*K2_),(-1,-2))/np.where(denom, denom, 1e-12)


## X has shape (num_neuron, ..., num_item)
## Y has shape (num_label, ..., num_item)

Ky = dot_product(Y, Y)
Kz = dot_product(Z, Z)

cka = centered_kernel_alignment(Kz, Ky)