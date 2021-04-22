import numpy as np

def recall(y_pred,y_true):
    legnagyobb_valszinuseg = np.tile(y_pred.argmax(axis=1),[len(y_pred[0])])
    gyakorisagok = np.unique(y_true, return_counts=True)[1]
    gyakorisagok_iteralva = np.repeat(gyakorisagok,len(y_true))
    tippek = np.arange(len(y_pred[0]))
    tippek_iteralva = np.repeat(tippek,[len(y_true)])
    helyes_klasszifikaciok = np.tile(y_true,[len(y_pred[0])])
    vp = np.logical_and((legnagyobb_valszinuseg == tippek_iteralva), (helyes_klasszifikaciok == tippek_iteralva))
    sumthis = np.sum(np.where(vp,1/gyakorisagok_iteralva,0))
    result = sumthis /  len(y_pred[0])
    return result





