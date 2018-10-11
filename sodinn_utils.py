"""
Various helping functions. Plotting, saving/loading results, creating synt 
cubes, image processing.
"""

from __future__ import division
from __future__ import print_function

__all__ = ['plot_traindata',
           'save_res',
           'load_res',
           'load_model',
           'svd_decomp',
           'get_cumexpvar',
           'frame_shift',
           'inspect_patch_multik',
           'normalize_01',
           'get_indices_annulus',
           'create_synt_cube',
           'create_feature_matrix']

import torch
import numpy as np
from keras import models
from skimage.draw import circle
from vip_hci.preproc import cube_derotate, cube_crop_frames, cube_derotate
from vip_hci.preproc import frame_shift, frame_crop
from vip_hci.phot import noise_per_annulus, cube_inject_companions
from vip_hci.var import pp_subplots as plots
from vip_hci.var import reshape_matrix, prepare_matrix, frame_center
from vip_hci.pca import pca, svd_wrapper, randomized_svd_gpu
from matplotlib.pyplot import (figure, show, subplot, title)
import cv2
import pickle
from matplotlib.pyplot import hist as histogram
import cupy


def normalize_01(array, mode='slice'):
    """
    """
    n1, n2, n3, n4 = array.shape
    array = array.copy()

    if mode == 'slice':
        array_reshaped = array.reshape(n1 * n2, n3 * n4)
    elif mode == 'sample':
        array_reshaped = array.reshape(n1, n2 * n3 * n4)
    else:
        raise RuntimeError('Normalization mode not recognized')

    minvec = np.abs(np.min(array_reshaped, axis=1))
    array_reshaped += minvec[:, np.newaxis]
    maxvec = np.max(array_reshaped, axis=1)
    array_reshaped /= maxvec[:, np.newaxis]
    return array_reshaped.reshape(n1,n2,n3,n4)



def plot_traindata(T, zeroind=None, oneind=None, full_info=False, 
                   plot_pair=True, dpi=100, indices=None, save_plot=False):
    """
    """
    xarr = T.x
    yarr = T.y
    if 'xnor' in T:
        xarrn = T.xnor
    
    if zeroind is None:
        zeroind = np.random.randint(0,xarr.shape[0]/2.)
    if oneind is None:
        oneind = np.random.randint(xarr.shape[0]/2.,xarr.shape[0])
    
    if full_info:
        msg1 = 'N samples : {} | Runtime : {}'
        print(msg1.format(T.nsamp, T.runtime))
        msg2 = 'FWHM : {} | PLSC : {} | K list : {}'
        print(msg2.format(T.fwhm, T.plsc, T.klist))
        msg3 = 'In Rad : {} | Out Rad : {} | Patch size : {}'
        print(msg3.format(T.inrad, T.outrad, T.sizepatch))
        msg4 = 'Collapse func : {} | Scaling : {}'
        print(msg4.format(T.collaf.__name__, T.scaling))
        msg5 = 'N patches : {} | Perc orig zeros : {}'
        print(msg5.format(T.npatches, T.perorigzeros))
        msg6 = 'Flux distro : {} | Par1 : {} | Par2 : {}'
        print(msg6.format(T.fluxdistro, T.fluxdistrop1, T.fluxdistrop2))
        msg7 = 'N injections : {} | Perc aug ones : {}'
        print(msg7.format(T.nsamp*0.5*T.peraugones, T.peraugones))
        msg8 = 'Aug shifts : {} | Aug range rotat : {}'
        print(msg8.format(T.shifts, T.rangerot))
        figure(figsize=(12,2))
        subplot(1, 3, 1)
        _ = histogram(T.fluxes, bins=int(np.sqrt(T.fluxes.shape[0])))
        title('Fluxes histogram')
        subplot(1, 3, 2)
        _ = histogram(np.array(T.dists).flatten(), 
                      bins=int(np.sqrt(T.fluxes.shape[0])))
        title('Distances histogram')
        subplot(1, 3, 3)
        _ = histogram(np.array(T.thetas).flatten(), 
                      bins=int(np.sqrt(T.fluxes.shape[0])))
        title('Thetas histogram')
        show()
        print()
    
    npatches = xarr[zeroind].shape[0]
    if plot_pair or save_plot:
        if indices is not None:
            zerarr = xarr[zeroind][indices]
            onearr = xarr[oneind][indices]
            if xarrn is not None: zerarrn = xarrn[zeroind][indices]
            if xarrn is not None: onearrn = xarrn[oneind][indices]
        else:
            zerarr = xarr[zeroind]
            onearr = xarr[oneind]
            if xarrn is not None: zerarrn = xarrn[zeroind]
            if xarrn is not None: onearrn = xarrn[oneind]

        if save_plot:
            print('{} | Sample {}'.format(int(yarr[zeroind]), zeroind))
            plots(zerarr, dpi=dpi, axis=False, vmin=xarr[zeroind].min(), 
                  vmax=xarr[zeroind].max(), save='patch_zero.pdf', colorb=False,
                  maxplots=npatches, horsp=0.1)
            if xarrn is not None:
                plots(zerarrn, axis=False, dpi=dpi, colorb=False,
                      save='patch_zero_nor.pdf', maxplots=npatches, horsp=0.1)
            print(int(yarr[oneind]),'| Sample', oneind) 
            plots(onearr, axis=False, vmin=xarr[oneind].min(), 
                  vmax=xarr[oneind].max(), dpi=dpi, save='patch_one.pdf', 
                  colorb=False, maxplots=npatches, horsp=0.1)
            if xarrn is not None:
                plots(onearr, axis=False, dpi=dpi, horsp=0.1,
                      save='patch_one_nor.pdf', colorb=False, maxplots=npatches)
        
        else:
            plots(zerarr, title='Unnormalized ZERO multiK patch', dpi=dpi,
                  axis=False, vmin=xarr[zeroind].min(), vmax=xarr[zeroind].max(),
                  maxplots=npatches, horsp=0.1)
            if xarrn is not None:
                plots(zerarrn, title='Normalized ZERO multiK patch', 
                      axis=False, dpi=dpi, maxplots=npatches, horsp=0.1)
            plots(onearr, title='Unnormalized ONE multiK patch', axis=False,
                  vmin=xarr[oneind].min(), vmax=xarr[oneind].max(), dpi=dpi,
                  maxplots=npatches, horsp=0.1)
            if xarrn is not None:
                plots(onearrn, title='Normalized ONE multiK patch', 
                      axis=False, dpi=dpi, maxplots=npatches, horsp=0.1)



def save_res(filename, data):
    pickle.dump(data, open(filename, "wb"))

def load_res(filename):
    out = pickle.load(open(filename, "rb"), encoding='latin1')
    return out

def load_model(filename):
    return models.load_model(filename)


def svd_decomp(cube, angle_list, size_patch, inrad, outrad, sca, k_list, 
               collapse_func, neg_ang=True, lr_mode='eigen', nproc=1,
               interp='nearneig', verbose=False):
    """
    """
    frsize = int(cube.shape[1])
    n_frames = cube.shape[0]

    if n_frames>1000:
        ind_for_svd = range(0,n_frames,2)
    else:
        ind_for_svd = range(0,n_frames)

    ann_width = outrad-inrad
    cent_ann = inrad + int(np.round(ann_width/2.))
    ann_width += size_patch+2
    matrix, annind = prepare_matrix(cube, sca, None, mode='annular', 
                                    annulus_radius=cent_ann, 
                                    annulus_width=ann_width, verbose=False)
    matrix_svd, _ = prepare_matrix(cube[ind_for_svd], sca, None, mode='annular', 
                                   annulus_radius=cent_ann, 
                                   annulus_width=ann_width, verbose=False)

    V = svd_wrapper(matrix_svd, lr_mode, k_list[-1], False, False, to_numpy=False)

    if verbose:
        print("SVD done")
        
    cube_residuals = []
    if neg_ang:
        cube_residuals2 = []

    for k in k_list:
        if lr_mode in ['cupy', 'randcupy', 'eigencupy']:
            matrix = cupy.array(matrix)
            transformed = cupy.dot(V[:k], matrix.T)
            reconstructed = cupy.dot(transformed.T, V[:k])
            residuals_ann = matrix - reconstructed
            residuals_ann = cupy.asnumpy(residuals_ann)
        elif lr_mode in ['pytorch', 'randpytorch', 'eigenpytorch']:
            matrix = matrix.astype('float32')
            matrix_gpu = torch.Tensor.cuda(torch.from_numpy(matrix))
            transformed = torch.mm(V[:k], torch.transpose(matrix_gpu, 0, 1))
            reconstructed = torch.mm(torch.transpose(transformed, 0, 1), V[:k])
            residuals_ann = matrix_gpu - reconstructed
        else:
            transformed = np.dot(V[:k], matrix.T)
            reconstructed = np.dot(transformed.T, V[:k])
            residuals_ann = matrix - reconstructed

        # This is a bottleneck when nframes grows. The cube derotation is not
        # very efficient in parallel. 
        residual_frames = np.zeros_like(cube)
        residual_frames[:,annind[0],annind[1]] = residuals_ann
        residual_frames_rot = cube_derotate(residual_frames, angle_list, 
                                            nproc=nproc, interpolation=interp)
        cube_residuals.append(collapse_func(residual_frames_rot, axis=0))
        if neg_ang: 
            residual_frames_rot_neg = cube_derotate(residual_frames, 
                                                    -angle_list, nproc=nproc,
                                                    interpolation=interp)
            cube_residuals2.append(collapse_func(residual_frames_rot_neg, axis=0))
    
    if neg_ang:
        return cube_residuals, cube_residuals2
    else:
        return cube_residuals


def get_cumexpvar(cube, expvar_mode, inrad, outrad, size_patch, k_list=None,
                  verbose=True):
    """
    """
    n_frames = cube.shape[0]

    if n_frames>1000:
        ind_for_svd = range(0,n_frames,2)
    else:
        ind_for_svd = range(0,n_frames)

    ann_width = outrad-inrad
    cent_ann = inrad + int(np.round(ann_width/2.))
    ann_width += size_patch+2

    if expvar_mode=='annular':
        matrix_svd, _ = prepare_matrix(cube[ind_for_svd], 'temp-mean', None, 
                                       mode=expvar_mode, annulus_radius=cent_ann, 
                                       annulus_width=ann_width, verbose=False)
        U, S, V = svd_wrapper(matrix_svd, 'lapack',
                              min(matrix_svd.shape[0], matrix_svd.shape[1]),
                              False, False, True)
    elif expvar_mode=='fullfr':
        matrix_svd = prepare_matrix(cube[ind_for_svd], 'temp-mean', None, 
                                    mode=expvar_mode, verbose=False)
        U,S,V = svd_wrapper(matrix_svd, 'lapack', n_frames, False, False, True)

    exp_var = (S ** 2) / (S.shape[0] - 1)
    full_var = np.sum(exp_var)
    explained_variance_ratio = exp_var / full_var      # % of variance explained by each PC
    ratio_cumsum = np.cumsum(explained_variance_ratio)

    if k_list is not None:
        ratio_cumsum_klist = []
        for k in k_list:
            ratio_cumsum_klist.append(ratio_cumsum[k-1])
    
        if verbose:
            print("SVD on input matrix (annulus from cube)")
            print("  Number of PCs :\t")
            print("  ",k_list)
            print("  Cum. explained variance ratios :\t")
            print("  ",str(["{0:0.2f}".format(i) for i in ratio_cumsum_klist]).replace("'", ""), "\n")
    else:
        ratio_cumsum_klist = ratio_cumsum

    return ratio_cumsum, ratio_cumsum_klist



def frame_shift(array, shift_y, shift_x, interpolation='bicubic'):
    """ Shifts an 2d array by shift_y, shift_x.
    """
    if not array.ndim == 2:
        raise TypeError ('Input array is not a frame or 2d array')

    image = array.copy()

    if interpolation == 'bilinear':
        intp = cv2.INTER_LINEAR
    elif interpolation == 'bicubic':
        intp= cv2.INTER_CUBIC
    elif interpolation == 'nearneig':
        intp = cv2.INTER_NEAREST
    else:
        raise TypeError('Interpolation method not recognized.')

    image = np.float32(image)
    y, x = image.shape
    M = np.float32([[1,0,shift_x],[0,1,shift_y]])
    array_shifted = cv2.warpAffine(image, M, (x,y), flags=intp, 
                                   borderMode=cv2.BORDER_REPLICATE)

    return array_shifted


def inspect_patch_multik(model, cube, angle_list, k_list=[1,2,3,5,10,20,50,100],
                         inrad=10, outrad=14, size_patch=11, xy=(0,0),
                         scaling='temp-standard', collapse_func=np.mean,
                         normalize='slice', plot=True, dpi=70, psf=None):
    """
    """
    n_frames = cube.shape[0]
    frsize = int(cube.shape[1])

    if hasattr(model, 'base_estimator'):
        mode = 'rf'
    elif hasattr(model, 'name'):
        mode = 'nn'
    else:
        raise RuntimeError('Model not recognized')

    im_zeros = np.zeros_like(cube[0])
    im_zeros[xy[1],xy[0]] = 1

    cube_residuals = svd_decomp(cube, angle_list, size_patch, inrad, outrad, 
                                scaling, k_list, collapse_func, neg_ang=False)

    patches_cube = []

    y, x = np.where(im_zeros==1)

    patch = cube_crop_frames(np.array(cube_residuals), size_patch, 
                             xy=(int(x),int(y)), verbose=False)
    
    patch_reshaped = np.expand_dims(patch, 0)
    if normalize is not None:
        patch_reshaped = normalize_01(patch_reshaped, normalize)
    
    if mode=='nn':
        # adding extra dimension (channel) for keras model
        patch_reshaped = np.expand_dims(patch_reshaped, -1)
        proba = model.predict(patch_reshaped, verbose=0)    
        #prediction = model.predict_classes(patch_reshaped, verbose=0)
        #proba = model.predict_proba(patch_reshaped, verbose=0)
    
    elif mode=='rf':
        if psf is not None:
            patch_vector = create_feature_matrix(patch_reshaped, psf)
        else:
            # vectorizing the 3d samples to get a feature matrix
            patch_vector = patch_reshaped.flatten()
        proba = model.predict_proba(patch_vector)

    if plot:
        plots(np.squeeze(patch_reshaped), cmap='viridis', axis=False, dpi=dpi, 
              maxplots=np.squeeze(patch_reshaped).shape[0], colorb=False)
    print('Proba :', proba, '\n')

    #return patch, prediction, proba
    return patch, proba



def get_indices_annulus(shape, inrad, outrad, mask=None, maskrad=None, 
                        verbose=False):
    """ mask is a list of tuples X,Y
    """
    framemp = np.zeros(shape)
    if mask is not None:
        if not isinstance(mask, list):
            raise TypeError('Mask should be a list of tuples')
        if maskrad is None:
            raise ValueError('Fwhm not given')
        for xy in mask:
            # patch_size/2 diameter aperture
            cir = circle(xy[1], xy[0], maskrad, shape)  
            framemp[cir] = 1

    annulus_width = outrad - inrad
    cy, cx = frame_center(framemp)
    yy, xx = np.mgrid[:framemp.shape[0], :framemp.shape[1]]
    circ = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    donut_mask = (circ <= (inrad + annulus_width)) & (circ >= inrad)
    y, x = np.where(donut_mask)
    if mask is not None:
        npix = y.shape[0]
        ymask, xmask = np.where(framemp) # masked pixels where == 1
        inds = []
        for i, tup in enumerate(zip(y,x)):
            if tup in zip(ymask,xmask):  inds.append(i)
        y = np.delete(y, inds)
        x = np.delete(x, inds)

    if verbose:
        print(y.shape[0], 'pixels in annulus')
    return y, x




def create_synt_cube(cube, psf, ang, plsc, dist=None, theta=None, flux=None,
                     verbose=False):
    """
    """
    centy_fr, centx_fr = frame_center(cube[0])
    np.random.seed()
    if theta is None:
        theta = np.random.randint(0,360)

    posy = dist * np.sin(np.deg2rad(theta)) + centy_fr
    posx = dist * np.cos(np.deg2rad(theta)) + centx_fr
    if verbose:
        print('Theta:', theta)
        print('Flux_inj:', flux)
    cubefc = cube_inject_companions(cube, psf, ang, flevel=flux, plsc=plsc,
                                    rad_dists=[dist], n_branches=1, theta=theta,
                                    verbose=verbose)

    return cubefc, posx, posy




def create_feature_matrix(X, psf, mode='opencv'):
    """
    http://docs.opencv.org/2.4/modules/imgproc/doc/object_detection.html?highlight=matchtemplate#matchtemplate
    """
    psf_corr = frame_crop(psf, 7, verbose=False)
    psf_corr = psf_corr + np.abs(np.min(psf_corr))
    psf_corr = psf_corr / np.abs(np.max(psf_corr))
    
    n_samples = X.shape[0]
    n_features = X.shape[1]
    
    Xfeatmat = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        for j in range(n_features):
            if mode=='opencv':
                Xfeatmat[i,j] = cv2.matchTemplate(X[i][j].astype('float32'), 
                                                  psf_corr.astype('float32'), 
                                                  cv2.TM_CCOEFF_NORMED)
            elif mode=='skimage':
                Xfeatmat[i,j] = match_template(X[i][j], psf_corr)

    Xfeatmat_std = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        for j in range(n_features):
            Xfeatmat_std[i,j] = np.std(X[i][j])
        
    Xfeatmat = np.concatenate((Xfeatmat, Xfeatmat_std), axis=1)
    return Xfeatmat




