"""
Prediction procedures.
"""
from __future__ import print_function
from __future__ import absolute_import

__all__ = ['predict',
           'evaluate_annulus_multik',
           'generate_mlar_samples',
           'generate_mlar_prediction']

import numpy as np
from vip_hci.preproc import cube_derotate, cube_crop_frames, cube_derotate
from vip_hci.conf import time_ini, timing, time_fin
from vip_hci.var import pp_subplots as plots
from vip_hci.var import frame_center, dist
from vip_hci.stats import cube_stats_aperture, cube_stats_annulus
from matplotlib.pyplot import figure, show, Circle, subplots, show
from photutils import detect_sources, source_properties, properties_table
from skimage.feature import peak_local_max
from munch import *

from .sodinn_utils import (normalize_01, inspect_patch_multik, 
                            get_indices_annulus, svd_decomp, save_res,
                            create_feature_matrix)



def predict(model, cube, angle_list, fwhm=4, in_ann=1, out_ann=8, size_patch=7,
            k_list=[1,2,3,5,10,20,50], collapse_func=np.median, scaling=None,
            npix=1, normalize='slice', min_proba=0.9, min_separation=1, plot=True,
            plot_patches=False, debug=False, raw_stats=False, verbose=True, 
            dpi=150, save_plot=False, psf=None, lr_mode='eigen', save=None):
    """
    With multiple models, the parameters are shared: normalize, klist, scaling
    """
    starttime = time_ini(verbose=verbose)

    n_annuli = out_ann - in_ann
    n_frames = cube.shape[0]
    frsize = int(cube.shape[1])
    half_frsize = np.floor(frsize/2.)

    if verbose: print('N annuli: {}'.format(n_annuli))

    if out_ann*fwhm > half_frsize:
        out_ann = half_frsize/fwhm

    results = Munch(minprob=min_proba, klist=k_list, probmaps=[], binmaps=[],
                    nsources=[])

    # In case of more than one model
    if isinstance(model, list):
        n_models = len(model)
    else:
        n_models = 1

    # Computing the MLAR samples once, even with multiple models to evaluate
    framesprob = []
    framesprob2 = []
    for ann in range(in_ann, out_ann):
        probone, probzer, _ = evaluate_annulus_multik(model, cube, angle_list,
                                        fwhm*ann, (fwhm*ann)+fwhm,
                                        size_patch, k_list, collapse_func,
                                        scaling, npix, normalize, min_proba,
                                        debug, False, verbose=verbose,
                                        psf=psf, lr_mode=lr_mode)
        framesprob.append(probone)
        framesprob2.append(probzer)

    # Computing the full-frame prob maps for each model
    for i in range(n_models):
        if isinstance(model, list):
            final_prob = np.array(np.array(framesprob)[:,i]).max(axis=0)
            final_prob_zero = np.array(np.array(framesprob2)[:,i]).max(axis=0)
        else:
            final_prob = np.array(framesprob).max(axis=0)
            final_prob_zero = np.array(framesprob2).max(axis=0)

        first_segm = detect_sources(final_prob, min_proba, npix)
        binary_map = peak_local_max(first_segm.data, min_distance=min_separation,
                                    indices=False)
        # from scipy.ndimage import label
        #final_segm, n_sources = label(binmap, structure=[[1,1,1],[1,1,1],[1,1,1]])
        final_segm = detect_sources(binary_map, 0.5, npix)
        n_sources = final_segm.nlabels

        props = source_properties(final_prob, final_segm)
        xx = [int(props[s].maxval_xpos.value) for s in range(len(props))]
        yy = [int(props[s].maxval_ypos.value) for s in range(len(props))]

        if plot:
            if save_plot:
                plots(final_prob, grid=False, dpi=dpi, vmin=0, vmax=1,
                      label=['Probability map'], cmap='viridis', labelpad=8,
                      labelsize=10, save='pred_probmap.pdf', axis=False,
                      showcent=True)
                plots(binary_map, grid=False, dpi=dpi, colorb=False,
                      label=['Binary map'], cmap='bone', axis=False,
                      labelpad=8, labelsize=10, save='pred_binmap.pdf',
                      showcent=True)
            else:
                if n_sources>0:
                    plots(final_prob, first_segm.data, binary_map, dpi=dpi, vmin=[0,0,0],
                        vmax=[1,n_sources,1], grid=False, labelpad=8, labelsize=10,
                        label=['Probability map', 'Segmentation map', 'Binary map'],
                        showcent=True, cmap=['viridis','tab20b','bone'],
                        circle=[xy for xy in zip(xx,yy)], circlerad=9)
                else:
                    plots(final_prob, first_segm.data, binary_map, dpi=dpi, vmin=[0,0,0],
                        vmax=[1,n_sources,1], grid=False, labelpad=8, labelsize=10,
                        label=['Probability map', 'Segmentation map', 'Binary map'],
                        showcent=True, cmap=['viridis','tab20b','bone'])

        if verbose: print('MODEL {} found {} sources \n'.format(i+1, n_sources))

        if plot_patches:
            yc, xc = frame_center(cube[0])
            for i in range(len(xx)):
                di = dist(yc, xc, yy[i], xx[i])
                print('X,Y=({},{}), R={:.2f} pxs'.format(xx[i], yy[i], di))
                print('Value in final probmap: {:.4f}'.format(final_prob[yy[i],xx[i]]))
                inrad = (di//fwhm)*fwhm
                outrad = inrad+fwhm
                cube_res = svd_decomp(cube, angle_list, size_patch, inrad,
                                      outrad, scaling, k_list, collapse_func,
                                      neg_ang=False)
                patch = cube_crop_frames(np.array(cube_res), size_patch,
                                         xy=(xx[i],yy[i]), verbose=False)
                plots(patch, cmap='viridis', axis=False, dpi=100, horsp=0.1,
                      maxplots=patch.shape[0], colorb=False)

                if raw_stats:
                    _ = cube_stats_aperture(cube_derotate(cube, angle_list),
                                                      int(np.round(fwhm)),
                                                      xy=(xx[i],yy[i]), plot=True)
                    show()

        results.probmaps.append(final_prob)
        results.binmaps.append(binary_map)
        results.nsources.append(n_sources)


    if verbose: timing(starttime)
    fintime = time_fin(starttime)
    results.runtimes = fintime

    if save is not None and isinstance(save, str):
        save_res(save+'.p', results)
        print("Saved file: "+save+'.p')
        timing(starttime)

    return results



def evaluate_annulus_multik(model, cube, angle_list, inrad=10, outrad=14,
                                 size_patch=7, k_list=[1,2,3,5,10,20,50], 
                                 collapse_func=np.median, scaling=None, npix=4, 
                                 normalize='slice', min_proba=0.9, plot=True,
                                 plot_patches=True, dpi=120, verbose=True,
                                 psf=None, lr_mode='eigen'):
    """
    """
    n_frames = cube.shape[0]
    frsize = int(cube.shape[1])
    yc, xc = frame_center(cube[0])
    # grabbing half of the frames and cropping images
    if frsize > outrad+outrad+size_patch+2:
        wing = int((frsize - (outrad+outrad+size_patch+2))/2)
        frsize = outrad+outrad+size_patch+2
    else:
        wing = None
    yy, xx = get_indices_annulus((frsize, frsize), inrad, outrad, verbose=False)
    num_patches = yy.shape[0]

    patches_frame = generate_mlar_samples(cube, angle_list, inrad=inrad,
                                          outrad=outrad, size_patch=size_patch,
                                          k_list=k_list, scaling=scaling,
                                          collapse_func=collapse_func,
                                          normalize=normalize, lr_mode=lr_mode,
                                          verbose=verbose)

    if isinstance(model, list):
        n_models = len(model)
        fr_probas = []
        fr_probas_ = []
        segm = []
        for i in range(n_models):
            res = generate_mlar_prediction(model[i], cube, angle_list,
                                           patches_frame, wing, yc, xc, frsize,
                                           yy, xx, num_patches, inrad, outrad,
                                           size_patch, k_list, collapse_func,
                                           scaling, npix, normalize[i], min_proba,
                                           plot, plot_patches, dpi, psf)
            fr_probas.append(res[0])
            fr_probas_.append(res[1])
            segm.append(res[2])

    else:
        fr_probas, fr_probas_, segm = generate_mlar_prediction(model, cube,
                                              angle_list, patches_frame, wing,
                                              yc, xc, frsize, yy, xx, num_patches,
                                              inrad, outrad, size_patch, k_list,
                                              collapse_func, scaling, npix,
                                              normalize, min_proba, plot,
                                              plot_patches, dpi, psf)

    return fr_probas, fr_probas_, segm



def generate_mlar_samples(cube, angle_list, inrad=10, outrad=14,
                          size_patch=7, k_list=[1,2,3,5,10,20,50],
                          collapse_func=np.median, scaling=None,
                          normalize='slice', lr_mode='eigen', verbose=True):
    """ """
    frsize = int(cube.shape[1])
    # grabbing half of the frames and cropping images
    if frsize > outrad+outrad+size_patch+2:
        cube_orig = cube.copy()
        wing = int((frsize - (outrad+outrad+size_patch+2))/2)
        frsize = outrad+outrad+size_patch+2
        cube = cube_crop_frames(cube, frsize, verbose=False)
    else:
        wing = None

    cube_residuals = svd_decomp(cube, angle_list, size_patch, inrad, outrad,
                                scaling, k_list, collapse_func, lr_mode=lr_mode,
                                neg_ang=False)

    yy, xx = get_indices_annulus((frsize,frsize), inrad, outrad, verbose=False)
    num_patches = yy.shape[0]
    if verbose:
        print("Ann {} to {} pxs: {} patches".format(inrad,outrad,num_patches))

    patches_frame = []
    for i in range(num_patches):
        patches_frame.append(cube_crop_frames(np.array(cube_residuals), size_patch,
                                              xy=(int(xx[i]), int(yy[i])),
                                              verbose=False))
    patches_frame = np.array(patches_frame)
    if normalize is not None:
        patches_frame = normalize_01(patches_frame, normalize)

    return patches_frame



def generate_mlar_prediction(model, cube, angle_list, patches_frame, wing,
                             yc, xc, frsize, yy, xx, num_patches, inrad=10,
                             outrad=14, size_patch=7, k_list=[1,2,3,5,10,20,50],
                             collapse_func=np.median, scaling=None, npix=4,
                             normalize='slice', min_proba=0.9, plot=True,
                             plot_patches=True, dpi=120, psf=None):
    """ """
    if hasattr(model, 'base_estimator'):
        mode = 'rf'
    elif hasattr(model, 'name'):
        mode = 'nn'
    else:
        raise RuntimeError('Model not recognized')

    if mode=='nn':
        # adding extra dimension (channel) for TF model
        patches = np.expand_dims(patches_frame, -1)
        probas = model.predict(patches, verbose=0)
        # when trained on several GPUs
        if model.output_names[0].startswith('merge'):
            if model.layers[-2].output_shape == (None, 1):  # sigmoid case
                frame_probas_one = np.zeros((frsize, frsize))
                frame_probas_one[yy, xx] = probas[:].flatten()
        else:
            if model.layers[-1].activation.__name__=='sigmoid':
                frame_probas_one = np.zeros((frsize, frsize))
                frame_probas_one[yy, xx] = probas[:].flatten()
            elif model.layers[-1].activation.__name__=='softmax':
                frame_probas_one = np.zeros((frsize, frsize))
                frame_probas_zero = np.zeros((frsize, frsize))
                frame_probas_one[yy, xx] = probas[:,1]
                frame_probas_zero[yy, xx] = probas[:,0]

    elif mode=='rf':
        if psf is not None:
            # creating the feature matrix (cross-corr and std)
            patches = create_feature_matrix(patches_frame, psf)
        else:
            # vectorizing the 3d samples to get a feature matrix
            patches = patches_frame.reshape(num_patches, -1)

        probas = model.predict_proba(patches)
        frame_probas_one = np.zeros((frsize, frsize))
        frame_probas_zero = np.zeros((frsize, frsize))
        frame_probas_one[yy, xx] = probas[:,1]
        frame_probas_zero[yy, xx] = probas[:,0]

    if wing is not None:
        frame_probas_pad = np.pad(frame_probas_one, wing, 'constant', 
                                  constant_values=0)
        if mode=='nn':
            if not model.output_names[0].startswith('merge'):
                if model.layers[-1].activation.__name__=='softmax':
                    frame_probas_pad2 = np.pad(frame_probas_zero, wing,
                                               'constant', constant_values=0)
        segm_pad = detect_sources(frame_probas_pad, min_proba, npix)
        if plot:
            plots(frame_probas_pad, frame_probas_one, cmap='bone',
                  title='Probability map  |  Probability map zoom', dpi=dpi)

    else:
        segm = detect_sources(frame_probas_one, min_proba, npix)    
        if plot:
            plots(frame_probas_one, title='Probability map ', dpi=dpi)

    if plot_patches:
        if wing is not None:
            props = source_properties(frame_probas_pad, segm_pad)
        else:
            props = source_properties(frame_probas_one, segm) 
        xx = [int(props[i].maxval_xpos.value) for i in range(len(props))]
        yy = [int(props[i].maxval_ypos.value) for i in range(len(props))]

        for i in range(len(xx)):
            print()
            print('X,Y=({},{}), R={:.2f} pxs'.format(xx[i], yy[i], 
                                                     dist(yc, xc, yy[i], xx[i])))
            _ = inspect_patch_multik(model, cube, angle_list, k_list, inrad,
                                     outrad, size_patch, (xx[i],yy[i]), scaling,
                                     collapse_func, normalize, True, psf=psf)

    if wing is not None:
        if mode=='nn':
            if model.output_names[0].startswith('merge'):
                # only sigmoid case
                return frame_probas_pad, np.zeros_like(frame_probas_pad), segm_pad.data
            else:
                if model.layers[-1].activation.__name__=='softmax':
                    return frame_probas_pad, frame_probas_pad2, segm_pad.data
                else:
                    return frame_probas_pad, np.zeros_like(frame_probas_pad), segm_pad.data
    else:
        if mode=='nn':
            if model.output_names[0].startswith('merge'):
                # only sigmoid case
                return frame_probas_one, np.zeros_like(frame_probas_one), segm.data
            else:
                if model.layers[-1].activation.__name__=='softmax':
                    return frame_probas_one, frame_probas_zero, segm.data
                else:
                    return frame_probas_one, np.zeros_like(frame_probas_one), segm.data

