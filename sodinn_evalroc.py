"""
Assessing performance. ROC curves generation.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

__all__ = ['compute_tpr_fps',
           'inject_and_postprocess',
           'compute_binary_map',
           'plot_roc_curves',
           'plot_detmaps']


import numpy as np
from skimage.draw import circle
from vip_hci.phot import (noise_per_annulus, cube_inject_companions,
                          frame_quick_report, snrmap_fast)
from vip_hci.conf import time_ini, timing, time_fin
from vip_hci.conf.utils_conf import eval_func_tuple as futup
from vip_hci.var import pp_subplots as plots
from vip_hci.var import frame_center
from vip_hci.pca import pca, svd_wrapper
from vip_hci.var import reshape_matrix, prepare_matrix
from vip_hci.llsg import llsg
from vip_hci.madi import adi
from scipy.stats import skewnorm
from matplotlib.pyplot import (figure, plot, grid, xlabel, ylabel, legend, xlim,
                               ylim, savefig)
from photutils import detect_sources
from skimage.feature import peak_local_max

from multiprocessing import Pool, cpu_count
import itertools as itt
from munch import *
from .sodinn_utils import (create_synt_cube, get_indices_annulus, save_res,
                          load_res, get_cumexpvar, svd_decomp)
from .sodinn_predict import predict


def inject_and_postprocess(cube, angle_list, psf, fwhm=4, plsc=0.0272,
                           n_samples=100, inrad=8, outrad=12,
                           dist_flux='uniform', dist_flux_p1=2,
                           dist_flux_p2=500, dist_flux_p3=5, mask=None,
                           cadi=True, pca_dict={'ncomp':None},
                           llsg_dict={'rank':None, 'thresh':1},
                           sodirf_dict={'model':None, 'k_list':None,
                                        'collapse_func':np.median,
                                        'scaling':None, 'normalize':'slice'},
                           sodinn_dict={'model':None, 'k_list':None,
                                        'collapse_func':np.median,
                                        'scaling':None, 'normalize':'slice'},
                           lr_mode='eigen' ,cevr=0.90, expvar_mode='annular',
                           size_patch=7, nproc=None, save=None):
    """ Injections and post-processing up to the generation of detection maps.

    size_patch=7, k_list=range(1, 17), collapse_func=np.median, scaling=None,
    normalize='slice', mask=None,

    expvar : In case PCA or LLSG are used. Number of components for which the
             CEVR is superior to the desired threshold.
    """
    starttime = time_ini()

    n_frames = cube.shape[0]
    frsize = int(cube.shape[1])
    half_frsize = int(frsize / 2.)

    if cevr is not None:
        ratio_cumsum, ratio_cumsum_klist = get_cumexpvar(cube, expvar_mode,
                                                         inrad, outrad,
                                                         size_patch, None,
                                                         verbose=False)
        optpcs = np.searchsorted(ratio_cumsum, cevr) + 1
        print(cevr, '% of CEVR with', optpcs, 'PCs')

    # Getting indices in annulus, taking into account the mask
    yy, xx = get_indices_annulus((frsize, frsize), inrad, outrad, mask=mask,
                                 verbose=False)
    num_patches = yy.shape[0]

    # Defining Fluxes according to chosen distribution
    if dist_flux == 'skewnormal':
        mean_flux = dist_flux_p1
        std_flux = dist_flux_p2
        fluxes = skewnorm.rvs(dist_flux_p3, loc=mean_flux, scale=std_flux,
                              size=n_samples)
    elif dist_flux == 'normal':
        mean_flux = dist_flux_p1
        std_flux = dist_flux_p2
        fluxes = np.random.normal(mean_flux, std_flux, size=n_samples)
    elif dist_flux == 'uniform':
        if not dist_flux_p2 > dist_flux_p1:
            err_msg = 'dist_flux_p2 must be larger than dist_flux_p1 when '
            err_msg += 'dist_flux==uniform'
            raise ValueError(err_msg)
        fluxes = np.random.uniform(dist_flux_p1, dist_flux_p2, size=n_samples)
    else:
        raise ValueError('Flux distribution not recognized')
    fluxes = np.sort(fluxes)

    inds_inj = np.random.randint(0, num_patches, size=n_samples)

    dists = []; thetas = []
    for m in range(n_samples):
        injx = xx[inds_inj[m]]
        injy = yy[inds_inj[m]]
        injx -= frame_center(cube[0])[1]
        injy -= frame_center(cube[0])[0]
        dist = np.sqrt(injx ** 2 + injy ** 2)
        theta = np.mod(np.arctan2(injy, injx) / np.pi * 180, 360)
        dists.append(dist)
        thetas.append(theta)

    if not nproc: nproc = int((cpu_count()))

    list_xy = []

    if cadi is not None:
        list_adi_frames = []; list_adi_probmaps = []
    if pca_dict is not None:
        list_pca_frames = []; list_pca_probmaps = []
    if llsg_dict is not None:
        list_llsg_frames = []; list_llsg_probmaps = []
    if sodirf_dict is not None:
        list_dnn_probmaps = []
    if sodinn_dict is not None:
        list_rf_probmaps = []

    print('Injections:')
    for m in range(n_samples):
        if (m+1)%10==0:
            if (m+1)%40==0:
                print(m+1)
            else:
                print(m+1, end=' ')
        else:
            print('.', end=' ')
        cufc, cox, coy = create_synt_cube(cube, psf, angle_list, plsc,
                                          theta=thetas[m], flux=fluxes[m],
                                          dist=dists[m], verbose=False)
        cox = int(np.round(cox))
        coy = int(np.round(coy))
        list_xy.append((cox, coy))

        if cadi is not None:
            frame_adi = adi(cufc, angle_list, verbose=False)
            snrm_adi = snrmap_fast(frame_adi, fwhm, nproc=nproc, verbose=False)
            list_adi_frames.append(frame_adi)
            list_adi_probmaps.append(snrm_adi)

        if pca_dict is not None:
            if pca_dict['ncomp'] is None and cevr is not None:
                ncomp = optpcs
            else:
                ncomp = pca_dict['ncomp']
            frame_pca = pca(cufc, angle_list, ncomp=ncomp, verbose=False)
            snrm_pca = snrmap_fast(frame_pca, fwhm, nproc=nproc, verbose=False)
            list_pca_frames.append(frame_pca)
            list_pca_probmaps.append(snrm_pca)

        if llsg_dict is not None:
            if llsg_dict['rank'] is None and cevr is not None:
                rank = optpcs
            else:
                rank = llsg_dict['rank']
            frame_llsg = llsg(cufc, angle_list, fwhm=fwhm, rank=rank,
                              thresh=llsg_dict['thresh'], verbose=False, nproc=1)
            snrm_llsg = snrmap_fast(frame_llsg, fwhm, nproc=nproc, verbose=False)
            list_llsg_frames.append(frame_llsg)
            list_llsg_probmaps.append(snrm_llsg)

        # Only sodirf
        if sodirf_dict['model'] is not None and sodinn_dict['model'] is None:
            resrf = predict(sodirf_dict['model'], cufc, angle_list, fwhm, in_ann=1,
                           out_ann=int((half_frsize / fwhm) - 1), plot=False,
                           size_patch=size_patch, k_list=sodirf_dict['k_list'],
                           collapse_func=sodirf_dict['collapse_func'],
                           scaling=sodirf_dict['scaling'], npix=1, min_proba=0.9,
                           normalize=sodirf_dict['normalize'], lr_mode=lr_mode,
                           verbose=False)
            prob_map_rf = resrf.probmaps[0]
            list_rf_probmaps.append(prob_map_rf)
        # Only sodinn
        elif sodirf_dict['model'] is None and sodinn_dict['model'] is not None:
            resnn = predict(sodinn_dict['model'], cufc, angle_list, fwhm, in_ann=1,
                           out_ann=int((half_frsize / fwhm) - 1), plot=False,
                           size_patch=size_patch, k_list=sodinn_dict['k_list'],
                           collapse_func=sodinn_dict['collapse_func'],
                           scaling=sodinn_dict['scaling'], npix=1, min_proba=0.9,
                           normalize=sodinn_dict['normalize'], lr_mode=lr_mode,
                           verbose=False)
            prob_map = resnn.probmaps[0]
            list_dnn_probmaps.append(prob_map)
        # Both sodinn and sodirf
        elif sodirf_dict['model'] is not None and sodinn_dict['model'] is not None:
            res = predict([sodirf_dict['model'], sodinn_dict['model']],
                          cufc, angle_list, fwhm, in_ann=1,
                          out_ann=int((half_frsize / fwhm) - 1), plot=False,
                          size_patch=size_patch, k_list=sodinn_dict['k_list'],
                          collapse_func=sodinn_dict['collapse_func'],
                          scaling=sodinn_dict['scaling'], npix=1,
                          min_proba=0.9, normalize=sodinn_dict['normalize'],
                          lr_mode=lr_mode, verbose=False)
            list_rf_probmaps.append(res.probmaps[0])
            list_dnn_probmaps.append(res.probmaps[1])

    fintime = time_fin(starttime)
    results = Munch(nsamp=n_samples, patch_size=size_patch,
                    inrad=inrad, outrad=outrad, flux_distribution=dist_flux,
                    fluxp1=dist_flux_p1, fluxp2=dist_flux_p2,
                    fluxp3=dist_flux_p3, fwhm=fwhm, plsc=plsc, filename=save,
                    list_xy=list_xy, fluxes=fluxes, dists=dists, thetas=thetas,
                    cevr=cevr, optpcs=optpcs, runtime=fintime,
                    cadi_frames=list_adi_frames,
                    cadi_probmaps=list_adi_probmaps,
                    pca_frames=list_pca_frames,
                    pca_probmaps=list_pca_probmaps,
                    llsg_frames=list_llsg_frames,
                    llsg_probmaps=list_llsg_probmaps,
                    sodirf_probmaps=list_rf_probmaps,
                    sodinn_probmaps=list_dnn_probmaps,
                    pca_ncomp=pca_dict['ncomp'], llsg_rank=llsg_dict['rank'],
                    llsg_thresh=llsg_dict['thresh'],
                    sodirf_klist=sodirf_dict['k_list'],
                    sodirf_normalize=sodirf_dict['normalize'],
                    sodinn_klist=sodinn_dict['k_list'],
                    sodinn_normalize=sodinn_dict['normalize'])

    if save is not None and isinstance(save, str):
        save_res('Roc_injections_' + save + '.p', results)

    print()
    timing(starttime)
    return results


def compute_tpr_fps(roc_injections, npix=1, save=None, min_distance=1,
                    adi_thresholds=[0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
                    llsg_thresholds=[0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
                    pca_thresholds=[0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
                    sodinn_thresholds=np.linspace(0.1, 0.99, 10).tolist(),
                    sodirf_thresholds=np.linspace(0.1, 0.99, 10).tolist()):
    """
    """
    starttime = time_ini()
    xy = roc_injections.list_xy

    if hasattr(roc_injections, 'cadi_probmaps'):
        snrm_adi = roc_injections.cadi_probmaps
        list_adi_bmaps = [];
        list_adi_fps = [];
        list_adi_detection = []
    if hasattr(roc_injections, 'pca_probmaps'):
        snrm_pca = roc_injections.pca_probmaps
        list_pca_bmaps = []
        list_pca_fps = []
        list_pca_detection = []
    if hasattr(roc_injections, 'llsg_probmaps'):
        snrm_llsg = roc_injections.llsg_probmaps
        list_llsg_bmaps = []
        list_llsg_fps = []
        list_llsg_detection = []
    if hasattr(roc_injections, 'sodirf_probmaps'):
        probmap_sodirf = roc_injections.sodirf_probmaps
        list_rf_bmaps = []
        list_rf_fps = []
        list_rf_detection = []
    if hasattr(roc_injections, 'sodinn_probmaps'):
        probmap_sodinn = roc_injections.sodinn_probmaps
        list_dnn_bmaps = []
        list_dnn_fps = []
        list_dnn_detection = []

    # Getting the binary maps
    print('Evaluating injections:')
    for i in range(roc_injections.nsamp):
        if (i+1)%10==0:
            if (i+1)%40==0:
                print(i+1)
            else:
                print(i+1, end=' ')
        else:
            print('.', end=' ')
        x, y = xy[i][0], xy[i][1]

        if hasattr(roc_injections, 'cadi_probmaps'):
            resadi = compute_binary_map(snrm_adi[i], adi_thresholds, x, y,
                                        npix=npix, min_distance=min_distance)
            list_adi_detection.append(resadi[0])
            list_adi_fps.append(resadi[1])
            list_adi_bmaps.append(resadi[2])

        if hasattr(roc_injections, 'pca_probmaps'):
            respca = compute_binary_map(snrm_pca[i], pca_thresholds, x, y,
                                        npix=npix, min_distance=min_distance)
            list_pca_detection.append(respca[0])
            list_pca_fps.append(respca[1])
            list_pca_bmaps.append(respca[2])

        if hasattr(roc_injections, 'llsg_probmaps'):
            resllsg = compute_binary_map(snrm_llsg[i], llsg_thresholds, x, y,
                                         npix=npix, min_distance=min_distance)
            list_llsg_detection.append(resllsg[0])
            list_llsg_fps.append(resllsg[1])
            list_llsg_bmaps.append(resllsg[2])

        if hasattr(roc_injections, 'sodirf_probmaps'):
            resrf = compute_binary_map(probmap_sodirf[i], sodirf_thresholds, x,
                                       y, npix=npix, min_distance=min_distance)
            list_rf_detection.append(resrf[0])
            list_rf_fps.append(resrf[1])
            list_rf_bmaps.append(resrf[2])

        if hasattr(roc_injections, 'sodinn_probmaps'):
            resdnn = compute_binary_map(probmap_sodinn[i], sodinn_thresholds, x,
                                        y, npix=npix, min_distance=min_distance)
            list_dnn_detection.append(resdnn[0])
            list_dnn_fps.append(resdnn[1])
            list_dnn_bmaps.append(resdnn[2])

    fintime = time_fin(starttime)
    results = Munch(filename_injections=roc_injections.filename,
                    filename=save, runtime=fintime,
                    cadi_thresholds=adi_thresholds,
                    cadi_bmaps=list_adi_bmaps,
                    cadi_fps=list_adi_fps,
                    cadi_detections=list_adi_detection,
                    pca_thresholds=pca_thresholds,
                    pca_bmaps=list_pca_bmaps,
                    pca_fps=list_pca_fps,
                    pca_detections=list_pca_detection,
                    llsg_thresholds=llsg_thresholds,
                    llsg_bmaps=list_llsg_bmaps,
                    llsg_fps=list_llsg_fps,
                    llsg_detections=list_llsg_detection,
                    sodirf_thresholds=sodirf_thresholds,
                    sodirf_bmaps=list_rf_bmaps,
                    sodirf_fps=list_rf_fps,
                    sodirf_detections=list_rf_detection,
                    sodinn_thresholds=sodinn_thresholds,
                    sodinn_bmaps=list_dnn_bmaps,
                    sodinn_fps=list_dnn_fps,
                    sodinn_detections=list_dnn_detection)

    if save is not None and isinstance(save, str):
        save_res('Roc_tprfps_' + save + '.p', results)
    print()
    timing(starttime)

    return results


def compute_binary_map(frame, thresholds, injx, injy, npix=1, min_distance=1,
                       debug=False):
    """
    min_distance=fwhm is problematic, when two blobs are less than fwhm apart
    the true blob could be discarded, depending on the ordering.
    """
    list_binmaps = []
    list_detections = []
    list_fps = []

    for threshold in thresholds:
        first_segm = detect_sources(frame, threshold, npix)
        binary_map = peak_local_max(first_segm.data, min_distance=min_distance,
                                    indices=False)
        final_segm = detect_sources(binary_map, 0.5, npix)
        n_sources = final_segm.nlabels
        if debug: plots(first_segm.data, binary_map, final_segm.data)

        ### Checking if the injection pxs match with detected blobs
        detection = 0
        for i in range(n_sources):
            yy, xx = np.where(final_segm.data==i+1)
            injyy, injxx = circle(injy, injx, 2)
            coords_ind = zip(injxx, injyy)
            for j in range(len(coords_ind)):
                if coords_ind[j] in zip(xx, yy):
                    detection = 1
                    fps = n_sources - 1
                    break

        if detection==0: fps = n_sources

        list_detections.append(detection)
        list_binmaps.append(binary_map)
        list_fps.append(fps)

    return list_detections, list_fps, list_binmaps
    

def plot_detmaps(roc_injections, roc_tprfps, i=30, thr=9, dpi=100,
                 show_axis=True, show_grid=False, minval=-10, maxval='max',
                 plot_type=1):
    """
    i - sample or iteration : 0-nsamp
    thr - threshold : 0-9

    plot_type :
        1 - One row per algorithm (frame, probmap, binmap)
        2 - 1 row for final frames, 1 row for probmaps and 1 row for binmaps
    """
    if isinstance(roc_injections, str):
        ROC1 = load_res(roc_injections)
    else:
        ROC1 = roc_injections

    if isinstance(roc_tprfps, str):
        ROC2 = load_res(roc_tprfps)
    else:
        ROC2 = roc_tprfps

    print('X,Y: {}'.format(ROC1.list_xy[i]))
    print('Dist: {}, Flux: {} \n'.format(ROC1.dists[i], ROC1.fluxes[i]))

    if plot_type == 1:
        msg = 'Detection state: {} | False postives: {}'
        print(msg.format(ROC2.cadi_detections[i][thr],ROC2.cadi_fps[i][thr]))
        thresh = str(ROC2.cadi_thresholds[thr])
        plots(ROC1.cadi_frames[i], ROC1.cadi_probmaps[i], ROC2.cadi_bmaps[i][thr],
              label=['CADI frame','CADI S/Nmap', 'Thresholded at '+thresh],
              dpi=dpi, horsp=0.2, axis=show_axis, grid=show_grid,
              cmap=['viridis','viridis','gray'])
            
        msg = 'Detection state: {} | False postives: {}'
        print(msg.format(ROC2.pca_detections[i][thr],ROC2.pca_fps[i][thr]))
        thresh = str(ROC2.pca_thresholds[thr])
        plots(ROC1.pca_frames[i], ROC1.pca_probmaps[i], ROC2.pca_bmaps[i][thr],
              label=['PCA frame','PCA S/Nmap', 'Thresholded at '+thresh],
              dpi=dpi, horsp=0.2, axis=show_axis, grid=show_grid,
              cmap=['viridis', 'viridis', 'gray'])

        msg = 'Detection state: {} | False postives: {}'
        print(msg.format(ROC2.llsg_detections[i][thr],ROC2.llsg_fps[i][thr]))
        thresh = str(ROC2.llsg_thresholds[thr])
        plots(ROC1.llsg_frames[i], ROC1.llsg_probmaps[i], ROC2.llsg_bmaps[i][thr],
              label=['LLSG frame','LLSG S/Nmap', 'Thresholded at '+thresh],
              dpi=dpi, horsp=0.2, axis=show_axis, grid=show_grid,
              cmap=['viridis','viridis','gray'])

        msg = 'Detection state: {} | False postives: {}'
        print(msg.format(ROC2.sodirf_detections[i][thr],ROC2.sodirf_fps[i][thr]))
        thresh = str(ROC2.sodirf_thresholds[thr])
        plots(ROC1.sodirf_probmaps[i], ROC2.sodirf_bmaps[i][thr], dpi=dpi,
              label=['SODIRF probability','Thresholded at '+thresh],
              axis=show_axis, grid=show_grid, cmap=['viridis','gray'])
        
        msg = 'Detection state: {} | False postives: {}'
        print(msg.format(ROC2.sodinn_detections[i][thr],ROC2.sodinn_fps[i][thr]))
        thresh = str(ROC2.sodinn_thresholds[thr])
        plots(ROC1.sodinn_probmaps[i], ROC2.sodinn_bmaps[i][thr], dpi=dpi,
              label=['SODINN probability','Thresholded at '+thresh],
              axis=show_axis, grid=show_grid, cmap=['viridis','gray'])
    
    elif plot_type==2:
        if isinstance(maxval, str) and maxval=='max':
            maxval = np.concatenate([ROC1.cadi_frames[i], ROC1.pca_frames[i],
                                     ROC1.llsg_frames[i]]).max()/2
        plots(ROC1.cadi_frames[i], ROC1.pca_frames[i], ROC1.llsg_frames[i], dpi=dpi,
              label=['CADI frame','PCA frame','LLSG frame'], vmax=maxval,
              vmin=minval, axis=show_axis, grid=True, cmap='viridis')
        
        plots(ROC1.cadi_probmaps[i], ROC1.pca_probmaps[i], ROC1.llsg_probmaps[i],
              ROC1.sodirf_probmaps[i], ROC1.sodinn_probmaps[i], dpi=dpi,
              label=['CADI S/Nmap','PCA S/Nmap','LLSG S/Nmap','SODIRF probmap',
                     'SODINN probmap'], axis=show_axis, grid=True, cmap='viridis')

        print('CADI det :',ROC2.cadi_detections[i][thr], ' CADI FPs:',ROC2.cadi_fps[i][thr])
        print('PCA det:',ROC2.pca_detections[i][thr], ' PCA FPs:',ROC2.pca_fps[i][thr])
        print('LLSG det:',ROC2.llsg_detections[i][thr], ' LLSG FPs:',ROC2.llsg_fps[i][thr])
        print('SODIRF det:',ROC2.sodirf_detections[i][thr], ' SODIRF FPs:',ROC2.sodirf_fps[i][thr])
        print('SODINN det:',ROC2.sodinn_detections[i][thr], ' SODINN FPs:',ROC2.sodinn_fps[i][thr])
        thresh1 = str(ROC2.cadi_thresholds[thr])
        thresh2 = str(ROC2.pca_thresholds[thr])
        thresh3 = str(ROC2.llsg_thresholds[thr])
        thresh4 = str(ROC2.sodirf_thresholds[thr])[:4]
        thresh5 = str(ROC2.sodinn_thresholds[thr])[:4]
        plots(ROC2.cadi_bmaps[i][thr], ROC2.pca_bmaps[i][thr],
              ROC2.llsg_bmaps[i][thr], ROC2.sodirf_bmaps[i][thr],
              ROC2.sodinn_bmaps[i][thr], dpi=dpi,
              label=['Thresholded at '+thresh1, 'Thresholded at '+thresh2,
                     'Thresholded at '+thresh3, 'Thresholded at '+thresh4,
                     'Thresholded at '+thresh5], axis=show_axis, grid=True,
                     colorb=False, cmap='bone')



def plot_roc_curves(roc_injections, roc_tprfps, dpi=100, figsize=(5, 5),
                    markersize=3, line_alpha=0.2, marker_alpha=0.5,
                    xmin=None, xmax=None, ymin=0, ymax=1.02, xlog=True,
                    show_data_labels=True, label_rotation=0,
                    label_fontsize=5.5, label_xgap=0, label_ygap=-0.028,
                    label_skip_one=False, label_weight='bold', label_alpha=1,
                    yaxis_label=True, xaxis_label=True, legend_loc='lower right',
                    legend_size=6, hide_overlap_label=True, verbose=True,
                    save_plot=False):
    """
    """
    if isinstance(roc_injections, str):
        ROC1 = load_res(roc_injections)
    else:
        ROC1 = roc_injections

    if isinstance(roc_tprfps, str):
        ROC2 = load_res(roc_tprfps)
    else:
        ROC2 = roc_tprfps

    n_thresholds = len(ROC2.sodinn_thresholds)
    n_injections = ROC1.nsamp

    if verbose:
        print('{} injections'.format(ROC1.nsamp))
        print('Flux distro : {} [{}:{}]'.format(ROC1.flux_distribution,
                                                ROC1.fluxp1, ROC1.fluxp2))
        print('Annulus from {} to {} pxs'.format(ROC1.inrad, ROC1.outrad))

    adi_detections = np.array(ROC2.cadi_detections)
    adi_fps = np.array(ROC2.cadi_fps)
    adi_tpr = np.zeros((n_thresholds))
    adi_mean_fps = np.zeros((n_thresholds))

    pca_detections = np.array(ROC2.pca_detections)
    pca_fps = np.array(ROC2.pca_fps)
    pca_tpr = np.zeros((n_thresholds))
    pca_mean_fps = np.zeros((n_thresholds))

    llsg_detections = np.array(ROC2.llsg_detections)
    llsg_fps = np.array(ROC2.llsg_fps)
    llsg_tpr = np.zeros((n_thresholds))
    llsg_mean_fps = np.zeros((n_thresholds))

    rf_detections = np.array(ROC2.sodirf_detections)
    rf_fps = np.array(ROC2.sodirf_fps)
    rf_tpr = np.zeros((n_thresholds))
    rf_mean_fps = np.zeros((n_thresholds))

    dnn_detections = np.array(ROC2.sodinn_detections)
    dnn_fps = np.array(ROC2.sodinn_fps)
    dnn_tpr = np.zeros((n_thresholds))
    dnn_mean_fps = np.zeros((n_thresholds))

    for i in range(n_thresholds):
        adi_tpr[i] = adi_detections[:, i].tolist().count(1) / n_injections
        adi_mean_fps[i] = adi_fps[:, i].mean()
        pca_tpr[i] = pca_detections[:, i].tolist().count(1) / n_injections
        pca_mean_fps[i] = pca_fps[:, i].mean()
        llsg_tpr[i] = llsg_detections[:, i].tolist().count(1) / n_injections
        llsg_mean_fps[i] = llsg_fps[:, i].mean()
        rf_tpr[i] = rf_detections[:, i].tolist().count(1) / n_injections
        rf_mean_fps[i] = rf_fps[:, i].mean()
        dnn_tpr[i] = dnn_detections[:, i].tolist().count(1) / n_injections
        dnn_mean_fps[i] = dnn_fps[:, i].mean()

    fig = figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)

    plot(adi_mean_fps, adi_tpr, '--',  alpha=line_alpha, color='#d62728')
    plot(adi_mean_fps, adi_tpr, '^', label='ADI-MEDSUB', alpha=marker_alpha,
         ms=markersize, color='#d62728')

    plot(pca_mean_fps, pca_tpr, '--', alpha=line_alpha, color='#ff7f0e')
    plot(pca_mean_fps, pca_tpr, 'X', label='PCA', alpha=marker_alpha,
         ms=markersize, color='#ff7f0e')

    plot(llsg_mean_fps, llsg_tpr, '--', alpha=line_alpha, color='#2ca02c')
    plot(llsg_mean_fps, llsg_tpr, 'P', label='LLSG', alpha=marker_alpha,
         ms=markersize, color='#2ca02c')

    plot(rf_mean_fps, rf_tpr, '--', alpha=line_alpha, color='#9467bd')
    plot(rf_mean_fps, rf_tpr, 's', label='SODIRF', alpha=marker_alpha,
         ms=markersize, color='#9467bd')

    plot(dnn_mean_fps, dnn_tpr, '--', alpha=line_alpha, color='#1f77b4')
    plot(dnn_mean_fps, dnn_tpr, 'p', label='SODINN', alpha=marker_alpha,
         ms=markersize, color='#1f77b4')

    legend(loc=legend_loc, prop={'size': legend_size})
    if xlog: ax.set_xscale("symlog")
    ylim(ymin=ymin, ymax=ymax)
    xlim(xmin=xmin)
    if yaxis_label: ylabel('TPR')
    if xaxis_label: xlabel('Full-frame mean FPs')
    grid(alpha=0.4)

    if show_data_labels:
        if not isinstance(label_skip_one, list):
            label_skip_one = [label_skip_one for _ in range(5)]

        if label_skip_one[0]:
            lab1x = adi_mean_fps[1::2]
            lab1y = adi_tpr[1::2]
            thr1 = ROC2.cadi_thresholds[1::2]
        else:
            lab1x = adi_mean_fps
            lab1y = adi_tpr
            thr1 = ROC2.cadi_thresholds

        if label_skip_one[1]:
            lab2x = pca_mean_fps[1::2]
            lab2y = pca_tpr[1::2]
            thr2 = ROC2.pca_thresholds[1::2]
        else:
            lab2x = pca_mean_fps
            lab2y = pca_tpr
            thr2 = ROC2.pca_thresholds

        if label_skip_one[2]:
            lab3x = llsg_mean_fps[1::2]
            lab3y = llsg_tpr[1::2]
            thr3 = ROC2.llsg_thresholds[1::2]
        else:
            lab3x = llsg_mean_fps
            lab3y = llsg_tpr
            thr3 = ROC2.llsg_thresholds

        if label_skip_one[3]:
            lab4x = rf_mean_fps[1::2]
            lab4y = rf_tpr[1::2]
            thr4 = ROC2.sodirf_thresholds[1::2]
        else:
            lab4x = rf_mean_fps
            lab4y = rf_tpr
            thr4 = ROC2.sodirf_thresholds

        if label_skip_one[4]:
            lab5x = dnn_mean_fps[1::2]
            lab5y = dnn_tpr[1::2]
            thr5 = ROC2.sodinn_thresholds[1::2]
        else:
            lab5x = dnn_mean_fps
            lab5y = dnn_tpr
            thr5 = ROC2.sodinn_thresholds

        labels = []
        for i, xy in enumerate(zip(lab5x + label_xgap, lab5y + label_ygap)):
            labels.append(ax.annotate('{:.2f}'.format(thr5[i]),
                                      xy=xy, xycoords='data',
                                      alpha=label_alpha,
                                      fontsize=label_fontsize,
                                      weight=label_weight,
                                      rotation=label_rotation, color='#1f77b4',
                                      annotation_clip=True))

        for i, xy in enumerate(zip(lab4x + label_xgap, lab4y + label_ygap)):
            labels.append(ax.annotate('{:.2f}'.format(thr4[i]),
                                      xy=xy, xycoords='data',
                                      alpha=label_alpha,
                                      fontsize=label_fontsize,
                                      weight=label_weight,
                                      rotation=label_rotation, color='#9467bd',
                                      annotation_clip=True))

        for i, xy in enumerate(zip(lab3x + label_xgap, lab3y + label_ygap)):
            labels.append(ax.annotate('{:.2f}'.format(thr3[i]),
                                      xy=xy, xycoords='data',
                                      alpha=label_alpha,
                                      fontsize=label_fontsize,
                                      weight=label_weight,
                                      rotation=label_rotation, color='#2ca02c',
                                      annotation_clip=True))

        for i, xy in enumerate(zip(lab2x + label_xgap, lab2y + label_ygap)):
            labels.append(ax.annotate('{:.2f}'.format(thr2[i]),
                                      xy=xy, xycoords='data',
                                      alpha=label_alpha,
                                      fontsize=label_fontsize,
                                      weight=label_weight,
                                      rotation=label_rotation, color='#ff7f0e',
                                      annotation_clip=True))

        for i, xy in enumerate(zip(lab1x + label_xgap, lab1y + label_ygap)):
            labels.append(ax.annotate('{:.2f}'.format(thr1[i]),
                                      xy=xy, xycoords='data',
                                      alpha=label_alpha,
                                      fontsize=label_fontsize,
                                      weight=label_weight,
                                      rotation=label_rotation, color='#d62728',
                                      annotation_clip=True))

        mask = np.zeros(fig.canvas.get_width_height(), bool)

        fig.canvas.draw()

        for a in labels:
            bbox = a.get_window_extent()
            negpad = -2
            x0 = int(bbox.x0) + negpad
            x1 = int(np.ceil(bbox.x1)) + negpad
            y0 = int(bbox.y0) + negpad
            y1 = int(np.ceil(bbox.y1)) + negpad

            s = np.s_[x0:x1, y0:y1]
            if np.any(mask[s]):
                if hide_overlap_label:
                    a.set_visible(False)
            else:
                mask[s] = True

    if save_plot:
        if isinstance(save_plot, str):
            savefig(save_plot, dpi=dpi, bbox_inches='tight')
        else:
            savefig('roc_curve.pdf', dpi=dpi, bbox_inches='tight')

    results = Munch(dnn_mean_fps=dnn_mean_fps, dnn_tpr=dnn_tpr,
                    llsg_mean_fps=llsg_mean_fps, llsg_tpr=llsg_tpr,
                    pca_mean_fps=pca_mean_fps, pca_tpr=pca_tpr,
                    rf_mean_fps=rf_mean_fps, rf_tpr=rf_tpr,
                    adi_mean_fps=adi_mean_fps, adi_tpr = adi_tpr)

    return results