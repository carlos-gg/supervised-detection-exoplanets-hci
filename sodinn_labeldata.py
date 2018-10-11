"""
Generation of labeled data for supervised learning. To be used to train the
discriminative models. 
"""
from __future__ import print_function
from __future__ import absolute_import

__all__ = ['make_training_data',
           'sample_flux_expvar']


import numpy as np
from vip_hci.preproc import cube_derotate, cube_crop_frames, cube_derotate
from vip_hci.preproc import frame_crop, check_pa_vector
from vip_hci.phot import (noise_per_annulus, frame_quick_report,
                          cube_inject_companions)
from vip_hci.conf import time_ini, timing, time_fin
from vip_hci.conf.utils_conf import eval_func_tuple as futup
from vip_hci.var import frame_center
from vip_hci.madi import adi
from scipy.stats import skewnorm
from matplotlib.pyplot import (figure, show, plot, grid,
                               xlabel, ylabel, ylim, xlim, legend)
from multiprocessing import Pool, cpu_count
from multiprocessing import get_start_method
import itertools as itt
from munch import *
from .sodinn_utils import (normalize_01, create_synt_cube, save_res, svd_decomp,
                           get_indices_annulus, frame_shift, get_cumexpvar)


def make_training_data(input_array, angle_list, psf, n_samples=5000, 
                       k_list=[1,2,3,5,10,20,50,100], inrad=0, outrad=12, 
                       size_patch=11, rangerot=359, dist_flux_p1=180,
                       dist_flux_p2=80, dist_flux_p3=5, dist_flux='uniform', 
                       fwhm=4, plsc=0.02719, collapse_func=np.mean, 
                       scaling=None, normalize='slice', mask=None,
                       percen_orig_zero_patches=0.1, save=None, nproc=1, 
                       nproc2=1, percen_one_augment=0.5, shifts=None, 
                       dtype=np.float32, interp='nearneig', lr_mode='eigen'):
    """
    n_samples : half ZEROS, half ONES. For ONEs, half_n_samples SVDs

    for 'skewnormal' flux distribution, dist_flux_p3 is the skewness

    mask is a list of tuples X,Y

    inputarr is a 3d array or list of 3d arrays
    """
    starttime = time_ini()
    
    if not isinstance(input_array, list):
        input_array = [input_array]

    list_X_zeros = []
    list_Y_zeros = []
    list_X_ones = []
    list_Y_ones = []
    if normalize:
        list_X_zerosnor = []
        list_X_onesnor = []

    if get_start_method() == 'fork' and lr_mode in ['pytorch', 'eigenpytorch',
                                                    'randpytorch']:
        msg = "Cannot use pytorch and multiprocessing outside main (i.e. from "
        msg += "a jupyter cell). See: "
        msg += "http://pytorch.org/docs/0.3.1/notes/multiprocessing.html."
        raise RuntimeError(msg)

    for ncu, cube in enumerate(input_array):
        print("\nCube {} out of {}".format(ncu+1, len(input_array)))
        n_frames = cube.shape[0]
        frsize = int(cube.shape[1])
        half_n_samples = int(n_samples/2.)

        if frsize>outrad+outrad+size_patch+2:
            zerosfr = np.zeros((frsize, frsize))
            if mask is not None:
                for i in range(len(mask)):
                    xy = mask[i]
                    zerosfr[xy[1], xy[0]] = 1

            frsize = outrad+outrad+size_patch+2
            cube = cube_crop_frames(cube, frsize, verbose=False)
            zerosfr = frame_crop(zerosfr, frsize)
            # mask to cropped cube
            if mask is not None:
                ymask, xmask = np.where(zerosfr==1)
                orig_mask = list(mask) # copying original mask
                mask = []
                for i in range(ymask.shape[0]):
                    mask.append((xmask[i],ymask[i]))
            else:
                orig_mask = mask

        ########################################################################
        # making zeros
        print("Creating the ZEROs samples")
        sca = scaling

        if not inrad >= int(size_patch/2.) + fwhm:
            msg = "Warning: The patches are overlapping with the inner 1xFWHM "
            msg += "annulus"
            print()
        if not inrad > int(np.round(size_patch/2.)):
            raise RuntimeError("Inner radius must be larger than half size_patch")

        resdec = svd_decomp(cube, angle_list, size_patch, inrad, outrad, sca,
                            k_list, collapse_func, neg_ang=True,  interp=interp,
                            nproc=nproc2, lr_mode=lr_mode)

        cube_residuals, cube_residuals2 = resdec

        yy, xx = get_indices_annulus((frsize,frsize), inrad, outrad, mask=mask,
                                    maskrad=int(np.round((size_patch/2.)+1)),
                                    verbose=False)
        num_patches = yy.shape[0]
        patches_array = []

        # 3D: n_k_list, y, x
        cube_residuals = np.array(cube_residuals)
        cube_residuals2 = np.array(cube_residuals2)
        print("Total patches in annulus = {:}".format(num_patches))

        # patches_array: 4D hal_n_frames*factor, n_k_list, size_patch, size_patch
        # taking percen_orig_zero_patches random patches with normal angles
        num_patches_cube_orig = min(half_n_samples, 
                                    int(num_patches * percen_orig_zero_patches))
        print("{} original cube zeros patches".format(num_patches_cube_orig))
        for i in range(num_patches_cube_orig):
            # cropping the multiK cube at same position
            which_patch = np.random.randint(0, num_patches, size=1)[0]
            patches_array.append(cube_crop_frames(cube_residuals, size_patch,
                                                  xy=(int(xx[which_patch]),
                                                      int(yy[which_patch])),
                                                  verbose=False))

        # DATA AUGMENTATION using negative angles
        if len(patches_array)<half_n_samples:
            print("Data augmentation of zeros patches")
            num_patches_negang = min(num_patches, 
                                     half_n_samples-num_patches_cube_orig)
            msg = "{} zeros patches from cubes with negative angles"
            print(msg.format(num_patches_negang))
            if num_patches_negang < num_patches:
                ind_negang = np.random.choice(num_patches, 
                                              size=num_patches_negang, 
                                              replace=False).astype(int)
            elif num_patches_negang == num_patches:
                ind_negang = range(num_patches)
            for i in range(num_patches_negang):
                which_patch = ind_negang[i]
                patches_array.append(cube_crop_frames(cube_residuals2, 
                                                      size_patch,
                                                      xy=(int(xx[which_patch]),
                                                          int(yy[which_patch])),
                                                      verbose=False))

            # DATA AUGMENTATION by mean combinations (including one patch from
            # the orig cube), 70%
            if len(patches_array) < half_n_samples:
                curr_len = len(patches_array)
                num_patches_aug_ave = int((half_n_samples - curr_len)*0.7)
                print("{} random averages".format(num_patches_aug_ave))
                for i in range(num_patches_aug_ave):
                    ind1 = np.random.randint(0, num_patches_cube_orig, 1)[0]
                    ind2 = np.random.randint(num_patches_cube_orig, 
                                             curr_len, 1)[0]
                    ind3 = np.random.randint(num_patches_cube_orig, 
                                             curr_len, 1)[0]
                    patches_array.append(np.mean((patches_array[ind1],
                                                  patches_array[ind2],
                                                  patches_array[ind3]), axis=0))

                # DATA AUGMENTATION by rotation
                curr_len = len(patches_array)
                num_patches_aug_rot = int(half_n_samples - len(patches_array))
                print("{} rotations".format(num_patches_aug_rot))
                for i in range(num_patches_aug_rot):
                    ind = np.random.randint(0, curr_len, 1)[0]
                    angs = np.random.randint(0, rangerot, 1)*np.ones((len(k_list)))
                    patches_array.append(cube_derotate(patches_array[ind], angs, 
                                                       nproc=nproc2, 
                                                       interpolation=interp))


        # 4D: n_samples/2, n_k_list, size_patch, size_patch
        X_zeros_array = np.array(patches_array)
        Y_zeros_vector = np.zeros((X_zeros_array.shape[0]))

        X_zeros_array_nor = normalize_01(X_zeros_array, normalize)

        print("Zeros shape:", X_zeros_array.shape)
        timing(starttime)

        ########################################################################
        ########################################################################
        ########################################################################
        # making ones, injecting companions. The other half of n_samples
        print("Creating the ONEs samples")
        n_req_augment = int(half_n_samples*percen_one_augment)
        n_req_inject = half_n_samples - n_req_augment
        # 4D: n_samples/2, n_k_list, size_patch, size_patch
        X_ones_array = np.empty((half_n_samples, len(k_list), size_patch, 
                                size_patch))
        print("{} injections:".format(n_req_inject))

        if dist_flux == 'skewnormal':
            mean_flux = dist_flux_p1
            std_flux = dist_flux_p2
            fluxes = skewnorm.rvs(dist_flux_p3, loc=mean_flux, scale=std_flux, 
                                  size=n_req_inject)
        elif dist_flux == 'normal':
            mean_flux = dist_flux_p1
            std_flux = dist_flux_p2
            fluxes = np.random.normal(mean_flux, std_flux, size=n_req_inject)
        elif dist_flux == 'uniform':
            if not dist_flux_p2 > dist_flux_p1:
                err_msg = 'dist_flux_p2 must be larger than dist_flux_p1 when '
                err_msg += 'dist_flux==uniform'
                raise ValueError(err_msg)
            fluxes = np.random.uniform(dist_flux_p1, dist_flux_p2, 
                                       size=n_req_inject)
        else:
            raise ValueError('Flux distribution not recognized')
        fluxes = np.sort(fluxes)
        inds_inj = np.random.randint(0, num_patches, size=n_req_inject)

        dists = []
        thetas = []
        for m in range(n_req_inject):
            injx = xx[inds_inj[m]] 
            injy = yy[inds_inj[m]] 
            injx -= frame_center(cube[0])[1]
            injy -= frame_center(cube[0])[0]
            dist = np.sqrt(injx**2+injy**2)
            theta = np.mod(np.arctan2(injy,injx)/np.pi*180,360)
            dists.append(dist)
            thetas.append(theta)

        if not nproc: nproc = int((cpu_count()/4))

        if nproc == 1:
            for m in range(n_req_inject):
                cufc, cox, coy = create_synt_cube(cube, psf, angle_list,
                                                  plsc, theta=thetas[m], 
                                                  flux=fluxes[m], dist=dists[m], 
                                                  verbose=False)
                cox = int(np.round(cox))
                coy = int(np.round(coy))

                cube_residuals = svd_decomp(cufc, angle_list, size_patch, 
                                            inrad, outrad, sca, k_list, 
                                            collapse_func, neg_ang=False,
                                            lr_mode=lr_mode, nproc=nproc2,
                                            interp=interp)

                # one patch residuals per injection
                X_ones_array[m, :] = cube_crop_frames(np.array(cube_residuals), size_patch,
                                                      xy=(cox,coy), verbose=False)

        elif nproc > 1:
            if lr_mode in ['cupy','randcupy','eigencupy']:
                raise RuntimeError('CUPY does not play well with multiprocessing')
                
            flux_dist_theta = zip(fluxes, dists, thetas)

            pool = Pool(processes=int(nproc))
            res = pool.map(futup, zip(itt.repeat(_inject_FC), itt.repeat(cube),
                                      itt.repeat(psf), itt.repeat(angle_list),
                                      itt.repeat(plsc), itt.repeat(inrad),
                                      itt.repeat(outrad), flux_dist_theta,
                                      itt.repeat(k_list), itt.repeat(sca),
                                      itt.repeat(collapse_func),
                                      itt.repeat(size_patch),
                                      itt.repeat(lr_mode), itt.repeat(interp)))
            pool.close()
            for m in range(n_req_inject):
                X_ones_array[m, :] = res[m]

        # ONES data augmentation, rotations + 1px shifts
        if n_req_augment>0:
            if shifts is not None:
                print("{} rotations+shifts:".format(n_req_augment))
            else:
                print("{} rotations:".format(n_req_augment))

            if percen_one_augment>=0.5:
                randind = np.random.choice(n_req_inject, size=n_req_augment, 
                                           replace=True)
            else:
                randind = np.random.choice(n_req_inject, size=n_req_augment, 
                                           replace=False)

            for i in range(n_req_augment):
                angs = np.random.randint(0, rangerot, 1)*np.ones((len(k_list)))
                patch_rot = cube_derotate(X_ones_array[int(randind[i])], angs,
                                          interpolation=interp)
                if shifts is not None:
                    # random shifts [0.5:shifts] pixels in x and y
                    shy = np.random.uniform(0.5, shifts, 1)
                    shx = np.random.uniform(0.5, shifts, 1)
                    for j in range(len(k_list)):
                        fr_shifted = frame_shift(patch_rot[j], shy, shx)
                        X_ones_array[n_req_inject+i, j, :, :] = fr_shifted
                else:
                    X_ones_array[n_req_inject+i] = patch_rot

        # Normalization between 0 and 1
        X_ones_array_nor = normalize_01(X_ones_array, normalize)

        Y_ones_vector = np.ones((X_ones_array.shape[0]))
        print("Ones shape:", X_ones_array.shape)

        # Populating the combined ZEROS and ONES arrays
        list_X_zeros.append(X_zeros_array)
        list_Y_zeros.append(Y_zeros_vector)
        list_X_ones.append(X_ones_array)
        list_Y_ones.append(Y_ones_vector)
        if normalize:
            list_X_onesnor.append(X_ones_array_nor)
            list_X_zerosnor.append(X_zeros_array_nor)
        timing(starttime)
    
    X_zeros_array = np.array(list_X_zeros).reshape(-1, len(k_list), 
                                                   size_patch, size_patch)
    X_ones_array = np.array(list_X_ones).reshape(-1, len(k_list), 
                                                 size_patch, size_patch)
    if normalize:
        X_zeros_array_nor = np.array(list_X_zerosnor).reshape(-1, len(k_list), 
                                                         size_patch, size_patch)
        X_ones_array_nor = np.array(list_X_onesnor).reshape(-1, len(k_list), 
                                                         size_patch, size_patch)
    Y_zeros_vector = np.array(list_Y_zeros).flatten()
    Y_ones_vector = np.array(list_Y_ones).flatten()

    X_train = np.concatenate((X_zeros_array, X_ones_array), axis=0)
    Y_train = np.concatenate((Y_zeros_vector, Y_ones_vector), axis=0)
    if normalize:
        X_train_nor = np.concatenate((X_zeros_array_nor, X_ones_array_nor), 
                                     axis=0)

    X_train = X_train.astype(dtype)
    Y_train = Y_train.astype(dtype)

    print("-------------------------------")
    print('X shape:', X_train.shape, '  |   Y shape:', Y_train.shape)

    timing(starttime)
    fintime = time_fin(starttime)

    bunch_results = Munch(x=X_train, y=Y_train, nsamp=n_samples*len(input_array), 
                          fwhm=fwhm, plsc=plsc, klist=k_list, 
                          inrad=inrad, outrad=outrad, 
                          sizepatch=size_patch, mask=orig_mask, 
                          collaf=collapse_func, scaling=scaling, 
                          norm=normalize, fluxes=fluxes, fluxdistro=dist_flux, 
                          fluxdistrop1=dist_flux_p1, fluxdistrop2=dist_flux_p2, 
                          fluxdistrop3=dist_flux_p3, dists=dists, thetas=thetas, 
                          runtime=fintime, perorigzeros=percen_orig_zero_patches, 
                          peraugones=percen_one_augment, shifts=shifts, 
                          rangerot=rangerot, procfrsize=frsize, 
                          npatches=num_patches)
    
    if normalize: bunch_results.xnor = X_train_nor

    if save is not None and isinstance(save, str):
        save_res(save+'.p', bunch_results)
        print("Saved file: "+save+'.p')
        timing(starttime)

    return bunch_results



def sample_flux_expvar(cube, angle_list, psf, fwhm=4, plsc=0.02719, ninj=100, 
                       inrad=0, outrad=12, flux_min=2, flux_max=100000, nproc=10, 
                       size_patch=7, k_list=[1,2,3,5,10,20,50,100], 
                       expvar_mode='fullfr', plotvlines=[1,3], figsize=(6,4),
                       dpi=100, ymax=None):
    """
    Sensible flux intervals depend on a combination of factors, # of frames, 
    range of rotation, correlation, glare intensity. Similar for the the PCs 
    intervals. We measure the cummulative explained variance ratio on the 
    standardized matrix.  
    """
    starttime = time_ini()

    n_frames = cube.shape[0]
    frsize = int(cube.shape[1])

    yy, xx = get_indices_annulus((frsize,frsize), inrad, outrad, verbose=False)
    num_patches = yy.shape[0]

    fluxes = np.random.uniform(flux_min, flux_max, size=ninj)
    inds_inj = np.random.randint(0, num_patches, size=ninj)

    snrs_list = []
    fluxes_list = []
    dists = []
    thetas = []
    for i in range(ninj):
        injx = xx[inds_inj[i]] 
        injy = yy[inds_inj[i]] 
        injx -= frame_center(cube[0])[1]
        injy -= frame_center(cube[0])[0]
        dist = np.sqrt(injx**2+injy**2)
        theta = np.mod(np.arctan2(injy,injx)/np.pi*180,360)
        dists.append(dist)
        thetas.append(theta)
    
    flux_dist_theta = zip(fluxes, dists, thetas)

    pool = Pool(processes=int(nproc))
    res = pool.map(futup, itt.izip(itt.repeat(_get_adi_snrs),
                                    itt.repeat(cube),
                                    itt.repeat(psf),
                                    itt.repeat(angle_list),
                                    itt.repeat(fwhm),
                                    itt.repeat(plsc),
                                    flux_dist_theta))
    pool.close()
    for i in range(ninj):
        fluxes_list.append(res[i][0])
        snrs_list.append(res[i][1])

    timing(starttime)
    fintime = time_fin(starttime)

    # Figure of flux vs s/n
    figure(figsize=figsize, dpi=dpi)
    plot(fluxes_list, snrs_list, 'o', alpha=0.3, markersize=6)
    grid(which='major')
    ylim(0, ymax)
    xlim(0, np.max(np.array(fluxes_list)))
    ylabel('ADI-medsub median S/N (three equidistant angles)')
    xlabel('Fake companion scale factor (ADUs)')
    for i in plotvlines:
        plot((0,max(fluxes_list)), (i,i), '--', color='#ff7f0e')
    show()
    
    #---------------------------------------------------------------------------
    # CUM EXPLAINED VARIANCE 
    ratio_cumsum, ratio_cumsum_klist = get_cumexpvar(cube, expvar_mode, inrad, 
                                                     outrad, size_patch,
                                                     k_list, verbose=True)

    # Plotting of cumulative explained variance
    figure(figsize=figsize, dpi=dpi)
    plot(range(1,ratio_cumsum.shape[0]+1), ratio_cumsum, 'o-', 
         alpha=0.5, lw=4)
    legend(loc='best', frameon=False, fontsize='medium')
    ylabel('Cumulative explained variance ratio')
    xlabel('Singular vectors')
    grid(linestyle='solid')
    xlim(-0.02, ratio_cumsum.shape[0]+1)
    plot((k_list[-1], k_list[-1]), (min(ratio_cumsum),1), '--', color='#ff7f0e')
    ylim(min(ratio_cumsum), top=1)

    return fluxes_list, snrs_list


def _get_adi_snrs(cube, psf, angle_list, fwhm, plsc, flux_dist_theta):
    """
    """
    meansnr = []
    theta = flux_dist_theta[2]
    for ang in [theta, theta+120, theta+240]:    
        cube_fc, cx, cy = create_synt_cube(cube, psf, angle_list, plsc,
                                           flux=flux_dist_theta[0], 
                                           dist=flux_dist_theta[1], theta=ang, 
                                           verbose=False)
        frtemp = adi(cube_fc, angle_list, verbose=False)
        res = frame_quick_report(frtemp, fwhm, source_xy=(cx, cy), verbose=False)
        meansnr.append(np.mean(res[1]))

    meansnr = np.array(meansnr)
    return flux_dist_theta[0], np.median(meansnr)


def _inject_FC(cube, psf, angle_list, plsc, inrad, outrad, flux_dist_theta,
               k_list, sca, collapse_func, size_patch, lr_mode, interp):
    """ One patch residuals per injection
    """
    cubefc, cox, coy = create_synt_cube(cube, psf, angle_list, plsc,
                                        flux=flux_dist_theta[0], 
                                        dist=flux_dist_theta[1], 
                                        theta=flux_dist_theta[2], verbose=False)
    cox = int(np.round(cox))
    coy = int(np.round(coy))

    cube_residuals = svd_decomp(cubefc, angle_list, size_patch, inrad, outrad, 
                                sca, k_list, collapse_func, neg_ang=False,
                                lr_mode=lr_mode, nproc=1, interp=interp)
    patch = cube_crop_frames(np.array(cube_residuals), size_patch, xy=(cox,coy), 
                             verbose=False)
    return patch





