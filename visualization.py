import numpy as np
import math
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import SimpleITK as sitk
from scipy import ndimage
import imageio
from pathlib import Path
import random
import shutil

def create_tile(img, xslices=[], yslices=[], zslices=[], meta_file=None):
    if meta_file is not None:
        null_template = sitk.Image([0,0], meta_file.GetPixelIDValue(), meta_file.GetNumberOfComponentsPerPixel())
    else:
        null_template = sitk.Image([0,0], img.GetPixelIDValue(), img.GetNumberOfComponentsPerPixel())

    img_xslices = [img[:,:,s] for s in xslices]
    img_yslices = [img[:,s,:] for s in yslices]
    img_zslices = [img[s,:,:] for s in zslices]

    maxlen = max(len(img_xslices), len(img_yslices), len(img_zslices))

    img_slices = []
    d = 0
    if len(img_xslices):
        img_slices += img_xslices + [null_template]*(maxlen-len(img_xslices))
        d += 1
        
    if len(img_yslices):
        img_slices += img_yslices + [null_template]*(maxlen-len(img_yslices))
        d += 1
    
    if len(img_zslices):
        img_slices += img_zslices + [null_template]*(maxlen-len(img_zslices))
        d +=1

    if maxlen != 0:
        if sum([len(img_xslices) > 0, len(img_yslices) > 0, len(img_zslices) > 0]) >= 2:
            img = sitk.Tile(img_slices, [maxlen,d])
        else:
            img = sitk.Tile(img_slices, [8,math.ceil(maxlen/8)])

        return img

def myshow(img, overlap=None, title=None, margin=0.05, dpi=80, titlesize=12):
    nda = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    
    if nda.ndim == 3:
        # fastest dim, either component or x
        c = nda.shape[-1]
        
        # the the number of components is 3 or 4 consider it an RGB image
        if not c in (3,4):
            nda = nda[nda.shape[0]//2,:,:]
    
    elif nda.ndim == 4:
        c = nda.shape[-1]
        
        if not c in (3,4):
            raise Runtime("Unable to show 3D-vector Image")
            
        # take a z-slice
        nda = nda[nda.shape[0]//2,:,:,:]
            
    ysize = nda.shape[0]
    xsize = nda.shape[1]

    # Make a figure big enough to accomodate an axis of xpixels by ypixels
    # as well as the ticklabels, etc...
    enlarge = 1000 / min([ysize, xsize])
    figsize = (1 + margin) * ysize / dpi * enlarge, (1 + margin) * xsize / dpi * enlarge

    fig = plt.figure(figsize=figsize, dpi=dpi)
    # Make the axis the right size...
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
    
    extent = (0, xsize*spacing[1], ysize*spacing[0], 0)

    t = ax.imshow(nda,extent=extent,interpolation=None)
    
    if nda.ndim == 2:
        t.set_cmap("gray")

    if overlap is not None:
        overlap = sitk.GetArrayFromImage(overlap)
        overlap = overlap.astype(np.float32)
        overlap[overlap == 0] = np.nan
        t2 = ax.imshow(overlap, cmap='Set2', extent=extent, interpolation=None)
    
    if(title):
        plt.title(title, fontsize=titlesize)

def myshow3d(mask, image=None, overlap=None, xslices=[], yslices=[], zslices=[], enlarge=2, title=None, margin=0.05, dpi=80, show=True, save=None, gif=False):
    '''

    '''
    if overlap is not None:
        # Add dilation, to create a larger dot for plotting - do this for each 2D layer as otherwise it looks weird
        slices = []
        for i, slice_overlap in enumerate(overlap.T):
            #slices.append(ndimage.binary_dilation(slice_overlap, mask=mask[i,:,:], iterations=enlarge).astype(slice_overlap.dtype))
            slices.append(ndimage.binary_dilation(slice_overlap, iterations=enlarge).astype(slice_overlap.dtype))

        overlap = np.stack(slices, axis=0).T

    if gif:
        if image is not None:
            img = sitk.GetImageFromArray(image)
            size = img.GetSize()
            if overlap is not None:
                mask = sitk.GetImageFromArray(overlap)
            else:
                mask = sitk.GetImageFromArray(mask)

        else:
            img = sitk.GetImageFromArray(mask)
            size = img.GetSize()
            if overlap is not None:
                mask = sitk.GetImageFromArray(overlap)

        filenames = []

        random_tmp = Path("tmp", str(random.randint(0,10000)))
        random_tmp.mkdir(parents=True, exist_ok=True)
        for i in range(0, size[2]):
            myshow(img[:,:,i], mask[:,:,i], title, margin, dpi)

            filename = f'tmp/tmp_gif{i}.png'
            filenames.append(filename)
            plt.savefig(filename)
            plt.close("all")

        with imageio.get_writer(f'{save}_3D.gif', mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        shutil.rmtree(random_tmp)
    else:
        if image is not None:
            img = sitk.GetImageFromArray(image)
            img = create_tile(img, xslices, yslices, zslices)
            mask = sitk.GetImageFromArray(mask)
            if overlap is not None:
                overlap = sitk.GetImageFromArray(overlap)
                mask = create_tile(overlap, xslices, yslices, zslices, mask) 
            else:
                mask = create_tile(mask, xslices, yslices, zslices, mask)   
            
            myshow(img, mask, title, margin, dpi, titlesize=30)
        else:
            img = sitk.GetImageFromArray(mask)
            meta_file = img
            img = create_tile(img, xslices, yslices, zslices)
            if overlap is not None:
                overlap = sitk.GetImageFromArray(overlap)
                overlap = create_tile(overlap, xslices, yslices, zslices, meta_file)

            myshow(img, overlap, title, margin, dpi, titlesize=30)
        
        if show:
            plt.show()
        if save:
            plt.savefig(f'{save}_snapshot.png')
            plt.close("all")