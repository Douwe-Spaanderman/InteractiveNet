import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import morphology

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def ImagePlot(img, GT, annotation=None, additional_scans=None, distancemap=False, CT=False, radius=1, zoom=False, show=None, save=None, save_type='png', colors=['dodgerblue', 'magenta', 'cyan', 'navy', 'purple']):
    plt.close("all")
    if not isinstance(object, list):
        segs = [GT]
    else:
        segs = GT

    if additional_scans:
        segs.extend(additional_scans)

    if annotation and not distancemap:
        segs.extend(annotation)
    elif distancemap:
        img = annotation[0]

    if len(img.shape) == 4:
        img = img[0]
        segs = [x[0] for x in segs]
        
    pad = 20
    if annotation and not distancemap:
        _, _, inds_z = np.where(segs[-1] > 0.5)
        inds_z = sorted(inds_z)
        slice = inds_z[int((len(inds_z) - 1)/2)]
    else:
        AreaCentre = ndimage.measurements.center_of_mass(segs[0])
        AreaCentre = [int(x) for x in AreaCentre]
        slice = AreaCentre[2]

    x, y = np.nonzero(segs[0][:, :, slice])

    # Threshold the image to get a decent window
    img_threshold = img.copy()
    if CT:
        window_center = 60
        window_width = 400
        img_threshold[img_threshold < window_center - window_width/2] = window_center - window_width/2
        img_threshold[img_threshold > window_center + window_width/2] = window_center + window_width/2

    f = plt.figure(frameon=False)
    w = img.shape[1] * 0.05
    h = (img.shape[1] * 0.05) * (img.shape[0] / img.shape[1])
    f.set_size_inches(w, h)
    ax = plt.Axes(f, [0., 0., 1., 1.])
    ax.set_axis_off()
    f.add_axes(ax)
    ax.imshow(img_threshold[:, :, slice], cmap=plt.cm.gray, aspect='auto')

    # Plot overlays for each contour
    disk = morphology.disk(radius)
    for seg, color in zip(segs, colors):
        contour_or = np.squeeze(seg[:, :, slice])
        contour_e = morphology.binary_dilation(contour_or, disk)
        contour = np.subtract(contour_e, contour_or)
        y, x = np.nonzero(contour)
        plt.scatter(x, y, color=color, marker='s')

    plt.axis('off')

    if show:
        plt.show()
    if save:
        name = save.name
        save = save.parent
        save.mkdir(parents=True, exist_ok=True)
        f.savefig(save / f'{name}.{save_type}')
        plt.close("all")

    # Create figure and plot zoomed image
    if zoom:
        x, y = np.nonzero(seg[:, :, slice])
        xmin = max(np.min(x) - pad, 0)
        xmax = min(np.max(x) + pad, img.shape[1])
        ymin = max(np.min(y) - pad, 0)
        ymax = min(np.max(y) + pad, img.shape[2])
        # f = plt.figure(figsize=(16, 9))
        f = plt.figure(frameon=False)
        w = 10
        h = 10 * img.shape[0] / img.shape[1]
        f.set_size_inches(w, h)
        # ax = f.add_subplot(1, 1, 1)
        ax = plt.Axes(f, [0., 0., 1., 1.])
        ax.set_axis_off()
        f.add_axes(ax)
        ax.imshow(img_threshold[xmin:xmax, ymin:ymax, slice], cmap=plt.cm.gray, aspect='auto')

        # Plot overlays for each contour
        disk = morphology.disk(radius)
        segs =[seg]
        for seg, color in zip(segs, colors):
            contour_or = np.squeeze(seg[xmin:xmax, ymin:ymax, slice])
            contour_e = morphology.binary_dilation(contour_or, disk)
            contour = np.subtract(contour_e, contour_or)
            y, x = np.nonzero(contour)
            plt.scatter(x, y, color=color, marker='s')

        plt.axis('off')
        if show:
            plt.show()
        if save:
            f.savefig(save / f'{name}_zoomed.{save_type}')
            plt.close("all")
    
    return f

