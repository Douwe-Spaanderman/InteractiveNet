import numpy as np

import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import morphology

from interactivenet.utils.utils import to_array


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def ImagePlot(
    img,
    GT,
    interaction=None,
    additional_scans=None,
    CT=False,
    radius=1,
    zoom=False,
    show=None,
    save=None,
    save_type="png",
    colors=["dodgerblue", "magenta", "cyan", "navy", "purple"],
    cmap=plt.cm.gray,
    ax=None,
):
    if not isinstance(GT, list):
        segs = [GT]
    else:
        segs = GT

    if additional_scans:
        segs.extend(additional_scans)

    if interaction:
        segs.extend(interaction)

    img = to_array(img)
    segs = [to_array(seg) for seg in segs]

    if len(img.shape) == 4:
        img = img[0]
        segs = [seg[0] for seg in segs]

    if interaction:
        _, _, inds_z = np.where(segs[-1] > 0.5)
        inds_z = sorted(inds_z)
        slice = inds_z[int((len(inds_z) - 1) / 2)]
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
        img_threshold[img_threshold < window_center - window_width / 2] = (
            window_center - window_width / 2
        )
        img_threshold[img_threshold > window_center + window_width / 2] = (
            window_center + window_width / 2
        )

    return_ax = True
    if not ax:
        return_ax = False
        fig = plt.figure(frameon=False)
        w = img.shape[1] * 0.05
        h = (img.shape[1] * 0.05) * (img.shape[0] / img.shape[1])
        fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
    else:
        ax.set_axis_off()

    ax.imshow(img_threshold[:, :, slice], cmap=cmap, aspect="auto")

    # Plot overlays for each contour
    disk = morphology.disk(radius)
    for seg, color in zip(segs, colors):
        contour_or = np.squeeze(seg[:, :, slice])
        contour_e = morphology.binary_dilation(contour_or, disk)
        contour = np.subtract(contour_e, contour_or)
        y, x = np.nonzero(contour)
        ax.scatter(x, y, color=color, marker="s")

    plt.axis("off")

    if show:
        plt.show()
    if save:
        name = save.name
        save = save.parent
        save.mkdir(parents=True, exist_ok=True)
        fig.savefig(save / f"{name}.{save_type}")
        plt.close("all")

    if return_ax:
        return ax
    else:
        return fig
