import torch
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import get_worker_info
from scipy.ndimage import zoom, distance_transform_edt as dist_t
import heapq
import math
from skimage.segmentation import find_boundaries
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, find_objects, minimum_filter, maximum_filter
from scipy.signal import fftconvolve
from skimage.restoration import richardson_lucy
import psfmodels as psfm
from scipy.interpolate import PchipInterpolator
from skimage.morphology import disk
from scipy.ndimage import binary_erosion


from numba import njit


class SyntheticFluorStains(Dataset):
    def __init__(
            self,
            samples_per_epoch=10000,
            seed=0,
            N = 200,
            z_slices = 20,
            z_ratio = 1.5,
            n_seeds = 55
    ):
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.N = N
        self.z_slices = z_slices
        self.z_ratio = z_ratio
        self.n_seeds = n_seeds

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed + idx)

        labels, nuc_labels, nonnuc_labels, dist_labels_bin_r = self._generate_cell_tiles(rng)

        #concentration
        conc, conc_nonnuc = self._generate_conc(dist_labels_bin_r.copy(), labels, rng)

        out_vol, out_nonnuc_vol = self._processing(conc, conc_nonnuc, rng)
        data = np.stack([out_vol, out_nonnuc_vol], axis=1)


        #structure = disk(5)

        #for j in range(self.z_slices):
        #  nonnuc[j] = nonnuc[j] & ~binary_erosion(nonnuc[j], structure=structure)

        labels = np.stack([nuc_labels, nonnuc_labels], axis=1)
        #print(f" marker shape {markers.shape}")
        return data, labels



    def _spectral_noise(self, shape, bands, bdwidth=0.3, lo=0, hi=1, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        
        white = rng.standard_normal(shape)

        F = np.fft.fftn(white)

        kz = np.fft.fftfreq(shape[0])[:, None, None]
        ky = np.fft.fftfreq(shape[1])[None, :, None]
        kx = np.fft.fftfreq(shape[2])[None, None, :]

        k2 = kx**2 + ky**2 + kz**2
        k = np.sqrt(k2)

        filt = 0
        for i in range(0, bands.shape[0]):
            filt += bands[i, 0] * np.exp(-0.5 * ((k - 1/bands[i,1]) / (bdwidth/bands[i,1]))**2)

        out = np.fft.ifftn(F * filt).real

        out = (hi-lo)*((out - out.min())/(out.max() - out.min())) + lo

        return out

    @staticmethod
    @njit(cache=True)
    def _voronoi_numba(mask, seeds, max_radius=np.inf):


        Z, H, W = mask.shape
        n_seeds = seeds.shape[0]

        labels = np.zeros((Z, H, W), dtype=np.int32)
        dist = np.full((Z, H, W), np.inf, dtype=np.float64)

        R2 = max_radius * max_radius

        for z in range(Z):
            for y in range(H):
                for x in range(W):

                    if not mask[z, y, x]:
                        continue

                    best_dist2 = np.inf
                    best_label = 0

                    for i in range(n_seeds):

                        dz = z - seeds[i, 0]
                        dy = y - seeds[i, 1]
                        dx = x - seeds[i, 2]

                        d2 = dz * dz + dy * dy + dx * dx

                        if d2 < best_dist2:
                            best_dist2 = d2
                            best_label = i + 1

                    if best_dist2 <= R2:
                        labels[z, y, x] = best_label
                        dist[z, y, x] = np.sqrt(best_dist2)

        return labels, dist

    @staticmethod
    @njit(cache=True)
    def _centroids_from_labels(labels, n_seeds):

        Z, H, W = labels.shape

        centroids = np.zeros((n_seeds, 3), dtype=np.int64)
        counts = np.zeros(n_seeds, dtype=np.int64)

        sum_z = np.zeros(n_seeds, dtype=np.float64)
        sum_y = np.zeros(n_seeds, dtype=np.float64)
        sum_x = np.zeros(n_seeds, dtype=np.float64)

        for z in range(Z):
            for y in range(H):
                for x in range(W):

                    label = labels[z, y, x]

                    if label <= 0 or label > n_seeds:
                        continue

                    i = label - 1

                    sum_z[i] += z
                    sum_y[i] += y
                    sum_x[i] += x
                    counts[i] += 1

        for i in range(n_seeds):
            if counts[i] > 0:
                centroids[i, 0] = int(round(sum_z[i] / counts[i]))
                centroids[i, 1] = int(round(sum_y[i] / counts[i]))
                centroids[i, 2] = int(round(sum_x[i] / counts[i]))

        return centroids, counts

    @staticmethod
    @njit(cache=True)
    def _lloyd_update(seeds, centroids, counts, alpha):

        new_seeds = seeds.copy()

        for i in range(seeds.shape[0]):
            if counts[i] > 0:
                for d in range(seeds.shape[1]):

                    new_seeds[i, d] = int(round(
                        (1.0 - alpha) * seeds[i, d]
                        + alpha * centroids[i, d]
                    ))

        return new_seeds

    @staticmethod
    @njit(cache=True)
    def _project_to_foreground_numba(
        seeds,
        mask,
        nearest_z,
        nearest_y,
        nearest_x
    ):
        z_slices, N, _ = mask.shape

        out = seeds.copy()

        for i in range(seeds.shape[0]):
            z = int(round(seeds[i, 0]))
            y = int(round(seeds[i, 1]))
            x = int(round(seeds[i, 2]))

            z = max(0, min(z_slices - 1, z))
            y = max(0, min(N - 1, y))
            x = max(0, min(N - 1, x))

            if mask[z, y, x]:
                out[i, 0] = z
                out[i, 1] = y
                out[i, 2] = x

            else:
                out[i, 0] = nearest_z[z, y, x]
                out[i, 1] = nearest_y[z, y, x]
                out[i, 2] = nearest_x[z, y, x]

        return out


    def _anisotropic_lloyd_relaxation(
        self,
        mask,
        seeds,
        ratios,
        thetas,
        noise_theta,
        noise_ratio,
        n_iter=10,
        alpha=0.4,
        R=None,
        rng=np.random.default_rng()
    ):
        z_slices, N, N = mask.shape

        if R is None:
            R = np.full(mask.shape, np.inf)

        seeds = np.asarray(seeds, dtype=np.int64).copy()


        # Precompute closest foreground pixel
        _, nearest = dist_t(
            ~mask,
            return_indices=True
        )

        nearest_z = nearest[0]
        nearest_y = nearest[1]
        nearest_x = nearest[2]

        for _ in range(n_iter):

            labels, dist = self._voronoi_numba(
                mask,
                seeds
            )

            centroids, counts = self._centroids_from_labels(
                labels,
                len(seeds)
            )

            seeds = self._lloyd_update(
                seeds,
                centroids,
                counts,
                alpha
            )

            seeds = self._project_to_foreground_numba(
                seeds,
                mask,
                nearest_z,
                nearest_y,
                nearest_x
            )


        #final tiling
        labels, dist = self._anisotropic_geodesic_voronoi_numba(
            np.full((z_slices, N, N), 1),
            seeds,
            ratios,
            thetas,
            noise_theta,
            noise_ratio,
            R#\max cell radius
        )

        return seeds, labels, dist

    def _sample_fg_pts_density(self, mask, n, rng):

        coords = np.argwhere(mask)


        idx = rng.choice(
            len(coords),
            size=n,
            replace=False
        )

        return coords[idx]
    
    def _generate_cell_tiles(self, rng):
        N = self.N
        z_slices = self.z_slices
        z_ratio = self.z_ratio
        n_seeds = rng.integers(round(self.n_seeds*0.6), round(self.n_seeds*2))


        surf = self._spectral_noise((z_slices, N, N), np.array([[5, 150], [0.5, 50]]), 0.9, -1, 1, rng)
        surf = (surf > -0.3)*(surf < 0.3)
        surf = surf.astype(np.bool)


        seeds = self._sample_fg_pts_density(surf, n_seeds, rng)
        ratios = rng.uniform(0.2, 20, size=(n_seeds))
        thetas = rng.uniform(-np.pi/2, np.pi/2, size=(n_seeds))
        #now these are just lists of values for each seed

        noise_theta = self._spectral_noise((z_slices, N, N), np.array([[5, 100]]), 1., -0.1, 0.1, rng)
        noise_ratio = self._spectral_noise((z_slices, N, N), np.array([[5, 100]]), 1., 0.6, 1.1, rng) #ratio movement should be biased negative?
        R = self._spectral_noise((z_slices, N, N), np.array([[5, 100]]), 1., 2,15, rng)





        seeds, labels, dist = self._anisotropic_lloyd_relaxation(surf, seeds, ratios, thetas, noise_theta, noise_ratio, 5, 1.0, R)

        boundary = np.zeros_like(labels, dtype=bool)


        for z in range(labels.shape[0]):
            boundary[z] = find_boundaries(
                labels[z],
                mode="inner"
            )

        labels_bin = labels.copy()
        labels_bin[boundary] = 0
        labels_bin[np.isinf(labels_bin)] = 0
        labels_bin = labels_bin != 0

        dist_labels_bin_r = np.zeros(labels_bin.shape, dtype=float)

        for z in range(labels_bin.shape[0]):
            dist_labels_bin_r[z] = dist_t(labels_bin[z])

        labels_bin = dist_labels_bin_r > 2

        nonnuc_labels_bin = dist_t(~boundary) < 3

        return labels, labels_bin, nonnuc_labels_bin, dist_labels_bin_r

    @staticmethod
    @njit(cache=True)
    def _assign_noise_to_labels(noise_map, objs, labels, rng):


        out = np.zeros(labels.shape, dtype=np.float64)

        Zn, Yn, Xn = noise_map.shape

        for lab, slc in enumerate(objs, 1):
            if slc is None:
                continue

            mask_roi = (labels[slc] == lab)

            dz_size, dy_size, dx_size = mask_roi.shape


            oz = rng.integers(0, Zn - dz_size + 1)
            oy = rng.integers(0, Yn - dy_size + 1)
            ox = rng.integers(0, Xn - dx_size + 1)

            patch = noise_map[
                oz:oz + dz_size,
                oy:oy + dy_size,
                ox:ox + dx_size
            ]

            out_roi = out[slc]

            for z in range(0, mask_roi.shape[0]):
                for y in range(0, mask_roi.shape[1]):
                    for x in range(0, mask_roi.shape[2]):
                        if(mask_roi[z,y,x] == True):
                            out_roi[z, y, x] = patch[z, y, x]

        return out

    def _generate_conc(self, dist, labels, rng):
        (z_slices, H, W) = labels.shape
        boundary = np.zeros_like(labels, dtype=bool)


        for z in range(labels.shape[0]):
            boundary[z] = find_boundaries(
                labels[z],
                mode="inner"
            )
        
        nuc_conc = self._gen_nuc_conc(labels,dist, rng)
        nonnuc_conc = self._gen_nonnuc_conc(labels, boundary, rng)
        return nuc_conc, nonnuc_conc
    
    def _gen_nuc_conc(self, labels, dist, rng):
        #nuclear concentration
        local_sz = 15  # neighborhood width
        vol_shape = labels.shape

        dist = np.nan_to_num(dist, nan=0, posinf=0, neginf=0)
        dist /= dist.max()

        local_ringmin = minimum_filter(dist, size=local_sz)
        local_ringmax = maximum_filter(dist, size=local_sz)

        ring_normal = (dist - local_ringmin) / (local_ringmax - local_ringmin + 1e-8)

        ring = ring_normal**rng.integers(2, 5)

        prenoise_conc = np.float32(dist!=0) + (rng.uniform(3, 6)*ring)

        #assigning noise to different regions
        big_noise_shape = (round(self.z_slices*1.5), round(self.N*1.5), round(self.N*1.5))

        bands = np.array([[25, 100], [15, 50], [8, 25], [10, 17], [12, 14], [7, 8], [3, 5], [0.8, 3], [0.6, 1]])
        noise_map = self._spectral_noise(big_noise_shape, bands, 0.3, 0, 1., rng)

        bands = np.array([[25, 150], [15, 100], [8, 50], [3, 25], [1, 10]])
        mod_noise = self._spectral_noise(big_noise_shape, bands, 0.4, 0.1, 1., rng)

        noise_map *= mod_noise

        objs = find_objects(labels)
        objs = [obj for obj in objs if obj is not None] #needed because sometimes there is a None in the list because obj couldn't be found.
        conc = self._assign_noise_to_labels(noise_map, objs, labels, rng)
        conc = prenoise_conc*conc


        conc = (conc - conc.min()) / (conc.max() - conc.min() + 1e-8)
        #conc_nonnuc = (conc_nonnuc - conc_nonnuc.min()) / (conc_nonnuc.max() - conc_nonnuc.min() + 1e-8)
        return conc

    def _gen_nonnuc_conc(self, labels, bdry, rng):
        #nonnuclear concentration
        vol_shape = labels.shape

        bands1 = np.array([[1, 100], [1, 50], [1, 20], [0.05, 10]])
        bands2 = np.array([[1, 100], [1, 50], [0.05, 20]])

        noise1 = self._spectral_noise(vol_shape, bands1, 1, 0, 2, rng)

        interior = np.zeros(vol_shape, dtype=np.bool)
        thresh = rng.integers(4, 15)
        for z in range(vol_shape[0]):
            interior[z] = dist_t(~bdry[z])*noise1[z] < thresh

        #interior = (interior - np.min(interior))/(np.max(interior) - np.min(interior) + 1e-4)

        cell_interiors = (labels != 0) ^ bdry
        nonnuc = interior & (~cell_interiors)

        noise2 = self._spectral_noise(vol_shape, bands2, 2, 0.4, 1.5, rng)
        #thicker boundaries
        rings = dist_t(~nonnuc)
        rings = (rings.max() - rings)/(rings.max() - rings.min() + 1e-4)
        rings *= cell_interiors + nonnuc
        rings *= noise2
        rings = rings**rng.integers(2, 4)

        conc_nonnuc = (rings - rings.min()) / (rings.max() - rings.min() + 1e-8)
        return conc_nonnuc

    def _processing(self, conc, conc_nonnuc, rng):
        num_iter_rl = rng.integers(1, 7)
        na1 = rng.uniform(0.6, 0.95)
        na2 = rng.uniform(0.6, 0.95)

        pz1 = rng.uniform(0, 2)
        pz2 = rng.uniform(0, 2)

        psf = psfm.vectorial_psf_centered(nz=15, dz=0.2, nx=31, dxy=0.1125,
                                        pz=pz1, wvl=0.461,
                                        params=dict(NA=na1, ni=1., ni0=1.0,
                                                    ns=1.40, tg=0, tg0=0))
        psf /= psf.sum()

        blurred = fftconvolve(conc, psf, mode="same")
        blurred = np.clip(blurred, 0, None)

        blurred_nonnuc = fftconvolve(conc_nonnuc, psf, mode="same")
        blurred_nonnuc = np.clip(conc_nonnuc, 0, None)


        photons = rng.uniform(100, 5500)
        photons_nonnuc = rng.uniform(100, 5500)

        bg_noise_lvl = rng.uniform(0, 0.2)
        bands = np.array([[5, 140], [0.05, 30], [0.01, 15], [0.005, 7]])
        backg = self._spectral_noise(conc.shape, bands, 0.7, 0, bg_noise_lvl, rng)
        blurred += backg

        backg_nn = self._spectral_noise(conc.shape, bands, 0.7, 0, 1, rng)
        blurred_nonnuc *= backg_nn

        lam = blurred * photons
        lam_nn = blurred_nonnuc * photons_nonnuc
        #print("lambda max:", lam.max())
        noisy = rng.poisson(blurred * photons) / photons
        noisy_nn = rng.poisson(blurred_nonnuc * photons_nonnuc) / photons_nonnuc
        #different psf
        psf = psfm.vectorial_psf_centered(nz=5, dz=0.2, nx=25, dxy=0.1125,
                                        pz=pz2, wvl=0.461,
                                        params=dict(NA=na2, ni=1., ni0=1.0,
                                                    ns=1.40, tg=0, tg0=0))
        psf /= psf.sum()


        recovered = richardson_lucy(
            noisy,
            psf,
            num_iter=num_iter_rl,
            clip=False
        )
        recovered_nn = richardson_lucy(
            noisy_nn,
            psf,
            num_iter=num_iter_rl,
            clip=False
        )

        #normalize
        scale = np.percentile(recovered, 99.9)
        recovered = np.clip(recovered / max(scale, 1e-8), 0, 1)

        scale_nn = np.percentile(recovered_nn, 99.9)
        recovered_nn = np.clip(recovered_nn / max(scale_nn, 1e-8), 0, 1)


        altered_img = self._random_bezier_transform(recovered, rng)
        altered_img = self._contrast(altered_img, rng)
        altered_img = self._brightness_scale(altered_img, rng)
        
        altered_img_nn = self._random_bezier_transform(recovered_nn, rng)
        altered_img_nn = self._contrast(altered_img_nn, rng)
        altered_img_nn = self._brightness_scale(altered_img_nn, rng)

        
        return altered_img, altered_img_nn

    def _random_bezier_transform(self, image, rng=None):
        if rng is None:
            rng = np.random.default_rng()

      
        x1 = rng.uniform(0.2, 0.5)
        y1 = rng.uniform(0.1, 0.6)
        x2 = rng.uniform(0.5, 0.8)
        y2 = rng.uniform(0.4, 0.9)

        p0 = np.array([0.0, 0.0])
        p1 = np.array([x1, y1])
        p2 = np.array([x2, y2])
        p3 = np.array([1.0, 1.0])

        x_curve, y_curve = self._bezier_curve(p0, p1, p2, p3)

        return np.interp(image, x_curve, y_curve)


    def _bezier_curve(self, p0, p1, p2, p3, n=256):
        t = np.linspace(0, 1, n)

        curve = (
            (1 - t)[:, None]**3 * p0
            + 3 * (1 - t)[:, None]**2 * t[:, None] * p1
            + 3 * (1 - t)[:, None] * t[:, None]**2 * p2
            + t[:, None]**3 * p3
        )

        return curve[:, 0], curve[:, 1]

    def _brightness_scale(self, image, rng):
        a = rng.uniform(0.5, 1.5)
        return image * a

    def _contrast(self, image, rng):
        c = rng.uniform(0.5, 1.5)
        mu = image.mean()
        return mu + c * (image - mu)