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


class MinimalSynthStains(Dataset):
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

        cell_structures, nuc_labels, nonnuc_labels = self._generate_cell_tiles(rng)

        #concentration
        conc, conc_nonnuc = self._generate_conc(cell_structures, nuc_labels, nonnuc_labels, rng)

        out_vol, out_nonnuc_vol = self._processing(conc, conc_nonnuc, rng)
        data = np.stack([out_vol, out_nonnuc_vol], axis=1)

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
        #do on smaller grid then upsample?
        N = self.N

        z_slices = self.z_slices
        z_ratio = self.z_ratio
        n_seeds = rng.integers(round(self.n_seeds*0.6), round(self.n_seeds*1.2))

        full_mask = np.ones((z_slices, N, N))
        seeds = self._sample_fg_pts_density(full_mask, n_seeds, rng)
        ratios = rng.uniform(0.2, 20, size=(n_seeds))
        thetas = rng.uniform(-np.pi/2, np.pi/2, size=(n_seeds))
        #now these are just lists of values for each seed

        noise_theta = self._spectral_noise((z_slices, N, N), np.array([[5, 100]]), 1., -0.1, 0.1, rng)
        noise_ratio = self._spectral_noise((z_slices, N, N), np.array([[5, 100]]), 1., 0.8, 1.05, rng) #ratio movement should be biased negative?
        R = self._spectral_noise((z_slices, N, N), np.array([[5, 100]]), 1., 10, 15, rng)





        seeds, labels, dist = self._anisotropic_geodesic_voronoi_numba(full_mask, seeds, ratios, thetas, noise_theta, noise_ratio, 5, 1.0, R)

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

        nonnuc_labels_bin = (dist_t(~boundary) < 4) & (~labels_bin)
        
        return labels, labels_bin, nonnuc_labels_bin

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

    def _generate_conc(self, cell_structures, nuc_labels_bin, nonnuc_labels_bin, rng):
        (z_slices, H, W) = cell_structures.shape

        
        nuc_conc = (nuc_labels_bin!=0) * self._spectral_noise(nuc_labels_bin.shape, self._random_bands(5, rng), 1, 0.1, 1, rng)
        nonnuc_conc = (nonnuc_labels_bin!=0) * self._spectral_noise(nonnuc_labels_bin.shape, self._random_bands(5, rng), 1, 0.1, 1, rng)
        return nuc_conc, nonnuc_conc

    @staticmethod
    @njit(cache=True)
    def _anisotropic_geodesic_voronoi_numba(
        mask,
        seeds,
        ratios,
        thetas,
        noise_theta,
        noise_ratio,
        R=None
    ):
        Z, H, W = mask.shape
        if R is None:
            R = np.full((Z, H, W), np.inf)

        dist = np.full((Z, H, W), np.inf)
        labels = np.zeros((Z, H, W), dtype=np.int32)

        ratios_f = np.zeros((Z, H, W))
        thetas_f = np.ones((Z, H, W))
        # Each heap item:
        # (distance, z, y, x, seed_label)

        # Numba needs the heap's element type to be inferable
        heap = [(0.0, 0, 0, 0, 0., 0., 0)]
        heapq.heappop(heap)

        # Initialize every seed
        for i in range(seeds.shape[0]):

            z = seeds[i, 0]
            y = seeds[i, 1]
            x = seeds[i, 2]
            #print(f"location: {x}, {y}, {z}")

            if (
                y >= 0 and y < H
                and x >= 0 and x < W
                and z >= 0 and z < Z
                and mask[z, y, x]
            ):
                label_id = i

                dist[z, y, x] = 0.0
                labels[z, y, x] = label_id



                heapq.heappush(
                    heap,
                    (0.0, z, y, x, thetas[i], ratios[i], label_id)
                )

        # 26 cell neighborhood
        dys = (-1, 1, 0, 0, -1, -1, 1, 1, -1, 1, 0, 0, -1, -1, 1, 1, -1, 1, 0, 0, -1, -1, 1, 1, 0, 0)
        dxs = (0, 0, -1, 1, -1, 1, -1, 1, 0, 0, -1, 1, -1, 1, -1, 1, 0, 0, -1, 1, -1, 1, -1, 1, 0, 0)
        dzs = (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1)


        while len(heap) > 0:

            current_dist, z, y, x, theta, ratio, label_id = heapq.heappop(heap)

            # Ignore stale heap entry
            if current_dist > dist[z, y, x]:
                continue

            # Important if another seed reached this pixel first
            if labels[z, y, x] != label_id:
                continue

            for k in range(26):
                dz = dzs[k]
                dy = dys[k]
                dx = dxs[k]

                zz = z + dz
                yy = y + dy
                xx = x + dx

                if (
                    yy < 0 or yy >= H
                    or xx < 0 or xx >= W
                    or zz < 0 or zz >= Z
                    or not mask[zz, yy, xx]
                ):
                    continue

                c = math.cos(theta) #this assumes for now that the theta and ratio fields are the same for all z slices
                s = math.sin(theta)

                r = ratio
                q = 1.0 / (r * r)
                #we use metric at parent
                reg = 1e-3
                #a11
                A11 = q * c * c + s * s + reg

                #a12
                A12 = (q - 1.0) * c * s

                #a22
                A22 = q * s * s + c * c + reg

                #a33
                A33 = 1 + reg#z_ratio**2



                step_cost = math.sqrt(
                    A11 * dx * dx
                    + 2.0 * A12 * dx * dy
                    + A22 * dy * dy
                    + A33 * dz * dz
                )

                if not math.isfinite(step_cost):
                    print("bad step_cost:", step_cost)
                    print("A11,A12,A22,A33:", A11, A12, A22, A33)
                    print("dx,dy,dz:", dx, dy, dz)

                new_dist = (current_dist) + (step_cost)

                if (new_dist < dist[zz, yy, xx]) and (new_dist < R[zz, yy, xx]):

                    dist[zz, yy, xx] = new_dist
                    labels[zz, yy, xx] = label_id
                    delt_theta = noise_theta[zz, yy, xx]
                    delt_ratio = noise_ratio[zz, yy, xx]
                    heapq.heappush(
                        heap,
                        (new_dist, zz, yy, xx, theta+delt_theta, ratio*delt_ratio, label_id)
                    )

        return labels, dist

    def _randomize_labels(self, labels, rng):
        unique, inverse = np.unique(labels, return_inverse=True)
        values = rng.uniform(0.4, 0.8, size=len(unique))
        return values[inverse].reshape(labels.shape)
    
    def _random_bands(self, num_bands, rng):
        bands = np.zeros((num_bands, 2))
        amp = 10
        scale = 100
        for j in range(num_bands):
          bands[j, 0] = amp  #amplitude
          bands[j, 1] = scale

          amp = (amp*(0.5**j)) * rng.uniform(0.3, 5)
          scale /= 2
        return bands

    def _gen_nuc_conc(self, labels, dist, boundary, rng):
        #nuclear concentration
        local_sz = 15  # neighborhood width
        vol_shape = labels.shape

        dist = np.nan_to_num(dist, nan=0, posinf=0, neginf=0)
        dist /= dist.max()

        local_ringmin = minimum_filter(dist, size=local_sz)
        local_ringmax = maximum_filter(dist, size=local_sz)

        ring_normal = (dist - local_ringmin) / (local_ringmax - local_ringmin + 1e-8)

        ring = ring_normal**rng.integers(2, 6)

        prenoise_conc = self._randomize_labels(labels, rng) #random background constant for each nuclei
        prenoise_conc *= (1 + (rng.uniform(2, 3)*ring))
        degraded_bdry = boundary * self._spectral_noise(vol_shape, np.array([[3, 100], [1, 50], [0.2, 25]]), 0.7, 0, 1., rng)
        
        prenoise_conc = prenoise_conc * (1 - degraded_bdry)

        #assigning noise to different regions
        big_noise_shape = (round(self.z_slices*1.5), round(self.N*1.5), round(self.N*1.5))

        bands = self._random_bands(6, rng)

        noise_map = self._spectral_noise(big_noise_shape, bands, 0.4, 0, 1., rng)

        bands = np.array([[25, 150], [15, 100], [8, 50], [3, 25], [1, 10]])
        mod_noise = self._spectral_noise(big_noise_shape, bands, 0.5, 0.1, 1., rng)

        noise_map *= mod_noise
      

        objs = find_objects(labels)
        objs = [obj for obj in objs if obj is not None] #needed because sometimes there is a None in the list because obj couldn't be found.
        conc = self._assign_noise_to_labels(noise_map, objs, labels, rng)
        conc = prenoise_conc*conc


        conc = (conc - conc.min()) / (conc.max() - conc.min() + 1e-8)
        conc = conc**rng.integers(1, 4)
        #conc_nonnuc = (conc_nonnuc - conc_nonnuc.min()) / (conc_nonnuc.max() - conc_nonnuc.min() + 1e-8)
        return conc

    def _gen_nonnuc_conc(self, labels, bdry, nonnuc_labels, rng):
        #print("nonnuc")
        #nonnuclear concentration
        vol_shape = labels.shape

        #bands1 = np.array([[1, 100], [1, 50], [1, 20], [0.05, 10]])
        bands2 = np.array([[1, 100], [1, 50], [0.05, 20]])

        bands1 = self._random_bands(6, rng)
        noise1 = self._spectral_noise(vol_shape, bands1, 1, 0, 2, rng)
        
        #interior = (interior - np.min(interior))/(np.max(interior) - np.min(interior) + 1e-4)

        cell_interiors = (labels != 0) ^ bdry
        
        nonnuc = nonnuc_labels & (~cell_interiors)
        
        
        noise2 = self._spectral_noise(vol_shape, bands2, 2, 0.4, 1.5, rng)
        
        rings = dist_t(~nonnuc)
        rings = (rings.max() - rings)/(rings.max() - rings.min() + 1e-4)
        
        local_sz = 25
        local_ringmin = minimum_filter(rings, size=local_sz)
        local_ringmax = maximum_filter(rings, size=local_sz)

        rings = (rings - local_ringmin) / (local_ringmax - local_ringmin + 1e-8)


        rings *= cell_interiors + nonnuc
        rings = gaussian_filter(rings, 2, mode="reflect")
        rings += self._spectral_noise(vol_shape, bands2, 2, 0.0, 0.3, rng)
        rings *= noise2
        rings = rings**rng.integers(2, 4)


        conc_nonnuc = (rings - rings.min()) / (rings.max() - rings.min() + 1e-8)
        return conc_nonnuc

    def _processing(self, conc_n, conc_nonnuc, rng):
        na1 = rng.uniform(0.75, 1.0)
        na2 = rng.uniform(0.75, 1.0)

        pz1 = rng.uniform(0, 2)
        pz2 = rng.uniform(0, 2)

        psf = psfm.vectorial_psf_centered(nz=15, dz=0.2, nx=31, dxy=0.1125,
                                        pz=pz1, wvl=0.461,
                                        params=dict(NA=na1, ni=1., ni0=1.0,
                                                    ns=1.40, tg=0, tg0=0))
        psf /= psf.sum()

        blurred_n = fftconvolve(conc_n, psf, mode="same")
        blurred_n = np.clip(blurred_n, 0, None)

        blurred_nonnuc = fftconvolve(conc_nonnuc, psf, mode="same")
        blurred_nonnuc = np.clip(conc_nonnuc, 0, None)


        photons = rng.uniform(100, 5500)
        photons_nonnuc = rng.uniform(100, 5500)


        noisy_n = rng.poisson(blurred_n * photons) / photons
        noisy_nn = rng.poisson(blurred_nonnuc * photons_nonnuc) / photons_nonnuc
        #different psf


        #normalize
        scale = np.percentile(noisy_n, 99.9)
        noisy_n = np.clip(noisy_n / max(scale, 1e-8), 0, 1)

        scale_nn = np.percentile(noisy_nn, 99.9)
        noisy_nn = np.clip(noisy_nn / max(scale_nn, 1e-8), 0, 1)

        altered_img_n = self._random_bezier_transform(noisy_n, rng)

        
        altered_img_nn = self._random_bezier_transform(noisy_nn, rng)

        return altered_img_n, altered_img_nn

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