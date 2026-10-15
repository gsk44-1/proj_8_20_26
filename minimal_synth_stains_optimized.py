"""Synthetic two-channel stain dataset (optimized active pipeline).

Requires numpy, scipy, scikit-image, numba, torch, and psfmodels.
"""
from functools import lru_cache
import heapq
import math
import numpy as np
from torch.utils.data import Dataset
from scipy.fft import rfftn, irfftn, fftfreq, rfftfreq
from scipy.ndimage import (zoom, distance_transform_edt as dist_t,
                           gaussian_filter, find_objects, minimum_filter,
                           maximum_filter)
from scipy.signal import fftconvolve
from skimage.segmentation import find_boundaries
from numba import njit


@lru_cache(maxsize=16)
def _radial_frequencies(shape):
    """Cached FFT grid, in cycles per voxel (last axis uses real FFT)."""
    z, y, x = shape
    kz = fftfreq(z).astype(np.float32)[:, None, None]
    ky = fftfreq(y).astype(np.float32)[None, :, None]
    kx = rfftfreq(x).astype(np.float32)[None, None, :]
    return np.sqrt(kz*kz + ky*ky + kx*kx)


class MinimalSynthStains(Dataset):
    def __init__(
            self,
            samples_per_epoch=10000,
            seed=0,
            N = 200,
            z_slices = 20,
            z_ratio = 1.5,
            n_seeds = 55,
            mask_sigma_xy = 1.5,
            fixed_psf = True
    ):
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.N = N
        self.z_slices = z_slices
        self.z_ratio = z_ratio
        self.n_seeds = n_seeds
        self.mask_sigma_xy = mask_sigma_xy
        self.fixed_psf = fixed_psf
        self._cached_psf = None  # Lazy per-worker initialization

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
        """Band-limited random field; real FFT + cached radial grid.

        Same frequency convention as the original (cycles/voxel). Uses
        float32 FFTs instead of NumPy's float64 FFTs for speed and memory.
        """
        if rng is None:
            rng = np.random.default_rng()
        shape = tuple(shape)
        white = rng.standard_normal(shape, dtype=np.float32)
        k = _radial_frequencies(shape)
        filt = np.zeros(k.shape, dtype=np.float32)
        for amp, wavelength in np.asarray(bands):
            center = 1.0 / wavelength
            width = bdwidth / wavelength
            filt += np.float32(amp) * np.exp(-0.5 * ((k - center) / width)**2)
        out = irfftn(rfftn(white) * filt, s=shape).real
        vmin, vmax = out.min(), out.max()
        if vmax <= vmin:
            return np.full(shape, (lo + hi) / 2, dtype=np.float32)
        out = (out - vmin) * ((hi - lo) / (vmax - vmin)) + lo
        return out.astype(np.float32, copy=False)

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
        """Create low-res cell instances, then full-res binary stain masks."""
        # Ceiling division permits arbitrary target dimensions; extra pixels
        # are cropped after upsampling rather than filled by extrapolation.
        N = (self.N + 3) // 4
        z_slices = (self.z_slices + 3) // 4
        low_shape = (z_slices, N, N)
        n_seeds = int(rng.integers(round(self.n_seeds * 0.6),
                                   round(self.n_seeds * 1.2)))
        n_seeds = min(n_seeds, np.prod(low_shape))

        full_mask = np.ones(low_shape, dtype=np.bool_)
        seeds = self._sample_fg_pts_density(full_mask, n_seeds, rng)
        ratios = rng.uniform(0.2, 20, size=n_seeds)
        thetas = rng.uniform(-np.pi/2, np.pi/2, size=n_seeds)
        band = np.array([[5, 100]], dtype=np.float32)
        noise_theta = self._spectral_noise(low_shape, band, 1., -0.1, 0.1, rng)
        noise_ratio = self._spectral_noise(low_shape, band, 1., 0.8, 1.05, rng)
        # Original endpoints were reversed: (5, 3.75). Their range is the
        # same, but specifying ascending endpoints changes noise polarity.
        R = self._spectral_noise(low_shape, band, 1., 3.75, 5.0, rng)

        labels, _ = self._anisotropic_geodesic_voronoi_numba(
            full_mask, seeds, ratios, thetas, noise_theta, noise_ratio, R
        )

        # Discrete labels must use nearest-neighbor interpolation; never
        # interpolate integer identities using cubic/linear interpolation.
        labels = zoom(labels, zoom=4, order=0, grid_mode=True,
                      mode="nearest")
        labels = self._match_shape(labels,
                                   (self.z_slices, self.N, self.N))

        # Generate the binary masks from the full-resolution instance map.
        boundary = np.zeros(labels.shape, dtype=bool)
        for z in range(labels.shape[0]):
            boundary[z] = find_boundaries(labels[z], mode="inner")

        foreground = (labels != 0) & ~boundary
        labels_bin = np.zeros_like(foreground)
        for z in range(foreground.shape[0]):
            labels_bin[z] = dist_t(foreground[z]) > 2

        # Retain the original 3D boundary distance calculation.
        nonnuc_labels_bin = (dist_t(~boundary) < 2) & ~labels_bin

        # Smooth only XY on the final binary masks, avoiding an expensive
        # per-voxel Python majority filter over categorical label IDs.
        if self.mask_sigma_xy > 0:
            sigma = (0, self.mask_sigma_xy, self.mask_sigma_xy)
            labels_bin = gaussian_filter(
                labels_bin.astype(np.float32), sigma=sigma) >= 0.5
            nonnuc_labels_bin = gaussian_filter(
                nonnuc_labels_bin.astype(np.float32), sigma=sigma) >= 0.5
        # Keep the two returned binary classes disjoint.
        nonnuc_labels_bin &= ~labels_bin
        return labels, labels_bin, nonnuc_labels_bin

    def _match_shape(self, arr, target_shape):
      # Crop dimensions that are too large
      arr = arr[tuple(slice(0, s) for s in target_shape)]

      # Zero-pad dimensions that are too small
      padding = [(0, max(0, s - n))
                for n, s in zip(arr.shape, target_shape)]

      return np.pad(arr, padding, mode='edge')

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

        base_conc = self._randomize_labels(cell_structures, 0.3, 1.6, rng)
        nuc_conc = (base_conc) * self._spectral_noise(nuc_labels_bin.shape, self._random_bands(rng), 1, 0.1, 1, rng)
        
        ridges = dist_t(nonnuc_labels_bin)
        #ridges = (ridges - ridges.min())/(ridges.max() - ridges.min() + 1e-4)
        inverted_ridges = dist_t(~nonnuc_labels_bin)
        inverted_ridges = np.max(inverted_ridges) - inverted_ridges
        #inverted_ridges = (inverted_ridges - inverted_ridges.min())/(inverted_ridges.max() - inverted_ridges.min() + 1e-4)
        inverted_ridges = inverted_ridges
        ridges += inverted_ridges
        #ridges = ridges**2
        ridges = (ridges - ridges.min())/(ridges.max() - ridges.min() + 1e-4)
        ridges = ridges**2

        nonnuc_conc = ridges * self._spectral_noise(nonnuc_labels_bin.shape, self._random_bands(rng), 1, 0.1, 1, rng)

        nuc_conc = (nuc_conc - nuc_conc.min())/(nuc_conc.max() - nuc_conc.min() + 1e-4)
        nonnuc_conc = (nonnuc_conc - nonnuc_conc.min())/(nonnuc_conc.max() - nonnuc_conc.min() + 1e-4)
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
            R = np.full((Z, H, W), np.inf, dtype=np.float32)

        dist = np.full((Z, H, W), np.inf)
        labels = np.zeros((Z, H, W), dtype=np.int32)

        # Each heap item:
        # (distance, z, y, x, seed_label)

        # Numba needs the heap's element type to be inferable
        heap = [(0.0, 0, 0, 0, 0., 0., 0)]
        heap.pop()  # Empty heap while retaining inferred Numba tuple type

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
                label_id = i + 1

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

            # Orientation and anisotropy are constant across the 26 outgoing
            # edges; compute their metric coefficients only once per pop.
            c = math.cos(theta)
            sn = math.sin(theta)
            q = 1.0 / (ratio * ratio)
            reg = 1e-3
            A11 = q*c*c + sn*sn + reg
            A12 = (q - 1.0)*c*sn
            A22 = q*sn*sn + c*c + reg
            A33 = 1.0 + reg

            for k in range(26):
                dz = dzs[k]
                dy = dys[k]
                dx = dxs[k]
                zz, yy, xx = z + dz, y + dy, x + dx
                if (zz < 0 or zz >= Z or yy < 0 or yy >= H
                        or xx < 0 or xx >= W or not mask[zz, yy, xx]):
                    continue

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

    def _randomize_labels(self, labels, lo=0.4, hi=1., rng=np.random.default_rng(0)):
        unique, inverse = np.unique(labels, return_inverse=True)
        values = rng.uniform(lo, hi, size=len(unique))

        result = values[inverse].reshape(labels.shape)
        result[labels==0] = 0.05
        return result
    
    def _random_bands(self, rng):
        num_bands = 6
        base_amps = [10, 15, 4, 2, 1, 0.1]
        bands = np.zeros((num_bands, 2))
        scale = 150

        for j in range(num_bands):
          bands[j, 0] = base_amps[j] * rng.uniform(0.3, 5)  #amplitude
          bands[j, 1] = scale
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

        bands = self._random_bands(rng)

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
        if self.fixed_psf and self._cached_psf is not None:
            psf = self._cached_psf
        else:
            # Lazy import makes geometry-only usage possible without psfmodels.
            import psfmodels as psfm
            na = 0.875 if self.fixed_psf else rng.uniform(0.75, 1.0)
            pz = 1.0 if self.fixed_psf else rng.uniform(0, 2)
            psf = psfm.vectorial_psf_centered(
                nz=15, dz=0.2, nx=31, dxy=0.1125,
                pz=pz, wvl=0.461,
                params=dict(NA=na, ni=1., ni0=1.0,
                            ns=1.40, tg=0, tg0=0)
            )
            psf = psf / psf.sum()
            if self.fixed_psf:
                self._cached_psf = psf

        blurred_n = fftconvolve(conc_n, psf, mode="same")
        blurred_n = np.clip(blurred_n, 0, None)

        blurred_nonnuc = fftconvolve(conc_nonnuc, psf, mode="same")
        blurred_nonnuc = np.clip(blurred_nonnuc, 0, None)


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

        return (altered_img_n.astype(np.float32, copy=False),
                altered_img_nn.astype(np.float32, copy=False))

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