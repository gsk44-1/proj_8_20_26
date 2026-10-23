from torch.utils.data import Dataset
import numpy as np
from scipy.ndimage import zoom, distance_transform_edt as dist_t, binary_dilation
import heapq
import math
from skimage.segmentation import find_boundaries
from scipy.ndimage import gaussian_filter, find_objects, minimum_filter, maximum_filter
from scipy.signal import fftconvolve
from scipy import fft as spfft
from scipy.spatial import cKDTree
from functools import lru_cache
import psfmodels as psfm


from numba import njit


class SyntheticFluorStains(Dataset):
    def __init__(
            self,
            samples_per_epoch=10000,
            seed=0,
            N = 200,
            z_slices = 20,
            z_ratio = 1.5,
            n_seeds = 55,
            xy_downsample = 2,
            boundary_smoothing_sigma = 0.8,
    ):
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.N = N
        self.z_slices = z_slices
        self.z_ratio = z_ratio
        self.n_seeds = n_seeds
        self.xy_downsample = int(xy_downsample)
        self.boundary_smoothing_sigma = float(boundary_smoothing_sigma)
        if self.xy_downsample < 1 or N % self.xy_downsample:
            raise ValueError("xy_downsample must be a positive divisor of N")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed + idx)

        labels, nuc_labels, nonnuc_labels, dist_labels_bin_r = self._generate_cell_tiles(rng)

        #concentration
        conc, conc_nonnuc = self._generate_conc(dist_labels_bin_r, labels, nonnuc_labels, rng)

        out_vol, out_nonnuc_vol = self._processing(conc, conc_nonnuc, rng)
        data = np.stack([out_vol, out_nonnuc_vol], axis=1)


        #structure = disk(5)

        #for j in range(self.z_slices):
        #  nonnuc[j] = nonnuc[j] & ~binary_erosion(nonnuc[j], structure=structure)

        labels = np.stack([nuc_labels, nonnuc_labels], axis=1)
        #print(f" marker shape {markers.shape}")
        return data, labels



    @staticmethod
    @lru_cache(maxsize=8)
    def _radial_frequency(shape):
        """Cached FFT coordinates (one copy per shape, per worker)."""
        kz = spfft.fftfreq(shape[0])[:, None, None]
        ky = spfft.fftfreq(shape[1])[None, :, None]
        kx = spfft.rfftfreq(shape[2])[None, None, :]
        return np.sqrt(kx * kx + ky * ky + kz * kz).astype(np.float32)

    def _spectral_noise(self, shape, bands, bdwidth=0.3, lo=0, hi=1, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        white = rng.standard_normal(shape).astype(np.float32)
        k = self._radial_frequency(tuple(shape))
        filt = np.zeros_like(k)
        for amplitude, wavelength in bands:
            filt += amplitude * np.exp(-0.5 * ((k - 1.0 / wavelength) /
                                                 (bdwidth / wavelength)) ** 2)
        out = spfft.irfftn(spfft.rfftn(white) * filt, s=shape).real
        vmin, vmax = out.min(), out.max()
        if vmax <= vmin:
            return np.full(shape, (lo + hi) / 2, dtype=np.float32)
        return ((hi - lo) * ((out - vmin) / (vmax - vmin)) + lo).astype(np.float32)

    @staticmethod
    def _upsample_labels(labels_coarse, scale, sigma=0.8):
        """Nearest-neighbor upscale with optional local boundary-only smoothing.

        Smooth one-hot membership, NEVER numeric label values. Background is retained.
        """
        if scale == 1:
            return labels_coarse.copy()
        labels = np.repeat(np.repeat(labels_coarse, scale, axis=1), scale, axis=2)
        if sigma <= 0:
            return labels
        # Only change voxels near interfaces, and only use labels occurring nearby.
        from scipy.ndimage import find_objects, gaussian_filter
        band = binary_dilation(find_boundaries(labels, mode="thick"), iterations=2)
        winner = labels.copy()
        best = np.zeros_like(labels, dtype=np.float32)
        for lab, roi in enumerate(find_objects(labels), 1):
            if roi is None:
                continue
            # Extend each bounding box to allow competing labels at its edge.
            pad = max(4, int(np.ceil(4 * sigma)))
            sl = tuple(slice(max(0, a.start-pad), min(dim, a.stop+pad))
                       for a, dim in zip(roi, labels.shape))
            mask = (labels[sl] == lab).astype(np.float32)
            score = gaussian_filter(mask, sigma=(0, sigma, sigma), mode="constant")
            relevant = band[sl] & (labels[sl] != 0)
            update = relevant & (score > best[sl])
            winner[sl][update] = lab
            best[sl][update] = score[update]
        return winner

    def _sample_fg_pts_density(self, mask, n, rng):

        coords = np.argwhere(mask)


        idx = rng.choice(
            len(coords),
            size=n,
            replace=False
        )

        return coords[idx]
    
    def _smooth_seed_axial_angles(self, seeds, thetas, alignment,
                                  k=8, sigma=40.0):
        """Spatially varying neighbor smoothing for unoriented XY axes.

        All angles are modulo pi. ``alignment`` is one value per seed in
        [0, 1]: zero preserves that seed's initial angle, and one uses
        the Gaussian-weighted axial mean of its k nearest seed neighbors.
        Distances use native XY pixel coordinates and z_ratio for Z.
        """
        coords = seeds.astype(np.float64).copy()
        coords[:, 0] *= self.z_ratio
        coords[:, 1:] *= self.xy_downsample
        tree = cKDTree(coords)
        distances, indices = tree.query(coords, k=min(k, len(seeds)), workers=-1)
        distances = np.asarray(distances).reshape(len(seeds), -1)
        indices = np.asarray(indices).reshape(len(seeds), -1)

        original = np.exp(2j * thetas)
        weights = np.exp(-0.5 * (distances / sigma) ** 2)
        neighbor_mean = np.sum(weights * original[indices], axis=1)
        # If a neighborhood has no preferred axis, keep its own angle.
        magnitude = np.abs(neighbor_mean)
        neighbor_direction = np.where(
            magnitude > 1e-12,
            neighbor_mean / np.maximum(magnitude, 1e-12),
            original,
        )
        combined = ((1.0 - alignment) * original
                    + alignment * neighbor_direction)
        combined = np.where(np.abs(combined) > 1e-12, combined, original)
        return (0.5 * np.angle(combined)) % np.pi

    def _generate_cell_tiles(self, rng):
        N = self.N
        z_slices = self.z_slices
        scale = self.xy_downsample
        shape = (z_slices, N, N)
        n_seeds = rng.integers(round(self.n_seeds * 0.6), round(self.n_seeds * 1.2))
        surf_field = self._spectral_noise(shape, np.array([[5, 150], [0.5, 50]]),
                                          0.9, -1, 1, rng)
        surf = (surf_field > -0.3) & (surf_field < 0.3)
        surf_coarse = surf[:, ::scale, ::scale]
        seeds = self._sample_fg_pts_density(surf_coarse, n_seeds, rng)
        ratios = rng.uniform(1., 20., size=n_seeds)
        # Axial orientations: theta and theta + pi describe the same axis.
        thetas = rng.uniform(0.0, np.pi, size=n_seeds)

        # Correlated alignment strength varies across space from 0 to 1.
        # Sample it only where seeds exist, then use it to blend each seed's
        # own random axis with the circular mean of nearby seed axes.
        alignment_field = self._spectral_noise(
            shape, np.array([[5, 100], [1, 50]]), 0.8, 0.0, 1.0, rng
        )
        alignment = alignment_field[
            seeds[:, 0], seeds[:, 1] * scale, seeds[:, 2] * scale
        ]
        thetas = self._smooth_seed_axial_angles(
            seeds, thetas, alignment, k=4, sigma=40.0
        )

        # Create correlated fields at native resolution; sample at coarse nodes.
        theta = self._spectral_noise(shape, np.array([[5, 100]]), 1., -0.1, 0.1, rng)
        ratio = self._spectral_noise(shape, np.array([[5, 100]]), 1., 0.8, 1.05, rng)
        radius = self._spectral_noise(shape, np.array([[5, 100]]), 1., 10, 15, rng)
        # The original model propagates across the entire image, not only surf.
        # XY steps on the coarse grid have physical length 'scale'.
        coarse_labels, coarse_dist = self._anisotropic_geodesic_voronoi_numba(
            np.ones(surf_coarse.shape, dtype=np.bool_), seeds, ratios, thetas,
            theta[:, ::scale, ::scale], ratio[:, ::scale, ::scale],
            radius[:, ::scale, ::scale], float(scale)
        )
        labels = self._upsample_labels(coarse_labels, scale,
                                       self.boundary_smoothing_sigma)
        # Interpolate coarse geodesic arrival cost to the fine image grid.
        # Infinite/unreached regions remain infinite; the concentration step
        # maps infinities to zero, as in the baseline.
        finite = np.isfinite(coarse_dist)
        dist = zoom(np.where(finite, coarse_dist, 0).astype(np.float32),
                    (1, scale, scale), order=1)
        reached = np.repeat(np.repeat(finite, scale, axis=1), scale, axis=2)
        dist[~reached] = np.inf

        boundary = np.zeros_like(labels, dtype=bool)


        for z in range(labels.shape[0]):
            boundary[z] = find_boundaries(
                labels[z],
                mode="inner"
            )

        labels_bin = (labels != 0) & ~boundary

        dist_labels_bin_r = np.zeros(labels_bin.shape, dtype=float)

        for z in range(labels_bin.shape[0]):
            dist_labels_bin_r[z] = dist_t(labels_bin[z])

        labels_bin = dist_labels_bin_r > 2

        nonnuc_surround = (dist_t(~surf) < 5) & (~labels_bin)
        nonnuc_labels_bin = dist_t(~boundary) < 3
        nonnuc_labels_bin = nonnuc_labels_bin | nonnuc_surround


        return labels, labels_bin, nonnuc_labels_bin, dist

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

    def _generate_conc(self, dist, labels, nonnuc_labels, rng):
        (z_slices, H, W) = labels.shape
        boundary = np.zeros_like(labels, dtype=bool)


        for z in range(labels.shape[0]):
            boundary[z] = find_boundaries(
                labels[z],
                mode="inner"
            )
        
        nuc_conc = self._gen_nuc_conc(labels, dist, boundary, rng)
        nonnuc_conc = self._gen_nonnuc_conc(labels, boundary, nonnuc_labels, rng)
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
        R=None,
        xy_scale=1.0
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
        heap.pop()

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

            c = math.cos(theta)
            sn = math.sin(theta)
            q = 1.0 / (ratio * ratio)
            reg = 1e-3
            A11 = q * c * c + sn * sn + reg
            A12 = (q - 1.0) * c * sn
            A22 = q * sn * sn + c * c + reg
            A33 = 1.0 + reg

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

                sx = dx * xy_scale
                sy = dy * xy_scale
                step_cost = math.sqrt(
                    A11 * sx * sx + 2.0 * A12 * sx * sy
                    + A22 * sy * sy + A33 * dz * dz
                )

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
        dist /= max(dist.max(), 1e-8)

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
        pz1 = rng.uniform(0, 2)

        psf = psfm.vectorial_psf_centered(nz=15, dz=0.2, nx=31, dxy=0.1125,
                                        pz=pz1, wvl=0.461,
                                        params=dict(NA=na1, ni=1., ni0=1.0,
                                                    ns=1.40, tg=0, tg0=0))
        psf /= psf.sum()

        blurred_n = fftconvolve(conc_n, psf, mode="same")
        blurred_n = np.clip(blurred_n, 0, None)

        blurred_nonnuc = fftconvolve(conc_nonnuc, psf, mode="same")
        blurred_nonnuc = np.clip(blurred_nonnuc, 0, None)


        photons = rng.uniform(100, 5500)
        photons_nonnuc = rng.uniform(100, 5500)

        noisy_n = rng.poisson(blurred_n * photons) / photons
        noisy_nn = rng.poisson(blurred_nonnuc * photons_nonnuc) / photons_nonnuc
        #normalize
        scale = np.percentile(noisy_n, 99.9)
        noisy_n = np.clip(noisy_n / max(scale, 1e-8), 0, 1)

        scale_nn = np.percentile(noisy_nn, 99.9)
        noisy_nn = np.clip(noisy_nn / max(scale_nn, 1e-8), 0, 1)

        altered_img_n = self._random_bezier_transform(noisy_n, rng)
        altered_img_n = self._contrast(altered_img_n, rng)
        altered_img_n = self._brightness_scale(altered_img_n, rng)
        
        altered_img_nn = self._random_bezier_transform(noisy_nn, rng)
        altered_img_nn = self._contrast(altered_img_nn, rng)
        altered_img_nn = self._brightness_scale(altered_img_nn, rng)

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