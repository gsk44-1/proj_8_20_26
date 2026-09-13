import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.ndimage import zoom, distance_transform_edt as dist_t
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

        labels, bdry, dist = self._generate_cell_tiles(rng)

        #concentration
        conc = self._generate_conc(dist, labels, bdry, rng)

        out_vol = self._processing(conc, rng)

        return out_vol, labels



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

    def _sample_fg_pts_density(self, mask, n, density, rng):

        coords = np.argwhere(mask)

        weights = density[mask].astype(float)
        weights /= weights.sum()

        idx = rng.choice(
            len(coords),
            size=n,
            replace=False,
            p=weights
        )

        return coords[idx]

    @staticmethod
    @njit(cache=True)
    def _anisotropic_geodesic_voronoi_numba(
        mask,
        seeds,
        theta,
        ratio,
        scale,
        z_ratio,
        R=None
    ):
        Z, H, W = mask.shape
        if R is None:
            R = np.full((Z, H, W), np.inf)

        dist = np.full((Z, H, W), np.inf)
        labels = np.zeros((Z, H, W), dtype=np.int32)


        #metric tensor
        a11 = np.empty((Z, H, W), dtype=np.float64)
        a12 = np.empty((Z, H, W), dtype=np.float64)
        a22 = np.empty((Z, H, W), dtype=np.float64)
        a33 = np.empty((Z, H, W), dtype=np.float64)#
        #other entries are zero

        for y in range(H):
            for x in range(W):

                c = math.cos(theta[0, y, x]) #this assumes for now that the theta and ratio fields are the same for all z slices
                s = math.sin(theta[0, y, x])

                r = ratio[0, y, x]
                q = 1.0 / (r * r)

                a11[0, y, x] = q * c * c + s * s #A = R(theta)(1/r^2 0 \\ 0 1)R(theta)
                a22[0, y, x] = q * s * s + c * c
                a12[0, y, x] = (q - 1.0) * c * s
                a33[0, y, x] = z_ratio**2

        #copy 0 level of grids to all z levels
        for i in range(Z):
            a11[i, :, :] = a11[0, :, :]
            a22[i, :, :] = a22[0, :, :]
            a12[i, :, :] = a12[0, :, :]
            a33[i, :, :] = a33[0, :, :]

        # Each heap item:
        # (distance, z, y, x, seed_label)

        # Numba needs the heap's element type to be inferable
        heap = [(0.0, 0, 0, 0, 0)]
        heapq.heappop(heap)

        # Initialize every seed
        for i in range(seeds.shape[0]):

            z = seeds[i, 0]
            y = seeds[i, 1]
            x = seeds[i, 2]

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
                    (0.0, z, y, x, label_id)
                )

        # 26 cell neighborhood
        dys = (-1, 1, 0, 0, -1, -1, 1, 1, -1, 1, 0, 0, -1, -1, 1, 1, -1, 1, 0, 0, -1, -1, 1, 1, 0, 0)
        dxs = (0, 0, -1, 1, -1, 1, -1, 1, 0, 0, -1, 1, -1, 1, -1, 1, 0, 0, -1, 1, -1, 1, -1, 1, 0, 0)
        dzs = (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1)


        while len(heap) > 0:

            current_dist, z, y, x, label_id = heapq.heappop(heap)

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

                # Average metric tensor across edge
                
                A11 = 0.5 * (
                    a11[z, y, x] + a11[zz, yy, xx]
                )

                A22 = 0.5 * (
                    a22[z, y, x] + a22[zz, yy, xx]
                )

                A12 = 0.5 * (
                    a12[z, y, x] + a12[zz, yy, xx]
                )

                A33 = 0.5 * (
                    a33[z, y, x] + a33[zz, yy, xx]
                )

                avg_scale = 0.5 * (scale[z,y,x] + scale[zz,yy,xx])
                #test - 
                '''
                A11 = a11[z, y, x]
                A22 = a22[z, y, x]
                A12 = a12[z, y, x]
                A33 = a33[z, y, x]
                '''

                step_cost = math.sqrt(
                    A11 * dx * dx
                    + 2.0 * A12 * dx * dy
                    + A22 * dy * dy
                    + A33 * dz * dz
                ) * avg_scale

                if not math.isfinite(step_cost):
                    print("bad step_cost:", step_cost)
                    print("A11,A12,A22,A33:", A11, A12, A22, A33)
                    print("dx,dy,dz:", dx, dy, dz)

                new_dist = current_dist + step_cost

                if (new_dist < dist[zz, yy, xx]) and (new_dist < R[zz, yy, xx]):

                    dist[zz, yy, xx] = new_dist
                    labels[zz, yy, xx] = label_id

                    heapq.heappush(
                        heap,
                        (new_dist, zz, yy, xx, label_id)
                    )

        return labels, dist

    @staticmethod
    @njit(cache=True)
    def _weighted_centroids(labels, density, n_seeds):

        Z, H, W = labels.shape

        sum_z = np.zeros(n_seeds, dtype=np.float64)
        sum_y = np.zeros(n_seeds, dtype=np.float64)
        sum_x = np.zeros(n_seeds, dtype=np.float64)
        sum_w = np.zeros(n_seeds, dtype=np.float64)
        for z in range(Z):
            for y in range(H):
                for x in range(W):

                    label = labels[z, y, x]

                    if label == 0:
                        continue

                    i = label - 1
                    w = density[z, y, x]

                    sum_z[i] += w * z
                    sum_y[i] += w * y
                    sum_x[i] += w * x
                    sum_w[i] += w

        centroids = np.zeros((n_seeds, 3), dtype=np.float64)

        for i in range(n_seeds):

            if sum_w[i] > 0:
                centroids[i, 0] = sum_z[i] / sum_w[i]
                centroids[i, 1] = sum_y[i] / sum_w[i]
                centroids[i, 2] = sum_x[i] / sum_w[i]

        return centroids, sum_w

    @staticmethod
    @njit(cache=True)
    def _lloyd_update(
        seeds,
        centroids,
        weights,
        alpha
    ):
        new_seeds = seeds.copy()

        for i in range(seeds.shape[0]):

            # Seed received a nonempty Voronoi region
            if weights[i] > 0:
                for d in range(seeds.shape[1]):
                    new_seeds[i, d] = (
                        (1.0 - alpha) * seeds[i, d]
                        + alpha * centroids[i, d]
                    )

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
        Z, H, W = mask.shape

        out = seeds.copy()

        for i in range(seeds.shape[0]):
            z = int(round(seeds[i, 0]))
            y = int(round(seeds[i, 1]))
            x = int(round(seeds[i, 2]))

            z = max(0, min(Z - 1, z))
            y = max(0, min(H - 1, y))
            x = max(0, min(W - 1, x))

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
        theta,
        ratio,
        scale,
        z_ratio,
        density=None,
        n_iter=3,
        alpha=0.4,
        R=None
    ):
        if R is None:
            R = np.full(mask.shape, np.inf)

        mask = np.asarray(mask, dtype=np.bool_)
        theta = np.asarray(theta, dtype=np.float64)
        ratio = np.asarray(ratio, dtype=np.float64)

        seeds = np.asarray(
            seeds,
            dtype=np.float64
        ).copy()

        if density is None:
            density = np.ones(
                mask.shape,
                dtype=np.float64
            )
        else:
            density = np.asarray(
                density,
                dtype=np.float64
            )

        # Precompute closest foreground pixel
        _, nearest = dist_t(
            ~mask,
            return_indices=True
        )

        nearest_z = nearest[0]
        nearest_y = nearest[1]
        nearest_x = nearest[2]

        for _ in range(n_iter):

            seed_pixels = np.rint(
                seeds
            ).astype(np.int64)

            labels, dist = self._anisotropic_geodesic_voronoi_numba(
                mask,
                seed_pixels,
                theta,
                ratio,
                scale,
                z_ratio,
                R
            )

            centroids, weights = self._weighted_centroids(
                labels,
                density,
                len(seeds)
            )

            seeds = self._lloyd_update(
                seeds,
                centroids,
                weights,
                alpha
            )

            seeds = self._project_to_foreground_numba(
                seeds,
                mask,
                nearest_z,
                nearest_y,
                nearest_x
            )

        seeds0_scaled = seeds.astype(float).copy()
        seeds0_scaled[:, 0] *= z_ratio

        #calculate R map
        tree = cKDTree(seeds0_scaled)

        dists, ind = tree.query(seeds0_scaled, k=2)

        dists_others = dists[:, 1:].mean(axis=1)

        zz, yy, xx = np.mgrid[:self.z_slices, :self.N, :self.N]

        locs = np.column_stack((
            (z_ratio * zz).ravel(),
            yy.ravel(),
            xx.ravel()
        ))

        _, nearest_seed = tree.query(locs, k=1)

        spacing_map = dists_others[nearest_seed].reshape(self.z_slices, self.N, self.N)
        spacing_map = np.minimum(spacing_map, 15)
        R = gaussian_filter(spacing_map, sigma=(0, 10, 10))

        # Final tessellation
        seed_pixels = np.rint(
            seeds
        ).astype(np.int64)

        labels, dist = self._anisotropic_geodesic_voronoi_numba(
            np.full((self.z_slices, self.N, self.N), 1),
            seed_pixels,
            theta,
            ratio,
            scale,
            z_ratio,
            R#\max cell radius
        )

        return seed_pixels, labels, dist


    def _generate_cell_tiles(self, rng):
        N = self.N
        z_slices = self.z_slices
        z_ratio = self.z_ratio
        n_seeds = self.n_seeds



        coarse = rng.random((N//42, N//42))

        zoom_tup = (N/coarse.shape[0] + 1, N/coarse.shape[1] + 1)

        # enlarge to roughly N x N
        surf = zoom(coarse, zoom=zoom_tup, order=3)
        surf = surf[:N, :N]  

        surf = (surf - surf.mean())/((np.abs(surf - surf.mean())).max())

        fg = (surf > -0.2)*(surf < 0.6)

        gy, gx = np.gradient(surf)
        
        ''' 
        grad_mag = np.sqrt(gx**2 + gy**2)
        grad_mag = (
            (grad_mag - grad_mag.min()) /
            (grad_mag.max() - grad_mag.min())
        )
        grad_mag = np.repeat(grad_mag[None, :, :], z_slices, axis=0)
        '''
        
        theta = np.arctan2(gy, gx)
        theta = np.repeat(theta[None, :, :], z_slices, axis=0)

        

        scale = np.ones((z_slices, N, N))
        dist_fg = dist_t(fg)
        density2 = (dist_fg + 1)

        #make 3d
        fg = np.repeat(fg[None, :, :], z_slices, axis=0)
        density2 = np.repeat(density2[None, :, :], z_slices, axis=0)


        
        x_pts = np.array([0, 3, 8, 20])
        y_pts = np.array([2, 4, 5, 1])

        f = PchipInterpolator(x_pts, y_pts)
        ratio = f(np.clip(dist_fg, 0, 20))

        ratio = np.repeat(ratio[None, :, :], z_slices, axis=0)*self._spectral_noise((z_slices, N, N), bands=np.array([[5,100],[3,50],[1, 24]]), bdwidth=0.3, lo=0.8, hi=1.5, rng=rng)


        seeds0 = self._sample_fg_pts_density(
            fg,
            n=n_seeds,
            density=density2,
            rng=rng
        )


        seeds0_scaled = seeds0.astype(float).copy()
        seeds0_scaled[:, 0] *= z_ratio

 
  

        seeds, labels, dist = self._anisotropic_lloyd_relaxation(
            fg,
            seeds0,
            theta,
            ratio,
            scale,
            z_ratio,
            density=None,
            n_iter=5,
            alpha=1.0,
            R=None,
        )

        boundary = np.zeros_like(labels, dtype=bool)

        for z in range(labels.shape[0]):
            boundary[z] = find_boundaries(
                labels[z],
                mode="inner"
            )

        bands = np.array([[5, 100], [5, 50], [0.1, 5]])

        #dist will be used for rings
        dist[np.isinf(dist)] = 0
        #dist = (dist - dist.min()) / (dist.max() - dist.min())

        return labels, boundary, dist

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

    def _generate_conc(self, dist, labels, bdry, rng):
        local_sz = 15  # neighborhood width
        z_slices = self.z_slices
        N = self.N
        vol_shape = (z_slices, N, N)


        ring = dist**6

        local_ringmin = minimum_filter(ring, size=local_sz)
        local_ringmax = maximum_filter(ring, size=local_sz)

        ring_normal = (ring - local_ringmin) / (local_ringmax - local_ringmin + 1e-8)

        bands = np.array([[1, 100], [4, 50], [1, 20], [0.3, 10], [0.1, 5]])
        ring_normal *= self._spectral_noise(vol_shape, bands, 0.5, 0.5, 2, rng)



        big_noise_shape = (round(z_slices*1.5), round(N*1.5), round(N*1.5))

        bands = np.array([[25, 100], [15, 50], [8, 25], [10, 17], [12, 14], [7, 8], [3, 5], [0.8, 3], [0.6, 1]])
        noise_map = self._spectral_noise(big_noise_shape, bands, 0.3, 0, 1., rng)

        bands = np.array([[25, 100], [15, 50], [8, 25]])
        mod_noise = self._spectral_noise(big_noise_shape, bands, 0.4, 0.1, 1., rng)

        noise_map *= mod_noise

        objs = find_objects(labels)

        conc = self._assign_noise_to_labels(noise_map, objs, labels, rng)
        conc = conc*(ring_normal+1.0) + (0.3*ring_normal)

        bands = np.array([[25, 100], [15, 50], [8, 25]])


        conc = conc*(1-(np.float32(bdry)*mod_noise[:z_slices, :N, :N]))

        conc = conc**2

        return conc

    def _processing(self, conc, rng):
        num_iter_rl = rng.integers(3, 7)
        na1 = rng.uniform(0.6, 0.95)
        na2 = rng.uniform(0.6, 0.95)

        pz1 = rng.uniform(-0.5, 0.5)
        pz2 = rng.uniform(-0.5, 0.5)

        psf = psfm.vectorial_psf_centered(nz=15, dz=0.2, nx=31, dxy=0.1125,
                                        pz=pz1, wvl=0.461,
                                        params=dict(NA=na1, ni=1., ni0=1.0,
                                                    ns=1.40, tg=0, tg0=0))
        psf /= psf.sum()

        blurred = fftconvolve(conc, psf, mode="same")

        blurred = (blurred - blurred.min())/(blurred.max() - blurred.min())
        photons = 1000
        noisy = rng.poisson(blurred * photons) / photons

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
        return recovered