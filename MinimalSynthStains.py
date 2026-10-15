        
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