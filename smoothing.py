"""
One-Euro filter: smooths shaky tracking *without* adding lag.
When your hand is almost still it smooths a lot (no jitter);
when your hand moves fast it smooths very little (no delay).
That is what makes the cursor feel "silky" instead of floaty.
"""
import math
import time


class _LowPass:
    def __init__(self):
        self.value = None

    def apply(self, x, alpha):
        self.value = x if self.value is None else alpha * x + (1 - alpha) * self.value
        return self.value


class OneEuro:
    def __init__(self, min_cutoff=1.2, beta=0.04, d_cutoff=1.0):
        self.min_cutoff = min_cutoff   # lower = smoother when still
        self.beta = beta               # higher = less lag when moving fast
        self.d_cutoff = d_cutoff
        self._x = _LowPass()
        self._dx = _LowPass()
        self._last_t = None
        self._last_raw = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t=None):
        t = time.time() if t is None else t
        if self._last_t is None:
            self._last_t, self._last_raw = t, x
            return self._x.apply(x, 1.0)
        dt = max(t - self._last_t, 1e-3)
        dx = (x - self._last_raw) / dt
        edx = self._dx.apply(dx, self._alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        out = self._x.apply(x, self._alpha(cutoff, dt))
        self._last_t, self._last_raw = t, x
        return out

    def reset(self):
        self.__init__(self.min_cutoff, self.beta, self.d_cutoff)


class OneEuro2D:
    """Same filter for an (x, y) point."""

    def __init__(self, **kw):
        self.fx, self.fy = OneEuro(**kw), OneEuro(**kw)

    def __call__(self, point, t=None):
        return self.fx(point[0], t), self.fy(point[1], t)

    def reset(self):
        self.fx.reset()
        self.fy.reset()
