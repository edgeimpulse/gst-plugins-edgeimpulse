//! Scalar constant-velocity (CV) Kalman filter — one per box coordinate.
//!
//! State is `[position, velocity]` with velocity in units per second. The
//! covariance is a symmetric 2×2 matrix stored as three scalars (`p00`, `p01`,
//! `p11`); all predict/update math is closed form, so the tracker needs no
//! linear-algebra crate. Pure and host-testable.

/// Process/measurement noise for the scalar CV Kalman filters, derived from the
/// single `box-responsiveness` knob (0..=1).
#[derive(Clone, Copy, Debug)]
pub struct KalmanConfig {
    /// Continuous white-noise-acceleration PSD (process-noise scale). Larger ⇒
    /// the model is trusted less and measurements more ⇒ faster tracking, less
    /// smoothing.
    pub q_c: f32,
    /// Measurement-noise variance (fixed; smoothing comes from the q_c/r ratio).
    pub r: f32,
}

impl KalmanConfig {
    /// Map a `box-responsiveness` knob in `0.0..=1.0` to noise parameters. `r`
    /// is fixed; `q_c` sweeps geometrically so the knob feels perceptually even
    /// (1.0 = fast / less smooth, 0.0 = very smooth / more lag).
    ///
    /// Starting constants — tuned by feel in the Task 6 webcam smoke test.
    pub fn from_responsiveness(responsiveness: f32) -> Self {
        const R: f32 = 4.0; // ~2px measurement std
        const Q_MIN: f32 = 10.0;
        const Q_MAX: f32 = 100_000.0;
        let k = responsiveness.clamp(0.0, 1.0);
        let q_c = Q_MIN * (Q_MAX / Q_MIN).powf(k);
        Self { q_c, r: R }
    }
}

/// A scalar constant-velocity Kalman filter tracking one coordinate.
#[derive(Clone, Copy, Debug)]
pub struct Kalman1D {
    p: f32,   // position estimate
    v: f32,   // velocity estimate (units/second)
    p00: f32, // Cov(p, p)
    p01: f32, // Cov(p, v) == Cov(v, p) (symmetry preserved by construction)
    p11: f32, // Cov(v, v)
    q_c: f32,
    r: f32,
}

impl Kalman1D {
    /// Initialize at a measured position with zero velocity and large initial
    /// covariance so the first updates move the estimate quickly.
    pub fn new(measurement: f32, cfg: KalmanConfig) -> Self {
        // A positive measurement variance keeps the innovation covariance
        // `s = p00 + r` strictly positive, so `update` can never divide by zero.
        // `from_responsiveness` always satisfies this; the assert guards direct
        // `KalmanConfig` construction in debug builds.
        debug_assert!(cfg.r > 0.0, "measurement variance r must be positive");
        const INIT_VAR: f32 = 1_000.0;
        Self {
            p: measurement,
            v: 0.0,
            p00: INIT_VAR,
            p01: 0.0,
            p11: INIT_VAR,
            q_c: cfg.q_c,
            r: cfg.r,
        }
    }

    /// Current position estimate.
    pub fn position(&self) -> f32 {
        self.p
    }

    /// Current velocity estimate (units/second).
    pub fn velocity(&self) -> f32 {
        self.v
    }

    /// Predict the state forward by `dt` seconds (no measurement). `dt <= 0` is
    /// a no-op.
    pub fn predict(&mut self, dt: f32) {
        if dt <= 0.0 {
            return;
        }
        // x' = F x, F = [[1, dt], [0, 1]]
        self.p += self.v * dt;
        // P' = F P Fᵀ + Q (symmetric closed form)
        let dt2 = dt * dt;
        let p00 = self.p00 + 2.0 * dt * self.p01 + dt2 * self.p11;
        let p01 = self.p01 + dt * self.p11;
        let p11 = self.p11;
        // Continuous white-noise-acceleration process noise.
        let dt3 = dt2 * dt;
        self.p00 = p00 + self.q_c * dt3 / 3.0;
        self.p01 = p01 + self.q_c * dt2 / 2.0;
        self.p11 = p11 + self.q_c * dt;
    }

    /// Fuse a scalar measurement `z` at the current time.
    pub fn update(&mut self, z: f32) {
        // S = H P Hᵀ + R = p00 + r ; H = [1, 0]
        let s = self.p00 + self.r;
        let k0 = self.p00 / s; // Kalman gain for position
        let k1 = self.p01 / s; // Kalman gain for velocity
        let y = z - self.p; // innovation
        self.p += k0 * y;
        self.v += k1 * y;
        // P = (I - K H) P (symmetry preserved for symmetric input)
        let p00 = (1.0 - k0) * self.p00;
        let p01 = (1.0 - k0) * self.p01;
        let p11 = self.p11 - k1 * self.p01;
        self.p00 = p00;
        self.p01 = p01;
        self.p11 = p11;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(responsiveness: f32) -> KalmanConfig {
        KalmanConfig::from_responsiveness(responsiveness)
    }

    #[test]
    fn new_reports_measurement_and_zero_velocity() {
        let f = Kalman1D::new(5.0, cfg(0.5));
        assert_eq!(f.position(), 5.0);
        assert_eq!(f.velocity(), 0.0);
    }

    #[test]
    fn converges_to_static_measurement() {
        let mut f = Kalman1D::new(0.0, cfg(0.5));
        for _ in 0..30 {
            f.predict(1.0);
            f.update(10.0);
        }
        assert!((f.position() - 10.0).abs() < 0.5, "pos {}", f.position());
        assert!(f.velocity().abs() < 0.5, "vel {}", f.velocity());
    }

    #[test]
    fn estimates_constant_velocity() {
        // Target moves +4 units per step at dt=1 ⇒ true velocity 4/s.
        let mut f = Kalman1D::new(0.0, cfg(1.0));
        let mut truth = 0.0f32;
        for _ in 0..40 {
            truth += 4.0;
            f.predict(1.0);
            f.update(truth);
        }
        assert!((f.velocity() - 4.0).abs() < 0.5, "vel {}", f.velocity());
    }

    #[test]
    fn predict_extrapolates_linearly() {
        // After learning velocity ~4, a bare predict advances position by ~v and
        // leaves the velocity estimate untouched (constant-velocity model).
        let mut f = Kalman1D::new(0.0, cfg(1.0));
        let mut truth = 0.0f32;
        for _ in 0..40 {
            truth += 4.0;
            f.predict(1.0);
            f.update(truth);
        }
        let before = f.position();
        let v = f.velocity();
        f.predict(1.0);
        assert!((f.position() - (before + v)).abs() < 1e-3);
        assert_eq!(f.velocity(), v, "bare predict must not change velocity");
        // A second predict advances by another v (two steps ⇒ 2·v total).
        f.predict(1.0);
        assert!((f.position() - (before + 2.0 * v)).abs() < 1e-3);
    }

    #[test]
    fn predict_ignores_nonpositive_dt() {
        // The documented `dt <= 0` no-op must leave the whole state untouched.
        let mut f = Kalman1D::new(3.0, cfg(1.0));
        f.predict(1.0);
        f.update(20.0);
        let (pos, vel) = (f.position(), f.velocity());
        f.predict(0.0);
        assert_eq!((f.position(), f.velocity()), (pos, vel), "dt=0 is a no-op");
        f.predict(-2.0);
        assert_eq!((f.position(), f.velocity()), (pos, vel), "dt<0 is a no-op");
    }

    #[test]
    fn higher_responsiveness_tracks_faster() {
        // Same step change; the higher-responsiveness filter moves further after
        // one update.
        let mut fast = Kalman1D::new(0.0, cfg(1.0));
        let mut slow = Kalman1D::new(0.0, cfg(0.0));
        fast.predict(1.0);
        fast.update(100.0);
        slow.predict(1.0);
        slow.update(100.0);
        assert!(
            fast.position() > slow.position(),
            "fast {} slow {}",
            fast.position(),
            slow.position()
        );
    }

    #[test]
    fn from_responsiveness_is_monotonic_in_q() {
        assert!(cfg(1.0).q_c > cfg(0.5).q_c);
        assert!(cfg(0.5).q_c > cfg(0.0).q_c);
    }

    #[test]
    fn from_responsiveness_clamps_out_of_range() {
        assert_eq!(cfg(5.0).q_c, cfg(1.0).q_c);
        assert_eq!(cfg(-5.0).q_c, cfg(0.0).q_c);
    }

    #[test]
    fn from_responsiveness_keeps_r_fixed() {
        // Only the process noise `q_c` responds to the knob; `r` is constant.
        assert_eq!(cfg(0.0).r, cfg(1.0).r);
        assert_eq!(cfg(0.5).r, cfg(0.0).r);
    }
}
