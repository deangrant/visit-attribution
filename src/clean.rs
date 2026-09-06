//! GPS ping cleaning: accuracy spikes, impossible jumps, and driving.

use crate::config::Config;
use crate::geo::haversine_m;
use crate::types::GpsPing;

/// Filters raw GPS pings before clustering.
///
/// # Contract
///
/// - Empty input yields an empty output.
/// - Output is sorted by ascending `time_s`.
/// - Non-finite coordinates, times, or accuracies may be dropped.
/// - Implementations must not panic on empty slices.
pub trait PingCleaner {
    /// Return a cleaned, time-ordered subset of `pings`.
    fn clean(&self, pings: &[GpsPing]) -> Vec<GpsPing>;
}

/// Default cleaner matching the pipeline's pre-processing rules.
#[derive(Debug, Clone, Copy)]
pub struct DefaultPingCleaner {
    config: Config,
}

impl DefaultPingCleaner {
    /// Create a cleaner from validated configuration.
    #[must_use]
    pub const fn new(config: Config) -> Self {
        Self { config }
    }
}

impl PingCleaner for DefaultPingCleaner {
    fn clean(&self, pings: &[GpsPing]) -> Vec<GpsPing> {
        let mut ordered: Vec<GpsPing> = pings
            .iter()
            .copied()
            .filter(|p| {
                p.horizontal_accuracy_m.is_finite()
                    && p.horizontal_accuracy_m <= self.config.max_horizontal_accuracy_m
                    && p.time_s.is_finite()
                    && p.point.lat.is_finite()
                    && p.point.lon.is_finite()
            })
            .collect();
        ordered
            .sort_by(|a, b| a.time_s.partial_cmp(&b.time_s).unwrap_or(std::cmp::Ordering::Equal));
        let after_jumps = filter_jumpy(&ordered, self.config.max_speed_m_s);
        filter_driving(
            &after_jumps,
            self.config.linearity_threshold,
            self.config.linearity_window_s,
            self.config.driving_speed_m_s,
        )
    }
}

fn take_masked(pings: &[GpsPing], keep: &[bool]) -> Vec<GpsPing> {
    pings
        .iter()
        .zip(keep.iter())
        .filter_map(|(p, &keep)| keep.then_some(*p))
        .collect()
}

fn clear_keep(keep: &mut [bool], i: usize) {
    if let Some(flag) = keep.get_mut(i) {
        *flag = false;
    }
}

fn filter_jumpy(pings: &[GpsPing], max_speed_m_s: f64) -> Vec<GpsPing> {
    if pings.is_empty() {
        return Vec::new();
    }
    let mut keep = vec![true; pings.len()];
    let mut last_kept = 0usize;
    for (i, ping) in pings.iter().enumerate().skip(1) {
        let prev = pings[last_kept];
        let dt = ping.time_s - prev.time_s;
        if dt <= 0.0 {
            clear_keep(&mut keep, i);
            continue;
        }
        let dist = haversine_m(prev.point, ping.point);
        if dist / dt > max_speed_m_s {
            // Drop the later ping of an impossible jump from the last kept.
            clear_keep(&mut keep, i);
        } else {
            last_kept = i;
        }
    }
    take_masked(pings, &keep)
}

fn window_path_m(window: &[GpsPing]) -> f64 {
    let mut path = 0.0;
    for i in 1..window.len() {
        path += haversine_m(window[i - 1].point, window[i].point);
    }
    path
}

fn mark_fast_arrivals(
    pings: &[GpsPing],
    left: usize,
    right: usize,
    driving_speed_m_s: f64,
    drop: &mut [bool],
) {
    let last = pings.len().saturating_sub(1);
    for abs_idx in left..right.min(last) {
        let a = pings[abs_idx];
        let b = pings[abs_idx + 1];
        let hop_dt = (b.time_s - a.time_s).max(1e-6);
        let hop_m = haversine_m(a.point, b.point);
        if hop_m / hop_dt >= driving_speed_m_s {
            drop[abs_idx + 1] = true;
        }
    }
}

fn advance_window_left(pings: &[GpsPing], left: &mut usize, right: usize, window_s: f64) {
    if right >= pings.len() {
        return;
    }
    let right_ping = pings[right];
    while *left < right {
        if right_ping.time_s - pings[*left].time_s <= window_s {
            break;
        }
        *left += 1;
    }
}

fn window_is_driving(window: &[GpsPing], linearity_threshold: f64, driving_speed_m_s: f64) -> bool {
    let path = window_path_m(window);
    let (Some(first), Some(last)) = (window.first(), window.last()) else {
        return false;
    };
    let net = haversine_m(first.point, last.point);
    if net < 1.0 {
        return false;
    }
    let linearity = path / net;
    let speed = path / (last.time_s - first.time_s).max(1e-6);
    linearity <= linearity_threshold && speed >= driving_speed_m_s
}

fn filter_driving(
    pings: &[GpsPing],
    linearity_threshold: f64,
    window_s: f64,
    driving_speed_m_s: f64,
) -> Vec<GpsPing> {
    if pings.len() < 3 {
        return pings.to_vec();
    }
    let mut drop = vec![false; pings.len()];
    let mut left = 0usize;
    for right in 0..pings.len() {
        advance_window_left(pings, &mut left, right, window_s);
        if right - left + 1 < 3 {
            continue;
        }
        let window = &pings[left..=right];
        if window_is_driving(window, linearity_threshold, driving_speed_m_s) {
            // Drop only arrivals on fast hops so dwell edges in the window survive.
            mark_fast_arrivals(pings, left, right, driving_speed_m_s, &mut drop);
        }
    }
    let keep: Vec<bool> = drop.iter().map(|d| !d).collect();
    take_masked(pings, &keep)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clean_default(pings: &[GpsPing]) -> Vec<GpsPing> {
        DefaultPingCleaner::new(Config::default()).clean(pings)
    }

    #[test]
    fn drops_high_accuracy_and_impossible_jumps() {
        let accurate = clean_default(&[
            GpsPing::new(0.0, 0.0, 0.0, 10.0),
            GpsPing::new(0.0, 0.0, 1.0, 5000.0),
        ]);
        assert_eq!(accurate.len(), 1);
        assert!((accurate[0].horizontal_accuracy_m - 10.0).abs() < f64::EPSILON);

        // ~111 km in 1 second.
        assert_eq!(
            clean_default(&[
                GpsPing::new(0.0, 0.0, 0.0, 10.0),
                GpsPing::new(1.0, 0.0, 1.0, 10.0),
            ])
            .len(),
            1
        );
    }

    #[test]
    fn jump_filter_anchors_while_stationary_cluster_survives() {
        // A kept; B impossible from A (dropped); C near B but still impossible from A.
        let jumped = clean_default(&[
            GpsPing::new(0.0, 0.0, 0.0, 10.0),
            GpsPing::new(1.0, 0.0, 1.0, 10.0),
            GpsPing::new(1.0001, 0.0, 2.0, 10.0),
        ]);
        assert_eq!(jumped.len(), 1);
        assert!((jumped[0].point.lat - 0.0).abs() < f64::EPSILON);
        assert!((jumped[0].point.lon - 0.0).abs() < f64::EPSILON);

        let stationary = clean_default(&[
            GpsPing::new(0.0, 0.0, 0.0, 10.0),
            GpsPing::new(0.00001, 0.0, 10.0, 10.0),
            GpsPing::new(0.0, 0.00001, 20.0, 10.0),
        ]);
        assert_eq!(stationary.len(), 3);
    }

    #[test]
    fn driving_filter_preserves_dwell_before_drive() {
        let pings = [
            GpsPing::new(0.0, 0.0, 0.0, 10.0),
            GpsPing::new(0.00001, 0.0, 10.0, 10.0),
            GpsPing::new(0.0, 0.00001, 20.0, 10.0),
            // ~1.1 km / 10 s ≈ 111 m/s, well above driving_speed_m_s.
            GpsPing::new(0.01, 0.0, 30.0, 10.0),
            GpsPing::new(0.02, 0.0, 40.0, 10.0),
            GpsPing::new(0.03, 0.0, 50.0, 10.0),
        ];
        let out = clean_default(&pings);
        assert_eq!(out.iter().filter(|p| p.time_s <= 20.0).count(), 3);
        assert!(out.len() < pings.len());
    }

    #[test]
    fn drops_non_finite_fields() {
        let good = GpsPing::new(0.0, 0.0, 0.0, 10.0);
        let out = clean_default(&[
            good,
            GpsPing::new(f64::NAN, 0.0, 1.0, 10.0),
            GpsPing::new(0.0, f64::INFINITY, 2.0, 10.0),
            GpsPing::new(0.0, 0.0, f64::NAN, 10.0),
            GpsPing::new(0.0, 0.0, 3.0, f64::NEG_INFINITY),
            GpsPing::new(0.00001, 0.0, 4.0, 10.0),
        ]);
        assert_eq!(out.len(), 2);
        assert!((out[0].time_s - 0.0).abs() < f64::EPSILON);
        assert!((out[1].time_s - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn drops_equal_timestamps() {
        // Cleaning sorts by time first; equal times yield dt <= 0 vs last kept.
        let out = clean_default(&[
            GpsPing::new(0.0, 0.0, 0.0, 10.0),
            GpsPing::new(0.00001, 0.0, 0.0, 10.0), // equal time → drop
            GpsPing::new(0.00002, 0.0, 10.0, 10.0),
            GpsPing::new(0.00003, 0.0, 10.0, 10.0), // equal to last kept → drop
            GpsPing::new(0.00004, 0.0, 20.0, 10.0),
        ]);
        assert_eq!(out.len(), 3);
        assert!((out[0].time_s - 0.0).abs() < f64::EPSILON);
        assert!((out[1].time_s - 10.0).abs() < f64::EPSILON);
        assert!((out[2].time_s - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn jumpy_filter_empty_and_short_inputs() {
        assert!(filter_jumpy(&[], 50.0).is_empty());
        let one = [GpsPing::new(0.0, 0.0, 0.0, 10.0)];
        assert_eq!(filter_jumpy(&one, 50.0).len(), 1);
        assert_eq!(filter_driving(&one, 1.15, 60.0, 8.0).len(), 1);
    }

    #[test]
    fn driving_filter_drops_fast_linear_arrivals() {
        // ~33 m hops / 3 s ≈ 11 m/s: above driving_speed, below max_speed.
        let pings = [
            GpsPing::new(0.0, 0.0, 0.0, 10.0),
            GpsPing::new(0.0003, 0.0, 3.0, 10.0),
            GpsPing::new(0.0006, 0.0, 6.0, 10.0),
            GpsPing::new(0.0009, 0.0, 9.0, 10.0),
        ];
        let cfg = Config::default();
        let out = filter_driving(
            &pings,
            cfg.linearity_threshold,
            cfg.linearity_window_s,
            cfg.driving_speed_m_s,
        );
        assert_eq!(out.first().map(|p| p.time_s), Some(0.0));
        assert!(out.len() < pings.len());

        let dwell = [
            GpsPing::new(0.0, 0.0, 0.0, 10.0),
            GpsPing::new(0.0, 0.0, 10.0, 10.0),
            GpsPing::new(0.0, 0.0, 20.0, 10.0),
        ];
        assert_eq!(filter_driving(&dwell, 1.15, 60.0, 8.0).len(), 3);
        assert!(!window_is_driving(&[], 1.15, 8.0));
        let mut drop = [false, false];
        mark_fast_arrivals(
            &[
                GpsPing::new(0.0, 0.0, 0.0, 10.0),
                GpsPing::new(0.0, 0.0, 10.0, 10.0),
            ],
            0,
            1,
            8.0,
            &mut drop,
        );
        assert!(!drop[1]);
        let mut left = 0usize;
        advance_window_left(&[], &mut left, 0, 60.0);
        assert_eq!(left, 0);
        let span = [
            GpsPing::new(0.0, 0.0, 0.0, 10.0),
            GpsPing::new(0.0, 0.0, 100.0, 10.0),
        ];
        advance_window_left(&span, &mut left, 1, 60.0);
        assert_eq!(left, 1);
    }
}
