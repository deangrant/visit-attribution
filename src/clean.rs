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
#[derive(Debug, Clone)]
pub struct DefaultPingCleaner {
    config: Config,
}

impl DefaultPingCleaner {
    /// Create a cleaner from validated configuration.
    #[must_use]
    pub fn new(config: Config) -> Self {
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

fn filter_jumpy(pings: &[GpsPing], max_speed_m_s: f64) -> Vec<GpsPing> {
    if pings.is_empty() {
        return Vec::new();
    }
    let mut keep = vec![true; pings.len()];
    let mut last_kept = 0usize;
    for i in 1..pings.len() {
        let dt = pings[i].time_s - pings[last_kept].time_s;
        if dt <= 0.0 {
            keep[i] = false;
            continue;
        }
        let dist = haversine_m(pings[last_kept].point, pings[i].point);
        if dist / dt > max_speed_m_s {
            // Drop the later ping of an impossible jump from the last kept.
            keep[i] = false;
        } else {
            last_kept = i;
        }
    }
    pings.iter().enumerate().filter_map(|(i, p)| keep[i].then_some(*p)).collect()
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
        while left < right && pings[right].time_s - pings[left].time_s > window_s {
            left += 1;
        }
        if right - left + 1 < 3 {
            continue;
        }
        let window = &pings[left..=right];
        let mut path = 0.0;
        for i in 1..window.len() {
            path += haversine_m(window[i - 1].point, window[i].point);
        }
        let net = haversine_m(window[0].point, window[window.len() - 1].point);
        if net < 1.0 {
            continue;
        }
        let linearity = path / net;
        let dt = (window[window.len() - 1].time_s - window[0].time_s).max(1e-6);
        let speed = path / dt;
        if linearity <= linearity_threshold && speed >= driving_speed_m_s {
            // Drop only arrivals on fast hops so dwell edges in the window survive.
            for i in left..right {
                let hop_dt = (pings[i + 1].time_s - pings[i].time_s).max(1e-6);
                let hop = haversine_m(pings[i].point, pings[i + 1].point);
                if hop / hop_dt >= driving_speed_m_s {
                    drop[i + 1] = true;
                }
            }
        }
    }
    pings.iter().enumerate().filter_map(|(i, p)| (!drop[i]).then_some(*p)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Point;

    fn ping(lat: f64, lon: f64, t: f64, ha: f64) -> GpsPing {
        GpsPing {
            point: Point::new(lat, lon),
            time_s: t,
            horizontal_accuracy_m: ha,
        }
    }

    #[test]
    fn drops_high_horizontal_accuracy() {
        let cleaner = DefaultPingCleaner::new(Config::default());
        let out = cleaner.clean(&[ping(0.0, 0.0, 0.0, 10.0), ping(0.0, 0.0, 1.0, 5000.0)]);
        assert_eq!(out.len(), 1);
        assert!((out[0].horizontal_accuracy_m - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn drops_impossible_jumps() {
        let cleaner = DefaultPingCleaner::new(Config::default());
        // ~111 km in 1 second.
        let out = cleaner.clean(&[ping(0.0, 0.0, 0.0, 10.0), ping(1.0, 0.0, 1.0, 10.0)]);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn jump_filter_anchors_on_last_kept_ping() {
        let cleaner = DefaultPingCleaner::new(Config::default());
        // A kept; B impossible from A (dropped); C near B but still impossible from A.
        let out = cleaner.clean(&[
            ping(0.0, 0.0, 0.0, 10.0),
            ping(1.0, 0.0, 1.0, 10.0),
            ping(1.0001, 0.0, 2.0, 10.0),
        ]);
        assert_eq!(out.len(), 1);
        assert!((out[0].point.lat - 0.0).abs() < f64::EPSILON);
        assert!((out[0].point.lon - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn keeps_stationary_cluster() {
        let cleaner = DefaultPingCleaner::new(Config::default());
        let out = cleaner.clean(&[
            ping(0.0, 0.0, 0.0, 10.0),
            ping(0.00001, 0.0, 10.0, 10.0),
            ping(0.0, 0.00001, 20.0, 10.0),
        ]);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn driving_filter_preserves_dwell_before_drive() {
        let cleaner = DefaultPingCleaner::new(Config::default());
        let pings = [
            ping(0.0, 0.0, 0.0, 10.0),
            ping(0.00001, 0.0, 10.0, 10.0),
            ping(0.0, 0.00001, 20.0, 10.0),
            // ~1.1 km / 10 s ≈ 111 m/s, well above driving_speed_m_s.
            ping(0.01, 0.0, 30.0, 10.0),
            ping(0.02, 0.0, 40.0, 10.0),
            ping(0.03, 0.0, 50.0, 10.0),
        ];
        let out = cleaner.clean(&pings);
        let dwell: Vec<_> = out.iter().filter(|p| p.time_s <= 20.0).collect();
        assert_eq!(dwell.len(), 3);
        assert!(out.len() < pings.len());
    }
}
