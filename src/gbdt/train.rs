//! Training helpers for [`super::GbdtModel`].

use crate::error::{Error, Result};

use super::{eval_tree, Node, TrainConfig, MAX_TREES, MAX_TREE_DEPTH};

pub(super) fn validate_xy(xs: &[Vec<f64>], ys: &[f64], dim: usize) -> Result<()> {
    if xs.is_empty() || ys.is_empty() || xs.len() != ys.len() {
        return Err(Error::InvalidInput(
            "training set must be non-empty with matching lengths".into(),
        ));
    }
    for (i, row) in xs.iter().enumerate() {
        if row.len() != dim {
            return Err(Error::InvalidInput(format!(
                "row {i} has width {}, expected {dim}",
                row.len()
            )));
        }
    }
    Ok(())
}

pub(super) fn validate_train_config(config: &TrainConfig) -> Result<()> {
    if config.n_trees == 0 || config.min_leaf == 0 || config.subsample_stride == 0 {
        return Err(Error::InvalidInput(
            "n_trees, min_leaf, and subsample_stride must be > 0".into(),
        ));
    }
    if config.n_trees > MAX_TREES {
        return Err(Error::InvalidInput(format!(
            "n_trees {} exceeds limit {MAX_TREES}",
            config.n_trees
        )));
    }
    if config.max_depth > MAX_TREE_DEPTH {
        return Err(Error::InvalidInput(format!(
            "max_depth {} exceeds limit {MAX_TREE_DEPTH}",
            config.max_depth
        )));
    }
    Ok(())
}

pub(super) fn build_residuals(
    xs: &[Vec<f64>],
    ys: &[f64],
    preds: &[f64],
    indices: &[usize],
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut residuals = Vec::with_capacity(indices.len());
    let mut sample_x = Vec::with_capacity(indices.len());
    for &i in indices {
        let Some(&pred) = preds.get(i) else {
            continue;
        };
        let Some(&y) = ys.get(i) else {
            continue;
        };
        let Some(row) = xs.get(i) else {
            continue;
        };
        let p = sigmoid(pred);
        // Map ±1 labels to {0,1} for the logistic residual.
        let y01 = if y > 0.0 { 1.0 } else { 0.0 };
        residuals.push(y01 - p);
        sample_x.push(row.clone());
    }
    (residuals, sample_x)
}

pub(super) fn apply_tree(
    preds: &mut [f64],
    xs: &[Vec<f64>],
    tree: &Node,
    learning_rate: f64,
) -> Result<()> {
    for (i, row) in xs.iter().enumerate() {
        let Some(pred) = preds.get_mut(i) else {
            continue;
        };
        *pred += learning_rate * eval_tree(tree, row)?;
    }
    Ok(())
}

pub(super) fn build_tree(xs: &[Vec<f64>], ys: &[f64], config: &TrainConfig, depth: usize) -> Node {
    let all_equal = ys.first().is_some_and(|&y0| ys.iter().all(|&y| (y - y0).abs() < 1e-12));
    if xs.len() < config.min_leaf * 2 || depth >= config.max_depth || all_equal {
        return Node::Leaf { value: mean(ys) };
    }
    let Some((feature, threshold, left_idx, right_idx)) = best_split(xs, ys, config) else {
        return Node::Leaf { value: mean(ys) };
    };
    let features_left: Vec<Vec<f64>> =
        left_idx.iter().filter_map(|&i| xs.get(i).cloned()).collect();
    let labels_left: Vec<f64> = left_idx.iter().filter_map(|&i| ys.get(i).copied()).collect();
    let features_right: Vec<Vec<f64>> =
        right_idx.iter().filter_map(|&i| xs.get(i).cloned()).collect();
    let labels_right: Vec<f64> = right_idx.iter().filter_map(|&i| ys.get(i).copied()).collect();
    Node::Branch {
        feature,
        threshold,
        left: Box::new(build_tree(&features_left, &labels_left, config, depth + 1)),
        right: Box::new(build_tree(
            &features_right,
            &labels_right,
            config,
            depth + 1,
        )),
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / crate::types::len_f64(values.len())
    }
}

type Split = (usize, f64, Vec<usize>, Vec<usize>);

fn best_split(xs: &[Vec<f64>], ys: &[f64], config: &TrainConfig) -> Option<Split> {
    let dim = xs.first()?.len();
    let mut best_gain = 0.0;
    let mut best = None;
    for f in 0..dim {
        let vals = feature_thresholds(xs, f);
        if vals.len() < 2 {
            continue;
        }
        let step = (vals.len() / config.max_bins.max(1)).max(1);
        for (k, &threshold) in vals.iter().enumerate().step_by(step) {
            if k + 1 == vals.len() {
                break;
            }
            consider_split(
                xs,
                ys,
                f,
                threshold,
                config.min_leaf,
                &mut best_gain,
                &mut best,
            );
        }
    }
    best
}

fn feature_thresholds(xs: &[Vec<f64>], feature: usize) -> Vec<f64> {
    let mut vals: Vec<f64> = xs.iter().filter_map(|r| r.get(feature).copied()).collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    vals.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
    vals
}

fn partition_by_threshold(
    xs: &[Vec<f64>],
    feature: usize,
    threshold: f64,
) -> (Vec<usize>, Vec<usize>) {
    let mut left_idx = Vec::new();
    let mut right_idx = Vec::new();
    for (i, row) in xs.iter().enumerate() {
        let Some(&value) = row.get(feature) else {
            continue;
        };
        if value <= threshold {
            left_idx.push(i);
        } else {
            right_idx.push(i);
        }
    }
    (left_idx, right_idx)
}

fn split_gain(ys: &[f64], left_idx: &[usize], right_idx: &[usize]) -> f64 {
    let parent_var = variance(ys);
    let left_ys: Vec<f64> = left_idx.iter().filter_map(|&i| ys.get(i).copied()).collect();
    let right_ys: Vec<f64> = right_idx.iter().filter_map(|&i| ys.get(i).copied()).collect();
    parent_var
        - crate::types::len_f64(left_ys.len()).mul_add(
            variance(&left_ys),
            crate::types::len_f64(right_ys.len()) * variance(&right_ys),
        ) / crate::types::len_f64(ys.len())
}

fn consider_split(
    xs: &[Vec<f64>],
    ys: &[f64],
    feature: usize,
    threshold: f64,
    min_leaf: usize,
    best_gain: &mut f64,
    best: &mut Option<Split>,
) {
    let (left_idx, right_idx) = partition_by_threshold(xs, feature, threshold);
    if left_idx.len() < min_leaf || right_idx.len() < min_leaf {
        return;
    }
    let gain = split_gain(ys, &left_idx, &right_idx);
    if gain > *best_gain {
        *best_gain = gain;
        *best = Some((feature, threshold, left_idx, right_idx));
    }
}

fn variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let m = mean(values);
    values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / crate::types::len_f64(values.len())
}
