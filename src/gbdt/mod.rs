//! In-crate gradient-boosted decision trees for preference scores.

mod format;

use std::fs;
use std::path::Path;

use crate::error::{Error, Result};
use crate::features::FeatureSchema;

/// Upper bound on forest size for train and load.
const MAX_TREES: usize = 256;
/// Maximum tree depth (root = 0) for train and load.
const MAX_TREE_DEPTH: usize = 32;
/// Maximum nodes allowed in a single loaded tree.
const MAX_NODES_PER_TREE: usize = 8192;
/// Maximum serialized model size accepted by load/parse.
const MAX_MODEL_BYTES: usize = 8 * 1024 * 1024;

/// Hyperparameters for GBDT training.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainConfig {
    /// Number of boosting rounds.
    pub n_trees: usize,
    /// Maximum tree depth (root depth = 0).
    pub max_depth: usize,
    /// Minimum samples in a leaf.
    pub min_leaf: usize,
    /// Shrinkage applied to each tree.
    pub learning_rate: f64,
    /// Deterministic subsample stride (`1` = use all rows).
    pub subsample_stride: usize,
    /// Maximum split thresholds evaluated per feature.
    pub max_bins: usize,
    /// Max distinct NAICS4 prefixes kept in the feature schema (0 = UNK only).
    pub max_naics4: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            n_trees: 32,
            max_depth: 3,
            min_leaf: 2,
            learning_rate: 0.1,
            subsample_stride: 1,
            max_bins: 16,
            max_naics4: 32,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Node {
    Leaf {
        value: f64,
    },
    Branch {
        feature: usize,
        threshold: f64,
        left: Box<Node>,
        right: Box<Node>,
    },
}

/// Trained gradient-boosted forest over preference difference vectors.
#[derive(Debug, Clone, PartialEq)]
pub struct GbdtModel {
    /// Feature schema the model was trained with.
    pub schema: FeatureSchema,
    /// Bias term before trees.
    pub base_score: f64,
    learning_rate: f64,
    trees: Vec<Node>,
}

impl GbdtModel {
    /// Train on difference feature rows with ±1 labels.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] when the training set is empty or
    /// feature widths are inconsistent.
    pub fn train(
        schema: FeatureSchema,
        xs: &[Vec<f64>],
        ys: &[f64],
        config: &TrainConfig,
    ) -> Result<Self> {
        if xs.is_empty() || ys.is_empty() || xs.len() != ys.len() {
            return Err(Error::InvalidInput(
                "training set must be non-empty with matching lengths".into(),
            ));
        }
        let dim = schema.dim();
        for (i, row) in xs.iter().enumerate() {
            if row.len() != dim {
                return Err(Error::InvalidInput(format!(
                    "row {i} has width {}, expected {dim}",
                    row.len()
                )));
            }
        }
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
        let base_score = 0.0;
        let mut preds = vec![base_score; xs.len()];
        let mut trees = Vec::with_capacity(config.n_trees);
        let indices: Vec<usize> = (0..xs.len()).step_by(config.subsample_stride).collect();
        for _ in 0..config.n_trees {
            let mut residuals = Vec::with_capacity(indices.len());
            let mut sample_x = Vec::with_capacity(indices.len());
            for &i in &indices {
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
            let tree = build_tree(&sample_x, &residuals, config, 0);
            for (i, row) in xs.iter().enumerate() {
                let Some(pred) = preds.get_mut(i) else {
                    continue;
                };
                *pred += config.learning_rate * eval_tree(&tree, row)?;
            }
            trees.push(tree);
        }
        Ok(Self {
            schema,
            base_score,
            learning_rate: config.learning_rate,
            trees,
        })
    }

    /// Predict a continuous preference score for a difference vector.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] when `x` does not match the schema width,
    /// or [`Error::Model`] if a tree indexes past the vector (corrupt model).
    pub fn predict_raw(&self, x: &[f64]) -> Result<f64> {
        if x.len() != self.schema.dim() {
            return Err(Error::InvalidInput(format!(
                "feature vector width {}, expected {}",
                x.len(),
                self.schema.dim()
            )));
        }
        let mut s = self.base_score;
        for tree in &self.trees {
            s += self.learning_rate * eval_tree(tree, x)?;
        }
        Ok(s)
    }

    /// Save the model to a versioned text file.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] on filesystem failures.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        fs::write(path, self.to_string_format())?;
        Ok(())
    }

    /// Load a model previously written by [`Self::save`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Model`] or [`Error::Io`] on parse or filesystem failure.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let text = fs::read_to_string(path)?;
        Self::from_string_format(&text)
    }

    fn to_string_format(&self) -> String {
        format::to_string_format(self)
    }

    fn from_string_format(text: &str) -> Result<Self> {
        format::from_string_format(text)
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn build_tree(xs: &[Vec<f64>], ys: &[f64], config: &TrainConfig, depth: usize) -> Node {
    let all_equal = ys.first().is_some_and(|&y0| ys.iter().all(|&y| (y - y0).abs() < 1e-12));
    if xs.len() < config.min_leaf * 2 || depth >= config.max_depth || all_equal {
        return Node::Leaf { value: mean(ys) };
    }
    let Some((feature, threshold, left_idx, right_idx)) = best_split(xs, ys, config) else {
        return Node::Leaf { value: mean(ys) };
    };
    let left_xs: Vec<Vec<f64>> = left_idx.iter().filter_map(|&i| xs.get(i).cloned()).collect();
    let left_ys: Vec<f64> = left_idx.iter().filter_map(|&i| ys.get(i).copied()).collect();
    let right_xs: Vec<Vec<f64>> = right_idx.iter().filter_map(|&i| xs.get(i).cloned()).collect();
    let right_ys: Vec<f64> = right_idx.iter().filter_map(|&i| ys.get(i).copied()).collect();
    Node::Branch {
        feature,
        threshold,
        left: Box::new(build_tree(&left_xs, &left_ys, config, depth + 1)),
        right: Box::new(build_tree(&right_xs, &right_ys, config, depth + 1)),
    }
}

fn best_split(
    xs: &[Vec<f64>],
    ys: &[f64],
    config: &TrainConfig,
) -> Option<(usize, f64, Vec<usize>, Vec<usize>)> {
    let dim = xs.first()?.len();
    let mut best_gain = 0.0;
    let mut best = None;
    for f in 0..dim {
        let mut vals: Vec<f64> = xs.iter().filter_map(|r| r.get(f).copied()).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        vals.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
        if vals.len() < 2 {
            continue;
        }
        let step = (vals.len() / config.max_bins.max(1)).max(1);
        for (k, &threshold) in vals.iter().enumerate().step_by(step) {
            if k + 1 == vals.len() {
                break;
            }
            let mut left_idx = Vec::new();
            let mut right_idx = Vec::new();
            for (i, row) in xs.iter().enumerate() {
                let Some(&value) = row.get(f) else {
                    continue;
                };
                if value <= threshold {
                    left_idx.push(i);
                } else {
                    right_idx.push(i);
                }
            }
            if left_idx.len() < config.min_leaf || right_idx.len() < config.min_leaf {
                continue;
            }
            let parent_var = variance(ys);
            let left_ys: Vec<f64> = left_idx.iter().filter_map(|&i| ys.get(i).copied()).collect();
            let right_ys: Vec<f64> = right_idx.iter().filter_map(|&i| ys.get(i).copied()).collect();
            let gain = parent_var
                - (left_ys.len() as f64 * variance(&left_ys)
                    + right_ys.len() as f64 * variance(&right_ys))
                    / ys.len() as f64;
            if gain > best_gain {
                best_gain = gain;
                best = Some((f, threshold, left_idx, right_idx));
            }
        }
    }
    best
}

fn variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let m = mean(values);
    values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / values.len() as f64
}

fn eval_tree(node: &Node, x: &[f64]) -> Result<f64> {
    match node {
        Node::Leaf { value } => Ok(*value),
        Node::Branch {
            feature,
            threshold,
            left,
            right,
        } => {
            let value = x.get(*feature).copied().ok_or_else(|| {
                Error::Model(format!(
                    "tree feature index {feature} out of range for width {}",
                    x.len()
                ))
            })?;
            if value <= *threshold {
                eval_tree(left, x)
            } else {
                eval_tree(right, x)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row4(first: f64) -> Vec<f64> {
        let mut v = vec![0.0; 28]; // empty-naics schema dim = 4 + 24 UNK
        if let Some(slot) = v.get_mut(0) {
            *slot = first;
        }
        v
    }

    /// Minimal single-leaf model text for rejection fixtures.
    fn leaf_model_text(header: &str, dim: usize, n_trees: usize, trees: &str) -> String {
        format!("{header}\nbase 0\nlr 0.1\ndim {dim}\nnaics\ntrees {n_trees}\n{trees}\n")
    }

    #[test]
    fn learns_simple_preference_direction() {
        let schema = FeatureSchema::new(vec![]);
        // Difference feature: positive when left is better on first dim.
        let xs = vec![row4(2.0), row4(1.0), row4(-2.0), row4(-1.0)];
        let ys = vec![1.0, 1.0, -1.0, -1.0];
        let model = GbdtModel::train(schema, &xs, &ys, &TrainConfig::default()).unwrap();
        assert!(model.predict_raw(xs.first().unwrap()).unwrap() > 0.0);
        assert!(model.predict_raw(xs.get(2).unwrap()).unwrap() < 0.0);
        let text = model.to_string_format();
        let loaded = GbdtModel::from_string_format(&text).unwrap();
        assert!(
            (loaded.predict_raw(xs.first().unwrap()).unwrap()
                - model.predict_raw(xs.first().unwrap()).unwrap())
            .abs()
                < 1e-9
        );
    }

    fn assert_model_err(text: &str) {
        let err = GbdtModel::from_string_format(text).unwrap_err();
        assert!(matches!(err, Error::Model(_)));
    }

    #[test]
    fn load_rejects_dim_mismatch_with_schema() {
        // Empty naics ⇒ schema.dim() == 4 + 24 (UNK); claim dim 99.
        assert_model_err(&leaf_model_text("VA_GBDT 1", 99, 1, "L 0"));
    }

    #[test]
    fn load_rejects_out_of_range_feature_index() {
        // Empty naics ⇒ dim 28; branch splits on feature 40.
        assert_model_err(&leaf_model_text("VA_GBDT 1", 28, 1, "B 40 0 L 1 L -1"));
    }

    #[test]
    fn predict_raw_rejects_wrong_width() {
        let schema = FeatureSchema::new(vec![]);
        let xs = vec![row4(1.0), row4(-1.0)];
        let ys = vec![1.0, -1.0];
        let model = GbdtModel::train(schema, &xs, &ys, &TrainConfig::default()).unwrap();
        let err = model.predict_raw(&[1.0, 0.0]).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn load_rejects_too_many_trees() {
        // MAX_TREES is 256.
        assert_model_err(&leaf_model_text("VA_GBDT 1", 28, 257, "L 0"));
    }

    #[test]
    fn load_rejects_excessive_tree_depth() {
        let mut tree = String::new();
        for _ in 0..=MAX_TREE_DEPTH {
            tree.push_str("B 0 0 ");
        }
        tree.push_str("L 0");
        for _ in 0..=MAX_TREE_DEPTH {
            tree.push_str(" L 0");
        }
        assert_model_err(&leaf_model_text("VA_GBDT 1", 28, 1, &tree));
    }

    #[test]
    fn load_rejects_oversized_model_text() {
        let mut text = leaf_model_text("VA_GBDT 1", 28, 1, "L 0");
        text.push_str(&"x".repeat(MAX_MODEL_BYTES));
        assert_model_err(&text);
    }

    #[test]
    fn train_rejects_excessive_n_trees() {
        let schema = FeatureSchema::new(vec![]);
        let xs = vec![row4(1.0), row4(-1.0)];
        let ys = vec![1.0, -1.0];
        let config = TrainConfig {
            n_trees: MAX_TREES + 1,
            ..TrainConfig::default()
        };
        let err = GbdtModel::train(schema, &xs, &ys, &config).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn load_rejects_wrong_header() {
        assert_model_err(&leaf_model_text("VA_GBDT 2", 28, 1, "L 0"));
    }

    #[test]
    fn load_rejects_truncated_tree_lines() {
        assert_model_err(&leaf_model_text("VA_GBDT 1", 28, 2, "L 0"));
    }

    #[test]
    fn save_load_filesystem_round_trip() {
        let schema = FeatureSchema::new(vec![]);
        let xs = vec![row4(2.0), row4(-2.0)];
        let ys = vec![1.0, -1.0];
        let model = GbdtModel::train(schema, &xs, &ys, &TrainConfig::default()).unwrap();
        let path = std::env::temp_dir().join(format!(
            "visit-attribution-model-{}-{}.va",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        model.save(&path).unwrap();
        let loaded = GbdtModel::load(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert!(
            (loaded.predict_raw(xs.first().unwrap()).unwrap()
                - model.predict_raw(xs.first().unwrap()).unwrap())
            .abs()
                < 1e-9
        );
    }
}
