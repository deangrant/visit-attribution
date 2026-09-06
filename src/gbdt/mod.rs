//! In-crate gradient-boosted decision trees for preference scores.

mod format;
mod train;

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
#[derive(Debug, Clone, Copy, PartialEq)]
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

/// One node in a trained preference tree.
#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    /// Terminal score.
    Leaf {
        /// Predicted residual.
        value: f64,
    },
    /// Split on a feature threshold.
    Branch {
        /// Feature index in the difference vector.
        feature: usize,
        /// Values `<= threshold` take the left child.
        threshold: f64,
        /// Left subtree.
        left: Box<Self>,
        /// Right subtree.
        right: Box<Self>,
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
        train::validate_xy(xs, ys, schema.dim())?;
        train::validate_train_config(config)?;
        let base_score = 0.0;
        let mut preds = vec![base_score; xs.len()];
        let mut trees = Vec::with_capacity(config.n_trees);
        let indices: Vec<usize> = (0..xs.len()).step_by(config.subsample_stride).collect();
        for _ in 0..config.n_trees {
            let (residuals, sample_x) = train::build_residuals(xs, ys, &preds, &indices);
            let tree = train::build_tree(&sample_x, &residuals, config, 0);
            train::apply_tree(&mut preds, xs, &tree, config.learning_rate)?;
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
            s = self.learning_rate.mul_add(eval_tree(tree, x)?, s);
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

/// Walk `node` to a leaf for feature vector `x`.
///
/// # Errors
///
/// Returns [`Error::Model`] when a branch indexes a feature past `x`.
pub fn eval_tree(node: &Node, x: &[f64]) -> Result<f64> {
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
#[allow(clippy::cognitive_complexity)]
mod tests;
