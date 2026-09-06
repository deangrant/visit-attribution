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

/// Walk `node` to a leaf for feature vector `x`.
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
mod tests {
    use super::*;
    use crate::error::Result;

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
    fn learns_simple_preference_direction() -> Result<()> {
        let schema = FeatureSchema::new(vec![]);
        // Difference feature: positive when left is better on first dim.
        let xs = vec![row4(2.0), row4(1.0), row4(-2.0), row4(-1.0)];
        let ys = vec![1.0, 1.0, -1.0, -1.0];
        let model = GbdtModel::train(schema, &xs, &ys, &TrainConfig::default())?;
        assert!(model.predict_raw(crate::test_util::some(xs.first())?)? > 0.0);
        assert!(model.predict_raw(crate::test_util::some(xs.get(2))?)? < 0.0);
        let text = model.to_string_format();
        let loaded = GbdtModel::from_string_format(&text)?;
        assert!(
            (loaded.predict_raw(crate::test_util::some(xs.first())?)?
                - model.predict_raw(crate::test_util::some(xs.first())?)?)
            .abs()
                < 1e-9
        );
        Ok(())
    }

    fn assert_model_err(text: &str) -> Result<()> {
        let err = crate::test_util::err(GbdtModel::from_string_format(text))?;
        assert!(matches!(err, Error::Model(_)));
        Ok(())
    }

    #[test]
    fn load_rejects_dim_mismatch_with_schema() -> Result<()> {
        // Empty naics ⇒ schema.dim() == 4 + 24 (UNK); claim dim 99.
        assert_model_err(&leaf_model_text("VA_GBDT 1", 99, 1, "L 0"))?;
        Ok(())
    }

    #[test]
    fn load_rejects_out_of_range_feature_index() -> Result<()> {
        // Empty naics ⇒ dim 28; branch splits on feature 40.
        assert_model_err(&leaf_model_text("VA_GBDT 1", 28, 1, "B 40 0 L 1 L -1"))?;
        Ok(())
    }

    #[test]
    fn predict_raw_rejects_wrong_width() -> Result<()> {
        let schema = FeatureSchema::new(vec![]);
        let xs = vec![row4(1.0), row4(-1.0)];
        let ys = vec![1.0, -1.0];
        let model = GbdtModel::train(schema, &xs, &ys, &TrainConfig::default())?;
        let err = crate::test_util::err(model.predict_raw(&[1.0, 0.0]))?;
        assert!(matches!(err, Error::InvalidInput(_)));
        Ok(())
    }

    #[test]
    fn load_rejects_too_many_trees() -> Result<()> {
        // MAX_TREES is 256.
        assert_model_err(&leaf_model_text("VA_GBDT 1", 28, 257, "L 0"))?;
        Ok(())
    }

    #[test]
    fn load_rejects_excessive_tree_depth() -> Result<()> {
        let mut tree = String::new();
        for _ in 0..=MAX_TREE_DEPTH {
            tree.push_str("B 0 0 ");
        }
        tree.push_str("L 0");
        for _ in 0..=MAX_TREE_DEPTH {
            tree.push_str(" L 0");
        }
        assert_model_err(&leaf_model_text("VA_GBDT 1", 28, 1, &tree))?;
        Ok(())
    }

    #[test]
    fn load_rejects_oversized_model_text() -> Result<()> {
        let mut text = leaf_model_text("VA_GBDT 1", 28, 1, "L 0");
        text.push_str(&"x".repeat(MAX_MODEL_BYTES));
        assert_model_err(&text)?;
        Ok(())
    }

    #[test]
    fn train_rejects_excessive_n_trees() -> Result<()> {
        let schema = FeatureSchema::new(vec![]);
        let xs = vec![row4(1.0), row4(-1.0)];
        let ys = vec![1.0, -1.0];
        let config = TrainConfig {
            n_trees: MAX_TREES + 1,
            ..TrainConfig::default()
        };
        let err = crate::test_util::err(GbdtModel::train(schema, &xs, &ys, &config))?;
        assert!(matches!(err, Error::InvalidInput(_)));
        Ok(())
    }

    #[test]
    fn train_rejects_empty_or_mismatched_inputs() -> Result<()> {
        let schema = FeatureSchema::new(vec![]);
        let err = crate::test_util::err(GbdtModel::train(
            schema.clone(),
            &[],
            &[],
            &TrainConfig::default(),
        ))?;
        assert!(matches!(err, Error::InvalidInput(_)));
        let err = crate::test_util::err(GbdtModel::train(
            schema.clone(),
            &[row4(1.0)],
            &[1.0, -1.0],
            &TrainConfig::default(),
        ))?;
        assert!(matches!(err, Error::InvalidInput(_)));
        let err = crate::test_util::err(GbdtModel::train(
            schema,
            &[vec![1.0]],
            &[1.0],
            &TrainConfig::default(),
        ))?;
        assert!(matches!(err, Error::InvalidInput(_)));
        Ok(())
    }

    #[test]
    fn train_rejects_zero_hyperparameters_and_deep_trees() -> Result<()> {
        let schema = FeatureSchema::new(vec![]);
        let xs = vec![row4(1.0), row4(-1.0)];
        let ys = vec![1.0, -1.0];
        let err = crate::test_util::err(GbdtModel::train(
            schema.clone(),
            &xs,
            &ys,
            &TrainConfig {
                n_trees: 0,
                ..TrainConfig::default()
            },
        ))?;
        assert!(matches!(err, Error::InvalidInput(_)));
        let err = crate::test_util::err(GbdtModel::train(
            schema,
            &xs,
            &ys,
            &TrainConfig {
                max_depth: MAX_TREE_DEPTH + 1,
                ..TrainConfig::default()
            },
        ))?;
        assert!(matches!(err, Error::InvalidInput(_)));
        Ok(())
    }

    #[test]
    fn load_rejects_wrong_header() -> Result<()> {
        assert_model_err(&leaf_model_text("VA_GBDT 2", 28, 1, "L 0"))?;
        Ok(())
    }

    #[test]
    fn load_parses_naics_and_rejects_bad_meta() -> Result<()> {
        let ok = "VA_GBDT 1\nbase 0\nlr 0.1\ndim 52\nnaics 4451\ntrees 1\nL 0\n";
        GbdtModel::from_string_format(ok)?;
        assert_model_err("VA_GBDT 1\nbase 0\nlr 0.1\ndim 28\nfoo\ntrees 1\nL 0\n")?;
        assert_model_err("VA_GBDT 1\nbase 0\nlr 0.1\ndim 28\nnaics xyz\ntrees 1\nL 0\n")?;
        assert_model_err(&leaf_model_text("VA_GBDT 1", 28, 1, "X 0"))?;
        assert_model_err(&leaf_model_text("VA_GBDT 1", 28, 1, "L"))?;
        assert_model_err(&leaf_model_text("VA_GBDT 1", 28, 1, "B 0"))?;
        assert_model_err(&leaf_model_text("VA_GBDT 1", 28, 1, "L 0 extra"))?;
        assert_model_err("VA_GBDT 1\nxxx 0\nlr 0.1\ndim 28\nnaics\ntrees 1\nL 0\n")?;
        let mut nodes = 0usize;
        let bushy = bushy_tree(13, &mut nodes, 8_193);
        assert_model_err(&leaf_model_text("VA_GBDT 1", 28, 1, &bushy))?;
        Ok(())
    }

    fn bushy_tree(depth: usize, count: &mut usize, limit: usize) -> String {
        *count += 1;
        if *count >= limit || depth == 0 {
            return "L 0".into();
        }
        format!(
            "B 0 0 {} {}",
            bushy_tree(depth - 1, count, limit),
            bushy_tree(depth - 1, count, limit)
        )
    }

    #[test]
    fn load_rejects_truncated_tree_lines() -> Result<()> {
        assert_model_err(&leaf_model_text("VA_GBDT 1", 28, 2, "L 0"))?;
        Ok(())
    }

    #[test]
    fn save_load_filesystem_round_trip() -> Result<()> {
        let schema = FeatureSchema::new(vec![]);
        let xs = vec![row4(2.0), row4(-2.0)];
        let ys = vec![1.0, -1.0];
        let model = GbdtModel::train(schema, &xs, &ys, &TrainConfig::default())?;
        let path = std::env::temp_dir().join(format!(
            "visit-attribution-model-{}-{}.va",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos())
        ));
        model.save(&path)?;
        let loaded = GbdtModel::load(&path)?;
        let _ = std::fs::remove_file(&path);
        assert!(
            (loaded.predict_raw(crate::test_util::some(xs.first())?)?
                - model.predict_raw(crate::test_util::some(xs.first())?)?)
            .abs()
                < 1e-9
        );
        Ok(())
    }

    #[test]
    fn eval_tree_and_train_helpers_cover_oob_paths() -> Result<()> {
        let branch = Node::Branch {
            feature: 99,
            threshold: 0.0,
            left: Box::new(Node::Leaf { value: 1.0 }),
            right: Box::new(Node::Leaf { value: -1.0 }),
        };
        let err = crate::test_util::err(eval_tree(&branch, &[0.0]))?;
        assert!(matches!(err, Error::Model(_)));
        let split = Node::Branch {
            feature: 0,
            threshold: -1.0,
            left: Box::new(Node::Leaf { value: 1.0 }),
            right: Box::new(Node::Leaf { value: -1.0 }),
        };
        assert!((eval_tree(&split, &[0.0])? + 1.0).abs() < f64::EPSILON);
        let xs = [vec![1.0, 0.0]];
        let ys = [1.0];
        let (res, sample) = train::build_residuals(&xs, &ys, &[], &[0]);
        assert!(res.is_empty() && sample.is_empty());
        train::apply_tree(&mut [], &xs, &Node::Leaf { value: 0.0 }, 0.1)?;
        let _leaf = train::build_tree(&[], &[], &TrainConfig::default(), 0);
        let schema = FeatureSchema::new(vec![4451]);
        let dim = schema.dim();
        let row = |first: f64| {
            let mut v = vec![0.0; dim];
            v[0] = first;
            v
        };
        let with_naics = GbdtModel::train(
            schema,
            &[row(2.0), row(-2.0)],
            &[1.0, -1.0],
            &TrainConfig::default(),
        )?;
        assert!(with_naics.to_string_format().contains("naics 4451"));
        Ok(())
    }
}
