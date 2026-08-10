//! In-crate gradient-boosted decision trees for preference scores.

use std::fs;
use std::path::Path;

use crate::error::{Error, Result};
use crate::features::FeatureSchema;

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
        let base_score = 0.0;
        let mut preds = vec![base_score; xs.len()];
        let mut trees = Vec::with_capacity(config.n_trees);
        let indices: Vec<usize> = (0..xs.len()).step_by(config.subsample_stride).collect();
        for _ in 0..config.n_trees {
            let mut residuals = Vec::with_capacity(indices.len());
            let mut sample_x = Vec::with_capacity(indices.len());
            for &i in &indices {
                let p = sigmoid(preds[i]);
                // Map ±1 labels to {0,1} for the logistic residual.
                let y01 = if ys[i] > 0.0 { 1.0 } else { 0.0 };
                residuals.push(y01 - p);
                sample_x.push(xs[i].clone());
            }
            let tree = build_tree(&sample_x, &residuals, config, 0);
            for (i, row) in xs.iter().enumerate() {
                preds[i] += config.learning_rate * eval_tree(&tree, row);
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
    #[must_use]
    pub fn predict_raw(&self, x: &[f64]) -> f64 {
        let mut s = self.base_score;
        for tree in &self.trees {
            s += self.learning_rate * eval_tree(tree, x);
        }
        s
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
        let mut out = String::new();
        out.push_str("VA_GBDT 1\n");
        out.push_str("base ");
        out.push_str(&self.base_score.to_string());
        out.push('\n');
        out.push_str("lr ");
        out.push_str(&self.learning_rate.to_string());
        out.push('\n');
        out.push_str("dim ");
        out.push_str(&self.schema.dim().to_string());
        out.push('\n');
        out.push_str("naics");
        for code in &self.schema.naics4 {
            out.push(' ');
            out.push_str(&code.to_string());
        }
        out.push('\n');
        out.push_str("trees ");
        out.push_str(&self.trees.len().to_string());
        out.push('\n');
        for tree in &self.trees {
            write_node(&mut out, tree);
            out.push('\n');
        }
        out
    }

    fn from_string_format(text: &str) -> Result<Self> {
        let mut lines = text.lines().filter(|l| !l.trim().is_empty());
        let header = lines.next().ok_or_else(|| Error::Model("empty model file".into()))?;
        if header != "VA_GBDT 1" {
            return Err(Error::Model(format!("unsupported header: {header}")));
        }
        let base_score = parse_keyed_f64(
            lines.next().ok_or_else(|| Error::Model("missing base".into()))?,
            "base",
        )?;
        let learning_rate = parse_keyed_f64(
            lines.next().ok_or_else(|| Error::Model("missing lr".into()))?,
            "lr",
        )?;
        let _dim = parse_keyed_usize(
            lines.next().ok_or_else(|| Error::Model("missing dim".into()))?,
            "dim",
        )?;
        let naics_line = lines.next().ok_or_else(|| Error::Model("missing naics".into()))?;
        let mut naics_parts = naics_line.split_whitespace();
        if naics_parts.next() != Some("naics") {
            return Err(Error::Model("expected naics line".into()));
        }
        let mut naics4 = Vec::new();
        for part in naics_parts {
            naics4
                .push(part.parse::<u32>().map_err(|_| Error::Model(format!("bad naics: {part}")))?);
        }
        let n_trees = parse_keyed_usize(
            lines.next().ok_or_else(|| Error::Model("missing trees".into()))?,
            "trees",
        )?;
        let mut trees = Vec::with_capacity(n_trees);
        for _ in 0..n_trees {
            let line = lines.next().ok_or_else(|| Error::Model("missing tree line".into()))?;
            let mut toks = line.split_whitespace().peekable();
            trees.push(parse_node(&mut toks)?);
        }
        Ok(Self {
            schema: FeatureSchema { naics4 },
            base_score,
            learning_rate,
            trees,
        })
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
    if xs.len() < config.min_leaf * 2
        || depth >= config.max_depth
        || ys.iter().all(|&y| (y - ys[0]).abs() < 1e-12)
    {
        return Node::Leaf { value: mean(ys) };
    }
    let Some((feature, threshold, left_idx, right_idx)) = best_split(xs, ys, config) else {
        return Node::Leaf { value: mean(ys) };
    };
    let left_xs: Vec<Vec<f64>> = left_idx.iter().map(|&i| xs[i].clone()).collect();
    let left_ys: Vec<f64> = left_idx.iter().map(|&i| ys[i]).collect();
    let right_xs: Vec<Vec<f64>> = right_idx.iter().map(|&i| xs[i].clone()).collect();
    let right_ys: Vec<f64> = right_idx.iter().map(|&i| ys[i]).collect();
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
    let dim = xs[0].len();
    let mut best_gain = 0.0;
    let mut best = None;
    for f in 0..dim {
        let mut vals: Vec<f64> = xs.iter().map(|r| r[f]).collect();
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
                if row[f] <= threshold {
                    left_idx.push(i);
                } else {
                    right_idx.push(i);
                }
            }
            if left_idx.len() < config.min_leaf || right_idx.len() < config.min_leaf {
                continue;
            }
            let parent_var = variance(ys);
            let left_ys: Vec<f64> = left_idx.iter().map(|&i| ys[i]).collect();
            let right_ys: Vec<f64> = right_idx.iter().map(|&i| ys[i]).collect();
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

fn eval_tree(node: &Node, x: &[f64]) -> f64 {
    match node {
        Node::Leaf { value } => *value,
        Node::Branch {
            feature,
            threshold,
            left,
            right,
        } => {
            if x[*feature] <= *threshold {
                eval_tree(left, x)
            } else {
                eval_tree(right, x)
            }
        }
    }
}

fn write_node(out: &mut String, node: &Node) {
    match node {
        Node::Leaf { value } => {
            out.push('L');
            out.push(' ');
            out.push_str(&value.to_string());
        }
        Node::Branch {
            feature,
            threshold,
            left,
            right,
        } => {
            out.push('B');
            out.push(' ');
            out.push_str(&feature.to_string());
            out.push(' ');
            out.push_str(&threshold.to_string());
            out.push(' ');
            write_node(out, left);
            out.push(' ');
            write_node(out, right);
        }
    }
}

fn parse_node<'a>(toks: &mut std::iter::Peekable<impl Iterator<Item = &'a str>>) -> Result<Node> {
    let tag = toks.next().ok_or_else(|| Error::Model("unexpected end of tree".into()))?;
    match tag {
        "L" => {
            let value = toks
                .next()
                .ok_or_else(|| Error::Model("missing leaf value".into()))?
                .parse::<f64>()
                .map_err(|_| Error::Model("bad leaf value".into()))?;
            Ok(Node::Leaf { value })
        }
        "B" => {
            let feature = toks
                .next()
                .ok_or_else(|| Error::Model("missing feature".into()))?
                .parse::<usize>()
                .map_err(|_| Error::Model("bad feature".into()))?;
            let threshold = toks
                .next()
                .ok_or_else(|| Error::Model("missing threshold".into()))?
                .parse::<f64>()
                .map_err(|_| Error::Model("bad threshold".into()))?;
            let left = Box::new(parse_node(toks)?);
            let right = Box::new(parse_node(toks)?);
            Ok(Node::Branch {
                feature,
                threshold,
                left,
                right,
            })
        }
        other => Err(Error::Model(format!("unknown node tag: {other}"))),
    }
}

fn parse_keyed_f64(line: &str, key: &str) -> Result<f64> {
    let mut parts = line.split_whitespace();
    if parts.next() != Some(key) {
        return Err(Error::Model(format!("expected key {key}")));
    }
    parts
        .next()
        .ok_or_else(|| Error::Model(format!("missing value for {key}")))?
        .parse::<f64>()
        .map_err(|_| Error::Model(format!("bad f64 for {key}")))
}

fn parse_keyed_usize(line: &str, key: &str) -> Result<usize> {
    let mut parts = line.split_whitespace();
    if parts.next() != Some(key) {
        return Err(Error::Model(format!("expected key {key}")));
    }
    parts
        .next()
        .ok_or_else(|| Error::Model(format!("missing value for {key}")))?
        .parse::<usize>()
        .map_err(|_| Error::Model(format!("bad usize for {key}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learns_simple_preference_direction() {
        let schema = FeatureSchema::new(vec![]);
        // Difference feature: positive when left is better on first dim.
        let xs = vec![
            vec![2.0, 0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0, 0.0],
            vec![-2.0, 0.0, 0.0, 0.0],
            vec![-1.0, 0.0, 0.0, 0.0],
        ];
        let ys = vec![1.0, 1.0, -1.0, -1.0];
        let model = GbdtModel::train(schema, &xs, &ys, &TrainConfig::default()).unwrap();
        assert!(model.predict_raw(&xs[0]) > 0.0);
        assert!(model.predict_raw(&xs[2]) < 0.0);
        let text = model.to_string_format();
        let loaded = GbdtModel::from_string_format(&text).unwrap();
        assert!((loaded.predict_raw(&xs[0]) - model.predict_raw(&xs[0])).abs() < 1e-9);
    }
}
