//! Text serialization for [`super::GbdtModel`].

use std::str::FromStr;

use crate::error::{Error, Result};
use crate::features::FeatureSchema;

use super::{GbdtModel, Node, MAX_MODEL_BYTES, MAX_NODES_PER_TREE, MAX_TREES, MAX_TREE_DEPTH};

/// Serialize a model to the versioned `VA_GBDT` text format.
pub(super) fn to_string_format(model: &GbdtModel) -> String {
    let mut out = String::new();
    out.push_str("VA_GBDT 1\n");
    out.push_str("base ");
    out.push_str(&model.base_score.to_string());
    out.push('\n');
    out.push_str("lr ");
    out.push_str(&model.learning_rate.to_string());
    out.push('\n');
    out.push_str("dim ");
    out.push_str(&model.schema.dim().to_string());
    out.push('\n');
    out.push_str("naics");
    for code in &model.schema.naics4 {
        out.push(' ');
        out.push_str(&code.to_string());
    }
    out.push('\n');
    out.push_str("trees ");
    out.push_str(&model.trees.len().to_string());
    out.push('\n');
    for tree in &model.trees {
        write_node(&mut out, tree);
        out.push('\n');
    }
    out
}

/// Parse a model previously written by [`to_string_format`].
pub(super) fn from_string_format(text: &str) -> Result<GbdtModel> {
    if text.len() > MAX_MODEL_BYTES {
        return Err(Error::Model(format!(
            "model text length {} exceeds limit {MAX_MODEL_BYTES}",
            text.len()
        )));
    }
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header = lines.next().ok_or_else(|| Error::Model("empty model file".into()))?;
    parse_header(header)?;
    let (base_score, learning_rate, schema, n_trees) = parse_meta(&mut lines)?;
    let dim = schema.dim();
    let trees = parse_trees(&mut lines, n_trees, dim)?;
    Ok(GbdtModel {
        schema,
        base_score,
        learning_rate,
        trees,
    })
}

fn parse_header(header: &str) -> Result<()> {
    if header != "VA_GBDT 1" {
        return Err(Error::Model(format!("unsupported header: {header}")));
    }
    Ok(())
}

fn parse_meta<'a>(
    lines: &mut impl Iterator<Item = &'a str>,
) -> Result<(f64, f64, FeatureSchema, usize)> {
    let base_score = parse_keyed(
        lines.next().ok_or_else(|| Error::Model("missing base".into()))?,
        "base",
    )?;
    let learning_rate = parse_keyed(
        lines.next().ok_or_else(|| Error::Model("missing lr".into()))?,
        "lr",
    )?;
    let dim: usize = parse_keyed(
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
        naics4.push(part.parse::<u32>().map_err(|_| Error::Model(format!("bad naics: {part}")))?);
    }
    let schema = FeatureSchema { naics4 };
    if dim != schema.dim() {
        return Err(Error::Model(format!(
            "dim {dim} does not match schema width {}",
            schema.dim()
        )));
    }
    let n_trees = parse_keyed(
        lines.next().ok_or_else(|| Error::Model("missing trees".into()))?,
        "trees",
    )?;
    if n_trees > MAX_TREES {
        return Err(Error::Model(format!(
            "trees {n_trees} exceeds limit {MAX_TREES}"
        )));
    }
    Ok((base_score, learning_rate, schema, n_trees))
}

fn parse_trees<'a>(
    lines: &mut impl Iterator<Item = &'a str>,
    n_trees: usize,
    dim: usize,
) -> Result<Vec<Node>> {
    let mut trees = Vec::with_capacity(n_trees);
    for _ in 0..n_trees {
        let line = lines.next().ok_or_else(|| Error::Model("missing tree line".into()))?;
        let mut toks = line.split_whitespace().peekable();
        let mut nodes = 0usize;
        let tree = parse_node(&mut toks, 0, &mut nodes)?;
        if toks.next().is_some() {
            return Err(Error::Model("trailing tokens on tree line".into()));
        }
        validate_node(&tree, dim)?;
        trees.push(tree);
    }
    Ok(trees)
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

fn parse_node<'a>(
    toks: &mut std::iter::Peekable<impl Iterator<Item = &'a str>>,
    depth: usize,
    nodes: &mut usize,
) -> Result<Node> {
    *nodes += 1;
    if *nodes > MAX_NODES_PER_TREE {
        return Err(Error::Model(format!(
            "tree exceeds node limit {MAX_NODES_PER_TREE}"
        )));
    }
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
            if depth >= MAX_TREE_DEPTH {
                return Err(Error::Model(format!(
                    "tree depth exceeds limit {MAX_TREE_DEPTH}"
                )));
            }
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
            let left = Box::new(parse_node(toks, depth + 1, nodes)?);
            let right = Box::new(parse_node(toks, depth + 1, nodes)?);
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

fn validate_node(node: &Node, dim: usize) -> Result<()> {
    match node {
        Node::Leaf { .. } => Ok(()),
        Node::Branch {
            feature,
            left,
            right,
            ..
        } => {
            if *feature >= dim {
                return Err(Error::Model(format!(
                    "tree feature index {feature} out of range for dim {dim}"
                )));
            }
            validate_node(left, dim)?;
            validate_node(right, dim)
        }
    }
}

fn parse_keyed<T: FromStr>(line: &str, key: &str) -> Result<T> {
    let mut parts = line.split_whitespace();
    if parts.next() != Some(key) {
        return Err(Error::Model(format!("expected key {key}")));
    }
    parts
        .next()
        .ok_or_else(|| Error::Model(format!("missing value for {key}")))?
        .parse::<T>()
        .map_err(|_| Error::Model(format!("bad value for {key}")))
}
