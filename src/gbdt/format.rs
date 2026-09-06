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

fn ensure_model_size(text: &str) -> Result<()> {
    if text.len() > MAX_MODEL_BYTES {
        return Err(Error::Model(format!(
            "model text length {} exceeds limit {MAX_MODEL_BYTES}",
            text.len()
        )));
    }
    Ok(())
}

fn nonempty_lines(text: &str) -> impl Iterator<Item = &str> {
    text.lines().filter(|l| !l.trim().is_empty())
}

fn parse_model_body(text: &str) -> Result<GbdtModel> {
    let mut lines = nonempty_lines(text);
    let header = lines.next().ok_or_else(|| Error::Model("empty model file".into()))?;
    parse_header(header)?;
    let (base_score, learning_rate, schema, n_trees) = parse_meta(&mut lines)?;
    let trees = parse_trees(&mut lines, n_trees, schema.dim())?;
    Ok(GbdtModel {
        schema,
        base_score,
        learning_rate,
        trees,
    })
}

/// Parse a model previously written by [`to_string_format`].
///
/// # Errors
///
/// Returns [`Error::Model`] when the text is oversized, truncated, or invalid.
pub(super) fn from_string_format(text: &str) -> Result<GbdtModel> {
    ensure_model_size(text)?;
    parse_model_body(text)
}

fn parse_header(header: &str) -> Result<()> {
    if header != "VA_GBDT 1" {
        return Err(Error::Model(format!("unsupported header: {header}")));
    }
    Ok(())
}

fn parse_next_keyed<'a, T: FromStr>(
    lines: &mut impl Iterator<Item = &'a str>,
    key: &str,
    missing: &str,
) -> Result<T> {
    parse_keyed(
        lines.next().ok_or_else(|| Error::Model(missing.into()))?,
        key,
    )
}

fn parse_naics_line(line: &str) -> Result<Vec<u32>> {
    let mut naics_parts = line.split_whitespace();
    if naics_parts.next() != Some("naics") {
        return Err(Error::Model("expected naics line".into()));
    }
    let mut naics4 = Vec::new();
    for part in naics_parts {
        naics4.push(part.parse::<u32>().map_err(|_| Error::Model(format!("bad naics: {part}")))?);
    }
    Ok(naics4)
}

fn parse_naics_from_lines<'a>(lines: &mut impl Iterator<Item = &'a str>) -> Result<Vec<u32>> {
    let line = lines.next().ok_or_else(|| Error::Model("missing naics".into()))?;
    parse_naics_line(line)
}

fn finish_meta(dim: usize, naics4: Vec<u32>, n_trees: usize) -> Result<(FeatureSchema, usize)> {
    let schema = FeatureSchema { naics4 };
    if dim != schema.dim() {
        return Err(Error::Model(format!(
            "dim {dim} does not match schema width {}",
            schema.dim()
        )));
    }
    if n_trees > MAX_TREES {
        return Err(Error::Model(format!(
            "trees {n_trees} exceeds limit {MAX_TREES}"
        )));
    }
    Ok((schema, n_trees))
}

fn parse_meta_header<'a>(lines: &mut impl Iterator<Item = &'a str>) -> Result<(f64, f64, usize)> {
    let base_score = parse_next_keyed(lines, "base", "missing base")?;
    let learning_rate = parse_next_keyed(lines, "lr", "missing lr")?;
    let dim = parse_next_keyed(lines, "dim", "missing dim")?;
    Ok((base_score, learning_rate, dim))
}

fn parse_meta<'a>(
    lines: &mut impl Iterator<Item = &'a str>,
) -> Result<(f64, f64, FeatureSchema, usize)> {
    let (base_score, learning_rate, dim) = parse_meta_header(lines)?;
    let naics4 = parse_naics_from_lines(lines)?;
    let n_trees = parse_next_keyed(lines, "trees", "missing trees")?;
    let (schema, n_trees) = finish_meta(dim, naics4, n_trees)?;
    Ok((base_score, learning_rate, schema, n_trees))
}

fn parse_one_tree(line: &str, dim: usize) -> Result<Node> {
    let mut toks = line.split_whitespace().peekable();
    let mut nodes = 0usize;
    let tree = parse_node(&mut toks, 0, &mut nodes)?;
    if toks.next().is_some() {
        return Err(Error::Model("trailing tokens on tree line".into()));
    }
    validate_node(&tree, dim)?;
    Ok(tree)
}

fn parse_trees<'a>(
    lines: &mut impl Iterator<Item = &'a str>,
    n_trees: usize,
    dim: usize,
) -> Result<Vec<Node>> {
    let mut trees = Vec::with_capacity(n_trees);
    for _ in 0..n_trees {
        let line = lines.next().ok_or_else(|| Error::Model("missing tree line".into()))?;
        trees.push(parse_one_tree(line, dim)?);
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

fn parse_token<'a, T: FromStr>(
    toks: &mut impl Iterator<Item = &'a str>,
    missing: &str,
    bad: &str,
) -> Result<T> {
    toks.next()
        .ok_or_else(|| Error::Model(missing.into()))?
        .parse::<T>()
        .map_err(|_| Error::Model(bad.into()))
}

fn parse_leaf<'a>(toks: &mut impl Iterator<Item = &'a str>) -> Result<Node> {
    let value = parse_token(toks, "missing leaf value", "bad leaf value")?;
    Ok(Node::Leaf { value })
}

fn ensure_tree_depth(depth: usize) -> Result<()> {
    if depth >= MAX_TREE_DEPTH {
        return Err(Error::Model(format!(
            "tree depth exceeds limit {MAX_TREE_DEPTH}"
        )));
    }
    Ok(())
}

fn bump_node_count(nodes: &mut usize) -> Result<()> {
    *nodes += 1;
    if *nodes > MAX_NODES_PER_TREE {
        return Err(Error::Model(format!(
            "tree exceeds node limit {MAX_NODES_PER_TREE}"
        )));
    }
    Ok(())
}

fn parse_branch_children<'a>(
    toks: &mut std::iter::Peekable<impl Iterator<Item = &'a str>>,
    depth: usize,
    nodes: &mut usize,
    feature: usize,
    threshold: f64,
) -> Result<Node> {
    let left = Box::new(parse_node(toks, depth + 1, nodes)?);
    let right = Box::new(parse_node(toks, depth + 1, nodes)?);
    Ok(Node::Branch {
        feature,
        threshold,
        left,
        right,
    })
}

fn parse_branch<'a>(
    toks: &mut std::iter::Peekable<impl Iterator<Item = &'a str>>,
    depth: usize,
    nodes: &mut usize,
) -> Result<Node> {
    ensure_tree_depth(depth)?;
    let feature = parse_token(toks, "missing feature", "bad feature")?;
    let threshold = parse_token(toks, "missing threshold", "bad threshold")?;
    parse_branch_children(toks, depth, nodes, feature, threshold)
}

fn parse_tagged_node<'a>(
    tag: &str,
    toks: &mut std::iter::Peekable<impl Iterator<Item = &'a str>>,
    depth: usize,
    nodes: &mut usize,
) -> Result<Node> {
    match tag {
        "L" => parse_leaf(toks),
        "B" => parse_branch(toks, depth, nodes),
        other => Err(Error::Model(format!("unknown node tag: {other}"))),
    }
}

fn parse_node<'a>(
    toks: &mut std::iter::Peekable<impl Iterator<Item = &'a str>>,
    depth: usize,
    nodes: &mut usize,
) -> Result<Node> {
    bump_node_count(nodes)?;
    let tag = toks.next().ok_or_else(|| Error::Model("unexpected end of tree".into()))?;
    parse_tagged_node(tag, toks, depth, nodes)
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
