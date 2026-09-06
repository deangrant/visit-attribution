use super::*;
use crate::error::Result;

fn row4(first: f64) -> Vec<f64> {
    // Empty-naics schema dim is 4 + 24 UNK.
    let mut v = vec![0.0; 28];
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
