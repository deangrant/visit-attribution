//! Axis-aligned bbox quadtree for place-join pruning.

use crate::geo::BBox;

const MAX_LEAF: usize = 16;
const MAX_DEPTH: u8 = 16;
/// Minimum root extent in degrees so subdivision has room to split.
const MIN_EXTENT_DEG: f64 = 1e-6;

/// Quadtree over place bounding boxes (indices into a parallel place list).
#[derive(Debug, Clone)]
pub(crate) struct BBoxQuadtree {
    root: Option<QuadNode>,
}

#[derive(Debug, Clone)]
enum QuadNode {
    Leaf {
        bounds: BBox,
        indices: Vec<usize>,
    },
    Branch {
        bounds: BBox,
        children: Box<[QuadNode; 4]>,
    },
}

impl BBoxQuadtree {
    /// Build a tree over `bboxes` (one entry per place index).
    #[must_use]
    pub(crate) fn build(bboxes: &[BBox]) -> Self {
        if bboxes.is_empty() {
            return Self { root: None };
        }
        let mut root_bounds = match bboxes.split_first() {
            Some((first, rest)) => {
                let mut bounds = *first;
                for b in rest {
                    bounds = bounds.union(*b);
                }
                bounds
            }
            None => return Self { root: None },
        };
        root_bounds = ensure_extent(root_bounds);
        let mut root = QuadNode::Leaf {
            bounds: root_bounds,
            indices: Vec::new(),
        };
        for idx in 0..bboxes.len() {
            insert(&mut root, idx, bboxes, 0);
        }
        Self { root: Some(root) }
    }

    /// Collect place indices whose node regions may intersect `query`.
    pub(crate) fn query(&self, query: BBox, out: &mut Vec<usize>) {
        if let Some(root) = &self.root {
            query_node(root, query, out);
        }
    }
}

fn ensure_extent(mut b: BBox) -> BBox {
    if b.max_lat - b.min_lat < MIN_EXTENT_DEG {
        let mid = (b.min_lat + b.max_lat) * 0.5;
        b.min_lat = mid - MIN_EXTENT_DEG * 0.5;
        b.max_lat = mid + MIN_EXTENT_DEG * 0.5;
    }
    if b.max_lon - b.min_lon < MIN_EXTENT_DEG {
        let mid = (b.min_lon + b.max_lon) * 0.5;
        b.min_lon = mid - MIN_EXTENT_DEG * 0.5;
        b.max_lon = mid + MIN_EXTENT_DEG * 0.5;
    }
    b
}

fn bounds_of(node: &QuadNode) -> BBox {
    match node {
        QuadNode::Leaf { bounds, .. } | QuadNode::Branch { bounds, .. } => *bounds,
    }
}

fn quadrants(b: BBox) -> [BBox; 4] {
    let mid_lat = (b.min_lat + b.max_lat) * 0.5;
    let mid_lon = (b.min_lon + b.max_lon) * 0.5;
    [
        BBox {
            min_lat: mid_lat,
            max_lat: b.max_lat,
            min_lon: b.min_lon,
            max_lon: mid_lon,
        },
        BBox {
            min_lat: mid_lat,
            max_lat: b.max_lat,
            min_lon: mid_lon,
            max_lon: b.max_lon,
        },
        BBox {
            min_lat: b.min_lat,
            max_lat: mid_lat,
            min_lon: b.min_lon,
            max_lon: mid_lon,
        },
        BBox {
            min_lat: b.min_lat,
            max_lat: mid_lat,
            min_lon: mid_lon,
            max_lon: b.max_lon,
        },
    ]
}

fn insert(node: &mut QuadNode, idx: usize, bboxes: &[BBox], depth: u8) {
    match node {
        QuadNode::Leaf { bounds, indices } => {
            indices.push(idx);
            if indices.len() <= MAX_LEAF || depth >= MAX_DEPTH {
                return;
            }
            let bounds = *bounds;
            let old = std::mem::take(indices);
            let quads = quadrants(bounds);
            let mut children = quads.map(|bounds| QuadNode::Leaf {
                bounds,
                indices: Vec::new(),
            });
            for old_idx in old {
                insert_into_children(&mut children, old_idx, bboxes, depth);
            }
            *node = QuadNode::Branch {
                bounds,
                children: Box::new(children),
            };
        }
        QuadNode::Branch { children, .. } => {
            insert_into_children(children, idx, bboxes, depth);
        }
    }
}

fn insert_into_children(children: &mut [QuadNode; 4], idx: usize, bboxes: &[BBox], depth: u8) {
    let Some(&item) = bboxes.get(idx) else {
        return;
    };
    let mut hit = false;
    for child in children.iter_mut() {
        if bounds_of(child).intersects(item) {
            insert(child, idx, bboxes, depth.saturating_add(1));
            hit = true;
        }
    }
    if !hit {
        if let Some(first) = children.first_mut() {
            insert(first, idx, bboxes, depth.saturating_add(1));
        }
    }
}

fn query_node(node: &QuadNode, query: BBox, out: &mut Vec<usize>) {
    if !bounds_of(node).intersects(query) {
        return;
    }
    match node {
        QuadNode::Leaf { indices, .. } => out.extend_from_slice(indices),
        QuadNode::Branch { children, .. } => {
            for child in children.iter() {
                query_node(child, query, out);
            }
        }
    }
}
