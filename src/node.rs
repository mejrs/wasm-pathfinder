use crate::feature::Feature;
use crate::pathfinding::Position;
use std::cmp::Ordering;
use std::hash::Hash;
use std::hash::Hasher;

#[derive(Clone, Copy)]
pub struct Node {
    pub x: i32,
    pub y: i32,
    pub cost: i32,
    pub heuristic: i32,
    pub parent: Option<(i32, i32)>,
}

impl Eq for Node {}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Node) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Node {
    fn cmp(&self, other: &Node) -> Ordering {
        other
            .cost
            .cmp(&self.cost)
            .then_with(|| self.x.cmp(&other.x))
            .then_with(|| self.y.cmp(&other.y))
    }
}

impl Hash for Node {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.x.hash(state);
        self.y.hash(state);
    }
}

impl Node {
    pub fn as_tuple(&self) -> (i32, i32) {
        let x = self.x;
        let y = self.y;
        (x, y)
    }

    pub fn to_position(self, plane: &i32, feature: &Feature) -> Position {
        let new_x = self.x + feature.offset.x;
        let new_y = self.y + feature.offset.y;
        let pos = Position::new(
            plane,
            &new_x,
            &new_y,
            format!("{}_{}", plane, feature.feature_name),
        );
        pos
    }
}
