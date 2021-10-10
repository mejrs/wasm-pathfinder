#![feature(generators, generator_trait)]
#![feature(associated_type_bounds)]

mod utils;

use futures::future::join_all;
use futures::try_join;
use gen_iter::GenIter;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::{console, Request, RequestInit, RequestMode, Response};

#[cfg(test)]
mod tests {
    use factorial::Factorial;
    #[test]
    fn test_vec_permutations() {
        for a in 1..10 {
            for b in 1..10 {
                let iter = vec![(0, a as usize), (1, b as usize)];
                let permutations: Vec<_> = super::unique_permutations(iter).collect();

                let n = (a + b).factorial() / (a.factorial() * b.factorial());
                assert!(permutations.len() == n);
            }
        }
    }
    use super::Accumulator;

    #[test]
    fn test_accumulator_cumulative() {
        let result: Vec<i32> = (1..6).accumulate(|x, y| x + y).collect();
        let expected_result: Vec<i32> = vec![1, 3, 6, 10, 15];
        assert!(result == expected_result)
    }

    #[test]
    fn test_accumulator_rolling_factorial() {
        let result: Vec<i32> = (1..6).accumulate(|x, y| x * y).collect();
        let expected_result: Vec<i32> = vec![1, 2, 6, 24, 120];
        assert!(result == expected_result)
    }
}

// Recursive functions for looping over all unique combinations of an iterable
// Called with vec![(element, repeats), ..., (nth_element, repeats)]
// Elements must be unique

pub fn unique_permutations<Sequence, I: 'static>(
    items: Sequence,
) -> Box<dyn Iterator<Item = Vec<I>>>
where
    Sequence: IntoIterator<Item = (I, usize)>,
    I: Clone + Eq + Hash,
{
    fn internal_unique_permutations<I: 'static>(
        items: Vec<(I, usize)>,
    ) -> Box<dyn Iterator<Item = Vec<I>>>
    where
        I: Clone + Eq + Hash,
    {
        Box::new(GenIter(move || {
            // Base case
            if items.len() == 1 {
                let (item, repeat) = items[0].clone();
                yield vec![item; repeat];
            } else {
                for (index, (item, repeats)) in items.clone().into_iter().enumerate() {
                    let mut remaining_items = items.clone();

                    if repeats == 1 {
                        remaining_items.remove(index);
                    } else {
                        remaining_items[index].1 = repeats - 1;
                    }

                    let inner_permutations = internal_unique_permutations(remaining_items);
                    for inner_permutation in inner_permutations {
                        let local_item = item.clone();
                        let mut result = vec![local_item];
                        result.extend(inner_permutation);
                        yield result;
                    }
                }
            }
        }))
    }

    //convert items into a vector
    let vec_sequence: Vec<_> = items.into_iter().collect();

    if vec_sequence.is_empty() {
        // there is one way to draw nothing from an empty set
        Box::new(vec![vec![]].into_iter())
    } else {
        // Enforce uniqueness of elements
        let mut uniq = HashSet::new();
        assert!(vec_sequence
            .clone()
            .into_iter()
            .all(move |x| uniq.insert(x)));

        internal_unique_permutations(vec_sequence)
    }
}

struct Accumulate<I, F>
where
    I: Iterator,
    F: Fn(I::Item, I::Item) -> I::Item,
{
    accum: Option<I::Item>,
    underlying: I,
    acc_fn: F,
}

impl<I, F> Iterator for Accumulate<I, F>
where
    I: Iterator,
    I::Item: Hash + Eq + Clone,
    F: Fn(I::Item, I::Item) -> I::Item,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        match self.underlying.next() {
            Some(x) => {
                let new_accum = match self.accum.clone() {
                    Some(accum) => (self.acc_fn)(accum, x),
                    None => x,
                };
                self.accum = Some(new_accum.clone());
                Some(new_accum)
            }
            None => None,
        }
    }
}

trait Accumulator: Iterator {
    fn accumulate<F>(self, f: F) -> Accumulate<Self, F>
    where
        F: Fn(Self::Item, Self::Item) -> Self::Item,
        Self::Item: Clone,
        Self: Sized,
        Self: Iterator,
    {
        Accumulate {
            accum: None,
            underlying: self,
            acc_fn: f,
        }
    }
}

impl<I: Iterator> Accumulator for I {}

// Returns true if all items in a sequence are distinct
fn all_unique<T>(iter: T) -> bool
where
    T: IntoIterator + Clone,
    T::Item: Clone + Eq + Hash,
{
    let mut uniq = HashSet::new();
    iter.into_iter().all(move |x| uniq.insert(x))
}

fn pairwise<I>(sequence: I) -> impl Iterator<Item = (I::Item, I::Item)>
where
    I: IntoIterator + Clone,
{
    let second_iter = sequence.clone().into_iter().skip(1);
    sequence.into_iter().zip(second_iter)
}

fn prepend_vec_append<T>(a: T, sequence: Vec<T>, z: T) -> Vec<T> {
    let mut vec_a = vec![a];
    vec_a.extend(sequence);
    vec_a.extend(vec![z]);
    vec_a
}

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

const ALPHA: f32 = 0.9;

// WASM interface to Pathfinding::race
#[wasm_bindgen]
pub async fn race(
    plane_1: i32,
    x_1: i32,
    y_1: i32,
    feature_1: i32,
    plane_2: i32,
    x_2: i32,
    y_2: i32,
    feature_2: i32,
) -> Result<JsValue, JsValue> {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    let feature_key_1 = format!("{}_{}", plane_1, feature_1);
    let position_1 = Position::new(&plane_1, &x_1, &y_1, feature_key_1);

    let feature_key_2 = format!("{}_{}", plane_2, feature_2);
    let position_2 = Position::new(&plane_2, &x_2, &y_2, feature_key_2);
    //console::log_1(&JsValue::from_serde(&position_1).unwrap());
    //console::log_1(&JsValue::from_serde(&position_2).unwrap());

    let result = Pathfinding::race(position_1, position_2).await?;

    let js_result = JsValue::from_serde(&result).unwrap();

    Ok(js_result)
}

#[wasm_bindgen]
pub async fn dive(plane_1: i32, x_1: i32, y_1: i32, feature_1: i32) -> Result<JsValue, JsValue> {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    let feature_key_1 = format!("{}_{}", plane_1, feature_1);
    let position_1 = Position::new(&plane_1, &x_1, &y_1, feature_key_1);

    let result = Dive::dive(position_1).await?;

    let js_result = JsValue::from_serde(&result).unwrap();

    Ok(js_result)
}

pub struct Dive {}

impl Dive {
    pub async fn dive(pos: Position) -> Result<Vec<Position>, JsValue> {
        let plane = pos.plane;
        let x = pos.x;
        let y = pos.y;
        let feature_key = pos.feature_key;

        let feature = Feature::new(&feature_key).await;
        let coords: Vec<(i32, i32)> = (-10..11).cartesian_product(-10..11).collect();

        let walkable_tiles: HashSet<(i32, i32)> = coords
            .into_iter()
            .map(|(dx, dy)| {
                Position::new(&plane, &(x + dx), &(y + dy), feature_key.clone())
                    .into_node_in_feature(&feature)
            })
            .filter(|p| feature.walkable_at(p))
            .map(|node| node.to_position(&plane, &feature))
            .map(|pos| (pos.x - x, pos.y - y))
            .collect();

        let walks: Vec<Position> = Dive::all_walk_permutations()
            .map(|path| {
                path.into_iter()
                    .take_while(|vector_step| walkable_tiles.contains(vector_step))
                    .map(|(dx, dy)| {
                        Position::new(&plane, &(x + dx), &(y + dy), feature_key.clone())
                            .into_node_in_feature(&feature)
                    })
                    .collect::<Vec<Node>>()
            })
            .filter(|path| feature.is_valid_path(path.clone()))
            .flatten()
            .unique()
            .map(|node| node.to_position(&plane, &feature))
            .collect();

        Ok(walks)
    }
    /*
    this is defined in code elsewhere
    const N :i32 = 4;
    const E :i32 = 5;
    const S :i32 = 6;
    const W :i32 = 7;
    const NW :i32 = 0;
    const SW :i32 = 1;
    const NE :i32 = 2;
    const SE :i32 = 3;
    */
    fn dive_move_sets() -> Vec<Vec<i32>> {
        vec![
            vec![N, W, NW],
            vec![N, E, NE],
            vec![S, W, SW],
            vec![S, E, SE],
        ]
    }

    fn all_walk_permutations() -> impl Iterator<Item = Vec<(i32, i32)>> {
        let a = Dive::walk_permutations(10, 10);
        let b = Dive::walk_permutations(-10, 10);
        let c = Dive::walk_permutations(10, -10);
        let d = Dive::walk_permutations(-10, -10);
        a.chain(b).chain(c).chain(d)
    }

    // Supplies an iterator containing all paths from (x,y) to (x+dx,y+dy)
    // See https://mathworld.wolfram.com/StaircaseWalk.html
    fn walk_permutations(dx: i32, dy: i32) -> impl Iterator<Item = Vec<(i32, i32)>> {
        let input = vec![
            ((0 as i32, dx.signum() as i32), dx.abs() as usize),
            ((dy.signum() as i32, 0 as i32), dy.abs() as usize),
        ];
        let all_steps = unique_permutations(input).map(|steps| {
            steps
                .into_iter()
                .accumulate(|a, b| (a.0 + b.0, a.1 + b.1))
                .collect()
        });
        all_steps
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FeatureSize {
    pub x: i32,
    pub y: i32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FeatureOffset {
    pub x: i32,
    pub y: i32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Feature {
    #[serde(alias = "f")]
    pub feature_name: u32,

    #[serde(alias = "s")]
    pub size: FeatureSize,

    #[serde(alias = "o")]
    pub offset: FeatureOffset,

    #[serde(skip_deserializing)]
    pub linear_vec: Vec<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CompressedFeature {
    #[serde(alias = "f")]
    feature_name: u32,

    #[serde(alias = "s")]
    size: FeatureSize,

    #[serde(alias = "o")]
    offset: FeatureOffset,

    #[serde(alias = "d")]
    compressed_vec: Vec<usize>,
}

impl CompressedFeature {
    pub async fn get_feature(feature_key: &str) -> Result<CompressedFeature, JsValue> {
        let mut opts = RequestInit::new();
        opts.method("GET");
        opts.mode(RequestMode::Cors);

        let url = format!(
            "wasm-pathfinder/data/features/feature_{}.json",
            &feature_key
        );

        let request = Request::new_with_str_and_init(&url, &opts)?;
        request
            .headers()
            .set("Accept", "application/vnd.github.v3+json")?;

        let window = web_sys::window().unwrap();

        let resp_value = JsFuture::from(window.fetch_with_request(&request)).await?;

        // `resp_value` is a `Response` object.
        assert!(resp_value.is_instance_of::<Response>());
        let resp: Response = resp_value.dyn_into().unwrap();

        if resp.ok() {
            let json = JsFuture::from(resp.json()?).await?;

            let feature_info: CompressedFeature = json.into_serde().unwrap();

            //console::log_2(&msg, &requested_url);
            Ok(feature_info)
        } else {
            let js_msg: JsValue = format!("{} Error fetching {}.", resp.status(), &url).into();
            Err(js_msg)
        }
    }

    fn decompress(self) -> Feature {
        Feature {
            feature_name: self.feature_name,
            size: self.size,
            offset: self.offset,
            linear_vec: self
                .compressed_vec
                .clone()
                .into_iter()
                .enumerate()
                .flat_map(|(i, len)| vec![i & 1 == 0; len])
                .collect(),
        }
    }
}

impl Feature {
    pub async fn new(feature_key: &str) -> Feature {
        CompressedFeature::get_feature(&feature_key)
            .await
            .expect("Feature could not be found")
            .decompress()
    }

    pub fn walkable_at(&self, node: &Node) -> bool {
        let size = &self.size;
        let value = &self.linear_vec[(4 * size.y * (1 + 4 * node.x) + 1 + 4 * node.y) as usize];
        *value
    }

    pub fn walkable_at_coord(&self, x: &i32, y: &i32) -> bool {
        let size = &self.size;

        let value = &self.linear_vec[(4 * size.y * (1 + 4 * x) + 1 + 4 * y) as usize];
        *value
    }

    pub fn is_valid_path(&self, path: Vec<Node>) -> bool {
        pairwise(path).all(|(node_a, node_b)| self.have_connectivity(node_a, node_b))
    }

    pub fn have_connectivity(&self, node_a: Node, node_b: Node) -> bool {
        let step: (i32, i32) = node_a.diff(&node_b);
        let subtiles = Directions::get_sub_tiles(step);
        let are_connected = subtiles.into_iter().all(|(sub_x, sub_y)| {
            self.linear_vec
                [(4 * self.size.y * (sub_x + 4 * node_a.x) + sub_y + 4 * node_a.y) as usize]
        });
        are_connected
    }

    //debug
    pub fn get_walkable_nodes(&self) -> Vec<(i32, i32)> {
        let dim_x = self.size.x;
        let dim_y = self.size.y;

        let mut container = Vec::new();

        for x in 0..dim_x {
            for y in 0..dim_y {
                if self.walkable_at_coord(&x, &y) {
                    container.push((x, y));
                };
            }
        }
        container
    }
}

struct Pathfinding {}

impl Pathfinding {
    pub async fn race(start: Position, end: Position) -> Result<PathfindingResult, JsValue> {
        //fetching connection data

        let data = GlobalNodeContainer::new().await;

        let connections: Vec<Vec<String>> =
            Pathfinding::find_connecting_features(&data, &start, &end)?;

        //console::log_1(&JsValue::from_serde(&connections).unwrap());

        //For when both start and end are in the same feature
        let pairs = if connections.len() == 1
            && connections[0].len() == 2
            && (connections[0][0] == connections[0][1])
        {
            let a = PointPair::new(start.clone(), end.clone());
            vec![vec![a]]
        } else {
            //Find permutations of features that can be navigated to get from a -> b
            let paired_connections: Vec<Vec<(String, String)>> = connections
                .into_iter()
                .map(|path| pairwise(path).collect())
                .collect();

            let mut pairs: Vec<Vec<PointPair>> = paired_connections
                .into_iter()
                //Finds all methods of travelling between each feature step
                .map(|vecs_of_feature_pairs| {
                    vecs_of_feature_pairs
                        .into_iter()
                        .map(|(a, b)| Pathfinding::feature_pair_as_transport(&data, &a, &b))
                        .collect()
                })
                //Finds all products of these travel methods
                .flat_map(|opts: Vec<Vec<Transport>>| {
                    opts.into_iter()
                        .multi_cartesian_product()
                        .collect::<Vec<Vec<Transport>>>()
                })
                //Some classes of travel methods are only allowed once, duplicates are removed
                .filter(|paths| {
                    all_unique(
                        paths
                            .clone()
                            .into_iter()
                            .map(|p| p.no_repeat_key)
                            .collect::<Vec<String>>(),
                    )
                })
                //Convert to pairs
                .map(|paths| {
                    paths
                        .into_iter()
                        .flat_map(|t| t.into_vec_positions())
                        .collect()
                })
                // inserts start position and end position at start and end of connections
                .map(|vec_positions| prepend_vec_append(start.clone(), vec_positions, end.clone()))
                .map(|vec_positions| PointPair::new_from_vec(vec_positions))
                .collect();

            pairs.sort_unstable_by(|path1, path2| {
                let cost1: i32 = path1.iter().map(|p| p.manhattan).sum();
                let cost2: i32 = path2.iter().map(|p| p.manhattan).sum();
                cost1.cmp(&cost2)
            });

            pairs.sort_by(|path1, path2| {
                let cost1: i32 = path1.iter().map(|p| p.minimum).sum();
                let cost2: i32 = path2.iter().map(|p| p.minimum).sum();
                cost1.cmp(&cost2)
            });

            pairs

            //console::log_1(&JsValue::from_serde(&pairs).unwrap());
        };
        // For now running the pathfinding algorithm on the first "naive" best path
        // TODO make this yield improved results and update page as those become available
        let path = pairs[0].clone();

        let astars = join_all(
            path.iter()
                .map(|pair| Astar::new(&pair.start.feature_key))
                .collect::<Vec<_>>(),
        )
        .await;
        let results: Vec<AstarResult> = astars
            .iter()
            .zip(path.into_iter())
            .map(|(astar, points)| astar.run(points))
            .collect();

        //let final_result = results.into_iter().map(|result| result.solution).collect();
        if results.is_empty() {
            Err(JsValue::from(
                "There is as of yet insufficient data to calculate this path.",
            ))
        } else {
            let p = PathfindingResult::new(start, end, results);
            Ok(p)
        }
    }

    fn find_connecting_features<'k>(
        data: &GlobalNodeContainer,
        start: &Position,
        end: &Position,
    ) -> Result<Vec<Vec<String>>, JsValue> {
        let first = start.feature_key.to_string();
        let finish = end.feature_key.to_string();

        if first == finish {
            return Ok(vec![vec![first, finish]]);
        };

        let mut connections = vec![vec![first.clone()]];

        let mut used_features: HashSet<&str> = HashSet::new();
        used_features.insert(&first);
        let mut done_connections: Vec<Vec<String>> = Vec::new();

        while !connections.is_empty() {
            let mut new_connections = Vec::new();

            for path in connections.iter() {
                let current_feature = path.last().unwrap();
                if let Some(next_features) = data.feature_table.get(current_feature) {
                    for next_feature in next_features.iter() {
                        if used_features.insert(next_feature) || *next_feature == finish {
                            let mut new_path = path.clone();
                            new_path.push(next_feature.to_string());
                            new_connections.push(new_path);
                        };
                    }
                };
            }

            let (done, running) = new_connections
                .into_iter()
                .partition(|path| *(path.last().unwrap()) == finish);

            //console::log_1(&JsValue::from_serde(&done).unwrap());
            done_connections.extend(done);
            connections = running;
        }
        if done_connections.is_empty() {
            let js_msg: JsValue = format!("Unable to find path from {} to {}. It likely involves missing data, or the path is not possible at all.", &first, &finish).into();
            Err(js_msg)
        } else {
            Ok(done_connections)
        }
    }
    pub fn feature_pair_as_transport(
        data: &GlobalNodeContainer,
        feature_a: &String,
        feature_b: &String,
    ) -> Vec<Transport> {
        let transports_from_a = data
            .feature_navigation
            .get(feature_a)
            .expect("Travel method was in table but not in navigation.");

        let transports_from_a_to_b = transports_from_a
            .into_iter()
            .filter(|t| t.destination.feature_key == *feature_b)
            .map(|item| item.clone())
            .collect();
        //console::log_1(&JsValue::from_serde(&transports_from_a_to_b).unwrap());
        transports_from_a_to_b
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PathfindingResult {
    start: Position,
    goal: Position,
    #[serde(rename = "closestPoint")]
    closest_point: Position,
    success: bool,
    #[serde(rename = "computeTime")]
    compute_time: f32,
    #[serde(rename = "totalDuration")]
    total_duration: i32,
    route: Vec<AstarResult>,
}
impl PathfindingResult {
    pub fn new(start: Position, goal: Position, route: Vec<AstarResult>) -> PathfindingResult {
        PathfindingResult {
            start,
            closest_point: goal.clone(),
            goal,
            success: true,
            compute_time: 0.1337,
            total_duration: 1337,
            route,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PointPair {
    start: Position,
    end: Position,
    manhattan: i32,
    minimum: i32,
}

impl PointPair {
    pub fn new(start: Position, end: Position) -> PointPair {
        assert!(
            start.feature_key == end.feature_key,
            "These points do not belong in the same feature."
        );
        let dx = start.x - end.x;
        let dy = start.y - end.y;
        PointPair {
            start,
            end,
            manhattan: dx.abs() + dy.abs(),
            minimum: std::cmp::max(dx.abs(), dy.abs()),
        }
    }
    pub fn new_from_vec(vec_positions: Vec<Position>) -> Vec<PointPair> {
        vec_positions
            .chunks_exact(2)
            .into_iter()
            .map(|a| PointPair::new(a[0].clone(), a[1].clone()))
            .collect()
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Transport {
    #[serde(alias = "rowNumber")]
    row_number: u32,
    #[serde(alias = "groupName")]
    group_name: String,
    #[serde(alias = "no-repeat-key")]
    no_repeat_key: String,
    destination: Position,
    origin: Position,
}

impl Transport {
    pub fn into_vec_positions(self) -> Vec<Position> {
        vec![self.origin, self.destination]
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GlobalNodeContainer {
    feature_table: HashMap<String, Vec<String>>,
    feature_navigation: HashMap<String, Vec<Transport>>,
}

impl GlobalNodeContainer {
    pub async fn new() -> GlobalNodeContainer {
        let feature_table_future = GlobalNodeContainer::get_feature_table();
        let feature_navigation_future = GlobalNodeContainer::get_feature_navigation();

        let (feature_table, feature_navigation) =
            try_join!(feature_table_future, feature_navigation_future)
                .expect("Could not load feature data");

        GlobalNodeContainer {
            feature_table,
            feature_navigation,
        }
    }

    pub async fn get_feature_table() -> Result<HashMap<String, Vec<String>>, JsValue> {
        let mut opts = RequestInit::new();
        opts.method("GET");
        opts.mode(RequestMode::Cors);

        let url = "data/rs3/feature_table.json";

        let request = Request::new_with_str_and_init(&url, &opts)?;
        request
            .headers()
            .set("Accept", "application/vnd.github.v3+json")?;

        let window = web_sys::window().unwrap();

        let resp_value = JsFuture::from(window.fetch_with_request(&request)).await?;

        assert!(resp_value.is_instance_of::<Response>());
        let resp: Response = resp_value.dyn_into().unwrap();

        if resp.ok() {
            let json = JsFuture::from(resp.json()?).await?;
            let feature_table: HashMap<String, Vec<String>> =
                json.into_serde().expect("Error parsing json data.");
            Ok(feature_table)
        } else {
            let js_msg: JsValue = format!("{} Error fetching {}.", resp.status(), &url).into();
            Err(js_msg)
        }
    }

    pub async fn get_feature_navigation() -> Result<HashMap<String, Vec<Transport>>, JsValue> {
        let mut opts = RequestInit::new();
        opts.method("GET");
        opts.mode(RequestMode::Cors);

        let url = "data/rs3/feature_navigation.json";

        let request = Request::new_with_str_and_init(&url, &opts)?;
        request
            .headers()
            .set("Accept", "application/vnd.github.v3+json")?;

        let window = web_sys::window().unwrap();

        let resp_value = JsFuture::from(window.fetch_with_request(&request)).await?;

        assert!(resp_value.is_instance_of::<Response>());
        let resp: Response = resp_value.dyn_into().unwrap();

        if resp.ok() {
            let json = JsFuture::from(resp.json()?).await?;
            let feature_navigation: HashMap<String, Vec<Transport>> =
                json.into_serde().expect("Error parsing json data.");
            Ok(feature_navigation)
        } else {
            let js_msg: JsValue = format!("{} Error fetching {}.", resp.status(), &url).into();
            Err(js_msg)
        }
    }
}

pub struct Astar {
    feature: Feature,
}

impl<'a> Astar {
    pub async fn new(feature_key: &String) -> Astar {
        let f = Feature::new(feature_key).await;
        Astar { feature: f }
    }

    //This function will panic if 'start' and 'end' do not share the same feature
    pub fn run(&self, points: PointPair) -> AstarResult {
        console::log_1(&JsValue::from(format!(
            "Running Astar instance for {}, {}, {} to {}, {}, {} in feature {}.",
            &points.start.plane,
            &points.start.x,
            &points.start.y,
            &points.end.plane,
            &points.end.x,
            &points.end.y,
            &points.start.feature_key
        )));

        assert!(
            &points.start.plane == &points.end.plane,
            "Attempted to run Astar instance on points in different planes."
        );
        let plane = points.start.plane;

        let mut open_nodes_heap = BinaryHeap::<Node>::new();
        let mut open_nodes_coordinates_set = HashSet::<(i32, i32)>::new();

        let mut closed_nodes = HashMap::<(i32, i32), Node>::new();
        let mut closed_nodes_coordinates_set = HashSet::<(i32, i32)>::new();

        let start_node = points.start.into_node_in_feature(&self.feature);
        let end_node = points.end.into_node_in_feature(&self.feature);

        //if the start tile is blocked, walk off this tile
        if !&self.feature.walkable_at(&start_node) {
            for ((dx, dy), subtiles) in Directions::walk_off() {
                if subtiles.into_iter().all(|(sub_x, sub_y)| {
                    self.feature.linear_vec[(4 * self.feature.size.y * (sub_x + 4 * start_node.x)
                        + sub_y
                        + 4 * start_node.y) as usize]
                }) {
                    let new_node = Node {
                        x: start_node.x + dx,
                        y: start_node.y + dy,
                        cost: start_node.cost + 1 + dx.abs() + dy.abs(),
                        heuristic: 0,
                        parent: Some(start_node.as_tuple()),
                        offset: start_node.offset,
                    };
                    open_nodes_coordinates_set.insert(new_node.as_tuple());
                    open_nodes_heap.push(new_node);
                }
            }
        } else {
            open_nodes_coordinates_set.insert(start_node.as_tuple());
            open_nodes_heap.push(start_node);
        };

        let mut solutions = HashSet::<(i32, i32)>::new();
        //if the end tile is blocked, accept solutions next to this tile
        if !&self.feature.walkable_at(&end_node) {
            for ((dx, dy), subtiles) in Directions::walk_off() {
                if subtiles.into_iter().all(|(sub_x, sub_y)| {
                    self.feature.linear_vec[(4 * self.feature.size.y * (sub_x + 4 * start_node.x)
                        + sub_y
                        + 4 * start_node.y) as usize]
                }) {
                    let x = end_node.x + dx;
                    let y = end_node.y + dy;
                    solutions.insert((x, y));
                }
            }
        } else {
            solutions.insert(end_node.as_tuple());
        };

        assert!(
            !open_nodes_coordinates_set.is_empty(),
            "Failed to find valid start position."
        );
        assert!(!solutions.is_empty(), "Failed to find valid end position.");

        while let Some(node) = open_nodes_heap.pop() {
            let removed_tuple_state = open_nodes_coordinates_set.remove(&node.as_tuple());
            assert!(removed_tuple_state, "Tuple was not found");

            closed_nodes_coordinates_set.insert(node.as_tuple());
            let current_key = node.as_tuple();
            closed_nodes.insert(current_key, node);
            let current_node = closed_nodes.get(&current_key).unwrap();

            if solutions.contains(&current_node.as_tuple()) {
                let mut solution: Vec<Position> = Vec::new();
                let mut key = current_node.as_tuple();
                solution.push(current_node.to_position(&plane, &self.feature));

                //would like to get a more elegant way to do this
                while let Some(node) = closed_nodes.get(&key) {
                    solution.push(node.to_position(&plane, &self.feature));

                    match node.parent {
                        Some(k) => key = k,
                        None => break,
                    };
                }

                return AstarResult {
                    start: points.start,
                    end: points.end,
                    duration: solution.len() as i32,
                    solution: solution,
                    description: "description".to_string(),
                    id: -1,
                    image: "image_link".to_string(),
                    title: "title".to_string(),
                };
            }

            for ((dx, dy), subtiles) in Directions::walk() {
                if !closed_nodes_coordinates_set
                    .contains(&(current_node.x + dx, current_node.y + dy))
                    && !open_nodes_coordinates_set
                        .contains(&(current_node.x + dx, current_node.y + dy))
                    && subtiles.into_iter().all(|(sub_x, sub_y)| {
                        self.feature.linear_vec[(4
                            * self.feature.size.y
                            * (sub_x + 4 * current_node.x)
                            + sub_y
                            + 4 * current_node.y)
                            as usize]
                    })
                {
                    let new_node = Node {
                        x: current_node.x + dx,
                        y: current_node.y + dy,
                        cost: current_node.cost + 1 + dx.abs() + dy.abs(),
                        heuristic: 0,
                        parent: Some(current_node.as_tuple()),
                        offset: current_node.offset,
                    };
                    assert!(
                        new_node.x != 0 && new_node.y != 0,
                        "Attempted to create node at map edge."
                    );
                    open_nodes_coordinates_set.insert(new_node.as_tuple());
                    open_nodes_heap.push(new_node);
                }
            }
        }

        //Data is formatted such that two points with the same feature will always have a path to each other.
        panic!(
            "Failed to find path between {}, {}, {} and {}, {}, {} in feature {}.",
            &points.start.plane,
            &points.start.x,
            &points.start.y,
            &points.end.plane,
            &points.end.x,
            &points.end.y,
            &points.start.feature_key
        );
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AstarResult {
    start: Position,
    end: Position,
    #[serde(rename = "coords")]
    solution: Vec<Position>,
    id: i32,
    title: String,
    description: String,
    image: String,
    duration: i32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Position {
    #[serde(rename(serialize = "z"))]
    plane: i32,
    x: i32,
    y: i32,
    #[serde(alias = "f")]
    feature_key: String,
}
impl Position {
    pub fn new(&plane: &i32, &x: &i32, &y: &i32, feature_key: String) -> Position {
        Position {
            plane,
            x,
            y,
            feature_key,
        }
    }
    pub fn into_node_in_feature<'b>(&self, feature: &'b Feature) -> Node<'b> {
        assert!(self.x > feature.offset.x);
        assert!(self.y > feature.offset.y);

        let x = self.x - feature.offset.x;
        let y = self.y - feature.offset.y;

        Node {
            x,
            y,
            cost: 0,
            heuristic: 0,
            parent: None,
            offset: &feature.offset,
        }
    }
}

impl Hash for Position {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.plane.hash(state);
        self.x.hash(state);
        self.y.hash(state);
    }
}

impl PartialEq for Position {
    fn eq(&self, other: &Self) -> bool {
        self.plane == other.plane && self.x == other.x && self.y == other.y
    }
}
impl Eq for Position {}

#[derive(Clone, Copy)]
pub struct Node<'n> {
    x: i32,
    y: i32,
    cost: i32,
    heuristic: i32,
    parent: Option<(i32, i32)>,
    offset: &'n FeatureOffset,
}

impl Eq for Node<'_> {}

impl<'n> PartialEq for Node<'n> {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y
    }
}

impl<'n> PartialOrd for Node<'n> {
    fn partial_cmp(&self, other: &Node<'n>) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'n> Ord for Node<'n> {
    fn cmp(&self, other: &Node) -> Ordering {
        other
            .cost
            .cmp(&self.cost)
            .then_with(|| self.x.cmp(&other.x))
            .then_with(|| self.y.cmp(&other.y))
    }
}

impl<'n> Hash for Node<'n> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.x.hash(state);
        self.y.hash(state);
    }
}

impl Node<'_> {
    pub fn as_tuple(&self) -> (i32, i32) {
        let x = self.x;
        let y = self.y;
        (x, y)
    }

    pub fn as_latlng(&self, feature: &Feature) -> (i32, i32) {
        let lat = self.y + feature.offset.y;
        let lng = self.x + feature.offset.x;
        (lat, lng)
    }

    pub fn priority(&self) -> f32 {
        self.cost as f32 + ALPHA * (self.heuristic as f32)
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

    fn diff(&self, other: &Node) -> (i32, i32) {
        (other.x - self.x, other.y - self.y)
    }
}

struct Directions {
    count: i32,
    walkoff: bool,
}

impl Directions {
    fn walk() -> Directions {
        Directions {
            count: -1,
            walkoff: false,
        }
    }
    fn walk_off() -> Directions {
        Directions {
            count: -1,
            walkoff: true,
        }
    }

    fn get_sub_tiles(step: (i32, i32)) -> Vec<(i32, i32)> {
        match step {
            (0, 1) => vec![(1, 4)],
            (1, 0) => vec![(4, 1)],
            (0, -1) => vec![(1, -1)],
            (-1, 0) => vec![(-1, 1)],
            _ => panic!("Invalid step!"),
        }
    }
}

const N: i32 = 0;
const E: i32 = 1;
const S: i32 = 2;
const W: i32 = 3;
const NW: i32 = 4;
const SW: i32 = 5;
const NE: i32 = 6;
const SE: i32 = 7;

impl Iterator for Directions {
    type Item = ((i32, i32), Vec<(i32, i32)>);

    fn next(&mut self) -> Option<Self::Item> {
        self.count += 1;

        if !self.walkoff {
            match self.count {
                N => Some(((0, 1), vec![(1, 3), (1, 4)])),
                E => Some(((1, 0), vec![(3, 1), (4, 1)])),
                S => Some(((0, -1), vec![(1, 0), (1, -1)])),
                W => Some(((-1, 0), vec![(-1, 1), (0, 1)])),
                NW => Some(((-1, 1), vec![(-1, 4), (0, 4), (-1, 3), (0, 3)])),
                SW => Some(((-1, -1), vec![(-1, -1), (-1, 0), (0, -1), (0, 0)])),
                NE => Some(((1, 1), vec![(4, 4), (3, 4), (4, 3), (3, 3)])),
                SE => Some(((1, -1), vec![(4, -1), (4, 0), (3, -1), (3, 0)])),
                _ => None,
            }
        } else {
            match self.count {
                N => Some(((0, 1), vec![(1, 4)])),
                E => Some(((1, 0), vec![(4, 1)])),
                S => Some(((0, -1), vec![(1, -1)])),
                W => Some(((-1, 0), vec![(-1, 1)])),
                _ => None,
            }
        }
    }
}
