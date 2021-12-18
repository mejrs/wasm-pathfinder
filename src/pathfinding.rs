use fnv::FnvHashMap as HashMap;
use fnv::FnvHashSet as HashSet;
use futures::{future::join_all, try_join};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::{
    collections::BinaryHeap,
    hash::{Hash, Hasher},
};
use wasm_bindgen::{prelude::*, JsCast};
use wasm_bindgen_futures::JsFuture;
use web_sys::{console, Request, RequestInit, RequestMode, Response};

use crate::feature::Feature;
use crate::node::Node;
use crate::utils::*;

pub async fn race(start: Position, end: Position) -> Result<PathfindingResult, JsValue> {
    //fetching connection data

    let data = GlobalNodeContainer::new().await;

    let connections: Vec<Vec<String>> = find_connecting_features(&data, &start, &end)?;

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
        let paired_connections = connections.into_iter().map(|path| pairwise(path).collect());

        let mut pairs: Vec<Vec<PointPair>> = paired_connections
            .into_iter()
            //Finds all methods of travelling between each feature step
            .map(|vecs_of_feature_pairs: Vec<(String, String)>| {
                vecs_of_feature_pairs
                    .into_iter()
                    .map(|(a, b)| feature_pair_as_transport(&data, &a, &b))
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
            .map(PointPair::new_from_vec)
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

fn find_connecting_features(
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

    let mut used_features: HashSet<&str> = HashSet::default();
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
        let js_msg: JsValue = format!(
            "Unable to find path from {} to {}. It likely involves missing data, or the path is not possible at all.",
            &first, &finish
        )
        .into();
        Err(js_msg)
    } else {
        Ok(done_connections)
    }
}
pub fn feature_pair_as_transport(
    data: &GlobalNodeContainer,
    feature_a: &str,
    feature_b: &str,
) -> Vec<Transport> {
    let transports_from_a = data
        .feature_navigation
        .get(feature_a)
        .expect("Travel method was in table but not in navigation.");

    let transports_from_a_to_b = transports_from_a
        .iter()
        .filter(|t| t.destination.feature_key == *feature_b)
        .cloned()
        .collect();
    //console::log_1(&JsValue::from_serde(&transports_from_a_to_b).unwrap());
    transports_from_a_to_b
}

#[derive(Debug, Serialize, Clone)]
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

#[derive(Debug)]
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

        let url = "wasm-pathfinder-data/feature_table.json";

        let request = Request::new_with_str_and_init(url, &opts)?;
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

        let url = "wasm-pathfinder-data/feature_navigation.json";

        let request = Request::new_with_str_and_init(url, &opts)?;
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
    pub async fn new(feature_key: &str) -> Astar {
        let f = Feature::new(feature_key).await.unwrap();
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
            points.start.plane == points.end.plane,
            "Attempted to run Astar instance on points in different planes."
        );
        let plane = points.start.plane;

        let mut open_nodes_heap = BinaryHeap::<Node>::default();
        let mut open_nodes_coordinates_set = HashSet::<(i32, i32)>::default();

        let mut closed_nodes = HashMap::<(i32, i32), Node>::default();
        let mut closed_nodes_coordinates_set = HashSet::<(i32, i32)>::default();

        let start_node = points.start.clone().into_node_in_feature(&self.feature);
        let end_node = points.end.clone().into_node_in_feature(&self.feature);

        //if the start tile is blocked, walk off this tile
        if !&self.feature.walkable_at(&start_node) {
            for ((dx, dy), subtiles) in Directions::walk_off() {
                if subtiles.into_iter().all(|(sub_x, sub_y)| {
                    /*
                    self.feature.linear_vec[(4 * self.feature.size.y * (sub_x + 4 * start_node.x)
                        + sub_y
                        + 4 * start_node.y) as usize]
                        */
                    self.feature.linear_vec[(
                        (sub_x + 4 * start_node.x) as usize,
                        (sub_y + 4 * start_node.y) as usize,
                    )]
                }) {
                    let new_node = Node {
                        x: start_node.x + dx,
                        y: start_node.y + dy,
                        cost: start_node.cost + 1 + dx.abs() + dy.abs(),
                        heuristic: 0,
                        parent: Some(start_node.as_tuple()),
                    };
                    open_nodes_coordinates_set.insert(new_node.as_tuple());
                    open_nodes_heap.push(new_node);
                }
            }
        } else {
            open_nodes_coordinates_set.insert(start_node.as_tuple());
            open_nodes_heap.push(start_node);
        };

        let mut solutions = HashSet::<(i32, i32)>::default();
        //if the end tile is blocked, accept solutions next to this tile
        if !&self.feature.walkable_at(&end_node) {
            for ((dx, dy), subtiles) in Directions::walk_off() {
                if subtiles.into_iter().all(|(sub_x, sub_y)| {
                    self.feature.linear_vec[(
                        (sub_x + 4 * start_node.x) as usize,
                        (sub_y + 4 * start_node.y) as usize,
                    )]
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
                    solution,
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
                        self.feature.linear_vec[(
                            (sub_x + 4 * current_node.x) as usize,
                            (sub_y + 4 * current_node.y) as usize,
                        )]
                    })
                {
                    let new_node = Node {
                        x: current_node.x + dx,
                        y: current_node.y + dy,
                        cost: current_node.cost + 1 + dx.abs() + dy.abs(),
                        heuristic: 0,
                        parent: Some(current_node.as_tuple()),
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

#[derive(Debug, Serialize, Clone)]
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

#[derive(Debug, Serialize, Deserialize, Clone, Eq)]
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
    pub fn into_node_in_feature(self, feature: &Feature) -> Node {
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

pub struct Directions {
    count: i32,
    walkoff: bool,
}

impl Directions {
    pub fn walk() -> Directions {
        Directions {
            count: -1,
            walkoff: false,
        }
    }
    pub fn walk_off() -> Directions {
        Directions {
            count: -1,
            walkoff: true,
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
#[derive(Debug, Deserialize, Clone)]
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
    pub fn into_vec_positions(self) -> [Position; 2] {
        [self.origin, self.destination]
    }
}


pub struct Key {
    plane: i32,
    feature: i32,
}
