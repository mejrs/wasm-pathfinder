mod feature;
mod node;
mod pathfinding;
mod utils;

use crate::pathfinding::Position;
use wasm_bindgen::prelude::*;

/// WASM interface to Pathfinding::race
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

    let result = pathfinding::race(position_1, position_2).await?;
    let js_result = JsValue::from_serde(&result).unwrap();

    Ok(js_result)
}
