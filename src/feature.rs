use serde::{Deserialize, Serialize};
use wasm_bindgen::{prelude::*, JsCast};
use wasm_bindgen_futures::JsFuture;
use web_sys::{Request, RequestInit, RequestMode, Response};

use crate::node::Node;

use serde::Deserializer;

use ndarray::{Array, Array2};

fn decompress<'de, D>(deserializer: D) -> Result<Array2<bool>, D::Error>
where
    D: Deserializer<'de>,
{
    let compressed_vec: Vec<u32> = Deserialize::deserialize(deserializer)?;
    let mut v = compressed_vec.into_iter();

    let x = v.next().unwrap() * 4;
    let y = v.next().unwrap() * 4;

    let v = v
        .enumerate()
        .flat_map(|(i, len)| vec![i & 1 == 0; len as usize]);

    let a = Array::from_iter(v)
        .into_shape((x as usize, y as usize))
        .unwrap();

    Ok(a)
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



#[derive(Debug, Deserialize)]
pub struct Feature {
    #[serde(alias = "f")]
    pub feature_name: u32,

    #[serde(alias = "s")]
    pub size: FeatureSize,

    #[serde(alias = "o")]
    pub offset: FeatureOffset,

    #[serde(alias = "d", deserialize_with = "decompress")]
    pub linear_vec: Array2<bool>,
}

impl Feature {
    pub async fn new(feature_key: &str) -> Result<Feature, JsValue> {
        let mut opts = RequestInit::new();
        opts.method("GET");
        opts.mode(RequestMode::Cors);

        let url = format!(
            "wasm-pathfinder-data/features/feature_{}.json",
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

            let feature_info: Feature = json.into_serde().unwrap();

            //console::log_2(&msg, &requested_url);
            Ok(feature_info)
        } else {
            let js_msg: JsValue = format!("{} Error fetching {}.", resp.status(), &url).into();
            Err(js_msg)
        }
    }

    pub fn walkable_at(&self, node: &Node) -> bool {
        self.linear_vec[((1 + 4 * node.x) as usize, (1 + 4 * node.y) as usize)]
    }
}
