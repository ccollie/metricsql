use std::fs::File;
use std::io::Read;

use chrono::{DateTime, Utc};
use serde::Deserialize;

/**
{
  "timestamp": "2023-07-19T23:42:08.640Z",
  "region": "madrid",
  "type": "business",
  "consumption": "42.6"
},
 **/
#[derive(Debug, Deserialize)]
pub struct PowerConsumption {
    pub timestamp: DateTime<Utc>,
    pub region: String,
    pub location_type: String,
    pub consumption: f64,
}

pub fn load_consumption_data() -> Vec<PowerConsumption> {
    let mut file = File::open("power_consumption.json").unwrap();
    let mut data = String::new();
    file.read_to_string(&mut data).unwrap();
    let data: Vec<PowerConsumption> = serde_json::from_str(&data).unwrap();
    data
}
