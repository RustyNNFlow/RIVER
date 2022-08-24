use serde::{Serialize, Deserialize};
use std::{
    collections::HashMap,
    path::PathBuf,
    path::Path,
};
use std::fs;
use std::fs::File;
use std::io::BufWriter;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CategoryInfo{
    pub id2cat: HashMap<String, String>,
    pub cat2id: HashMap<String, i64>,
}

impl CategoryInfo{
    pub fn loads(json_str: &String) -> CategoryInfo {
        serde_json::from_str(json_str).unwrap()
    }
    pub fn dumps(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }

    pub fn load_by_file(json_path: &String) -> CategoryInfo{
        let data = fs::read_to_string(json_path).unwrap();
        serde_json::from_str(&data).unwrap()
    }
    pub fn dump_to_file(&self, json_path: &String){
        serde_json::to_writer_pretty(&File::create(json_path).unwrap(), &self);
    }
}