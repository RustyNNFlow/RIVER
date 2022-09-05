use serde::{Serialize, Deserialize};
use super::instance::DetInstancesGroup;
use std::fs;
use std::fs::File;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DetDatasetUsage{
    dataset_name:String,
}
#[derive(Serialize, Deserialize, Debug)]
pub struct DetInstancesDataset{
    pub data: Vec<DetInstancesGroup>,
    usage: DetDatasetUsage,
}

impl DetInstancesDataset {
    pub fn loads(json_str: &String) -> DetInstancesDataset {
        serde_json::from_str(json_str).unwrap()
    }
    pub fn dumps(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }
    pub fn dataset_name(&self) -> String {
        self.usage.dataset_name.clone()
    }
    pub fn load_by_file(json_path: &String) -> DetInstancesDataset{
        let data = fs::read_to_string(json_path).unwrap();
        serde_json::from_str(&data).unwrap()
    }
    pub fn dump_to_file(&self, json_path: &String){
        serde_json::to_writer_pretty(&File::create(json_path).unwrap(), &self);
    }
    pub fn new(
        vec_ins_group:Vec<DetInstancesGroup>,
        dataset_name:String,
    )->DetInstancesDataset{
        DetInstancesDataset{
            data:vec_ins_group,
            usage:DetDatasetUsage{dataset_name: dataset_name.clone()},
        }
    }
}