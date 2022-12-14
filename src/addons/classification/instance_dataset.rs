use serde::{Serialize, Deserialize};
use super::instance::ClsInstancesGroup;
use std::fs;
use std::fs::File;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ClsDatasetUsage{
    dataset_name:String,
}
#[derive(Serialize, Deserialize, Debug)]
pub struct ClsInstancesDataset{
    pub data: Vec<ClsInstancesGroup>,
    usage: ClsDatasetUsage,
}

impl ClsInstancesDataset {
    pub fn loads(json_str: &String) -> ClsInstancesDataset {
        serde_json::from_str(json_str).unwrap()
    }
    pub fn dumps(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }
    pub fn dataset_name(&self) -> String {
        self.usage.dataset_name.clone()
    }
    pub fn load_by_file(json_path: &String) -> ClsInstancesDataset{
        let data = fs::read_to_string(json_path).unwrap();
        serde_json::from_str(&data).unwrap()
    }
    pub fn dump_to_file(&self, json_path: &String){
        serde_json::to_writer_pretty(&File::create(json_path).unwrap(), &self);
    }
    pub fn new(
        vec_ins_group:Vec<ClsInstancesGroup>,
        dataset_name:String,
    )->ClsInstancesDataset{
        ClsInstancesDataset{
            data:vec_ins_group,
            usage:ClsDatasetUsage{dataset_name: dataset_name.clone()},
        }
    }
}