use serde::{Serialize, Deserialize};
use super::instance::ClsInstancesGroup;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ClsDatasetUsage{
    dataset_name:String,
}
#[derive(Serialize, Deserialize, Debug)]
pub struct ClsInstancesDataset{
    data: Vec<ClsInstancesGroup>,
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
}