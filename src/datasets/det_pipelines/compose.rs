use serde::{Serialize, Deserialize};
use std::fs;
use std::fs::File;

use crate::{
    datasets::det_dataset_result,
};
use core::fmt::Debug;

pub trait TransformOpsclone {
    fn clone_box(&self) -> Box<dyn TransformOps>;
}

impl<T> TransformOpsclone for T
    where
        T: 'static + TransformOps + Clone,
{
    fn clone_box(&self) -> Box<dyn TransformOps> {
        Box::new(self.clone())
    }
}
#[typetag::serde(tag = "type")]
// pub trait TransformOps:TransformOpsclone+erased_serde::Serialize{
pub trait TransformOps:TransformOpsclone{
    fn call(
        &self,
        res:det_dataset_result::DetDatasetResult,
    )->det_dataset_result::DetDatasetResult;
    fn repr(&self)->String;
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Compose{
    transforms:Vec<Box<dyn TransformOps>>,
}

impl Debug for dyn TransformOps {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "TRANSFORM_OPS{{{}}}", self.repr())
    }
}


impl Clone for Box<dyn TransformOps> {
    fn clone(&self) -> Box<dyn TransformOps> {
        self.clone_box()
    }
}

impl Compose{
    pub fn new(transforms:Vec<Box<dyn TransformOps>>)->Compose{
        Compose{transforms:transforms}
    }
}

impl Compose {
    pub fn call(
        &self,
        res:det_dataset_result::DetDatasetResult,
    )->det_dataset_result::DetDatasetResult
    {
        let mut res_ = res;
        for transform in self.transforms.iter(){
            res_ = transform.call(res_);
        }
        res_
    }
    pub fn loads(json_str: &String) -> Compose {
        serde_json::from_str(json_str).unwrap()
    }
    pub fn dumps(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }
    pub fn load_by_file(json_path: &String) -> Compose{
        let data = fs::read_to_string(json_path).unwrap();
        serde_json::from_str(&data).unwrap()
    }
    pub fn dump_to_file(&self, json_path: &String){
        serde_json::to_writer_pretty(&File::create(json_path).unwrap(), &self);
    }
}