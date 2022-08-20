//! A simple dataset structure shared by various computer vision datasets.
use crate::data::Iter2;
use crate::{IndexOp, Tensor};
use rand::Rng;

use std::collections::{
    HashMap,
    HashSet,
};
use crate::{
    addons::classification::instance::ClsInstancesGroup,
    addons::classification::instance_dataset::ClsInstancesDataset,
    vision::image as image_op,
    Kind,
};

use std::{
    fs::File,
    io::Read,
    path::Path,
};
#[derive(Debug)]
pub struct DatasetResult{
    pub img:Tensor,
    pub instances_group:ClsInstancesGroup,
}
#[derive(Debug)]
pub struct ClsDataset {
    pub map: HashMap<usize, ClsInstancesGroup>,
    pub image_root:String,
}

impl ClsDataset {
    pub fn new(
        list_path:&String,
        image_root:&String,
        train:bool,
        anno_path:Option<String>,
    ) -> ClsDataset {

        let mut file = File::open(&list_path).unwrap();
        let mut buff = String::new();
        file.read_to_string(&mut buff).unwrap();
        let list:Vec<String> = serde_json::from_str(&buff).unwrap();

        let mut map: HashMap<usize, ClsInstancesGroup> = HashMap::new();
        let mut name_map: HashMap<String, ClsInstancesGroup> = HashMap::new();
        if train {
            let anno_path = match anno_path{
                Some(s)=>s,
                None=>String::from(""),
            };
            let dataset: ClsInstancesDataset = ClsInstancesDataset::load_by_file(&anno_path);

            for data_single in dataset.data.iter() {
                name_map.insert(data_single.group_name(), data_single.clone());
            }
        }
        else {
            for name in list.iter(){
                name_map.insert(
                    name.clone(),
                    ClsInstancesGroup::new(
                        &String::from(""),
                        &name,
                        0,
                        0,
                        0.0,
                    ),
                );
            }
        }

        for (i, name) in list.iter().enumerate() {
            if let Some(ins) = name_map.get(name) {
                map.insert(i, ins.clone());
            }
        }

        ClsDataset{
            map:map,
            image_root:image_root.clone(),
        }
    }
    pub fn get_info(&self, idx: usize)->Option<&ClsInstancesGroup>{
        self.map.get(&idx)
    }
    pub fn prepare(&self, idx:usize)->Option<DatasetResult>{
        if let Some(ins) = self.get_info(idx){
            let name = ins.group_name();
            let path = Path::new(&self.image_root).join(&name);
            let mut img = image_op::load(path).unwrap();
            //pipeline process
            img = img.mean_dim(&[0], true, Kind::Float).view((1,1,28,28))/255.;
            Some(DatasetResult{img:img, instances_group:ins.clone()})
        }
        else{
            None
        }

    }
}

