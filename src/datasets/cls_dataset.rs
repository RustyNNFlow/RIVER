//! A simple dataset structure shared by various computer vision datasets.
use crate::data::Iter2;
use crate::{IndexOp, Tensor};
use rand::{Rng};
use rand::seq::SliceRandom;
use std::collections::{
    HashMap,
    HashSet,
};
use crate::{
    addons::classification::instance::ClsInstancesGroup,
    addons::classification::instance_dataset::ClsInstancesDataset,
    vision::image as image_op,
    Kind,
    datasets::category_info,
    datasets::dataset_iter,
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
impl DatasetResult{
    pub fn group_name(&self)->String{
        self.instances_group.group_name()
    }
    pub fn image_height(&self)->i32{
        self.instances_group.image_height()
    }
    pub fn image_width(&self)->i32{
        self.instances_group.image_width()
    }
    pub fn x(&self)->Tensor{
        self.img.shallow_clone()
    }
    pub fn y(&self)->Tensor{
        let mut ys:Vec<i64> = Vec::new();
        for d in self.instances_group.data.iter(){
            if let Some(s) = d.category_index{
                ys.push(s);
            }
        }
        Tensor::of_slice(&ys)
    }
}
#[derive(Debug, Clone)]
pub struct ClsDataset {
    pub map: HashMap<usize, ClsInstancesGroup>,
    pub image_root:String,
    pub category_info: Option<category_info::CategoryInfo>,
}

impl ClsDataset {
    pub fn new(
        list_path:&String,
        image_root:&String,
        train:bool,
        shuffle:bool,
        anno_path:Option<String>,
        category_info:Option<String>,
    ) -> ClsDataset {

        let mut file = File::open(&list_path).unwrap();
        let mut buff = String::new();
        file.read_to_string(&mut buff).unwrap();
        let mut list:Vec<String> = serde_json::from_str(&buff).unwrap();
        if shuffle{
            let mut rng = rand::thread_rng();
            list.shuffle(&mut rng);
        }

        let mut map: HashMap<usize, ClsInstancesGroup> = HashMap::new();
        let mut name_map: HashMap<String, ClsInstancesGroup> = HashMap::new();

        let category_info:Option<category_info::CategoryInfo> = match category_info {
            Some(s)=>Some(category_info::CategoryInfo::load_by_file(&s)),
            None=>None,
        };
        if train {
            let anno_path = match anno_path{
                Some(s)=>s,
                None=>String::from(""),
            };

            let mut dataset: ClsInstancesDataset = ClsInstancesDataset::load_by_file(&anno_path);

            for data_single in dataset.data.iter_mut() {
                if let Some(category_info) = category_info.clone() {
                    data_single.update_category_index(category_info);
                }
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
            category_info:category_info,
        }
    }
    pub fn get_info(&self, idx: usize)->Option<&ClsInstancesGroup>{
        self.map.get(&idx)
    }
    pub fn len(&self)->usize{
        self.map.len()
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
    pub fn iter(&self, batch_size: usize) -> dataset_iter::DatasetIter {
        dataset_iter::DatasetIter::new(
            self.clone(),
            batch_size,
        )
    }
}

