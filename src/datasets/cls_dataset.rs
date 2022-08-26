use serde::{Serialize, Deserialize};
use rand::seq::SliceRandom;
use crate::{
    addons::classification::instance::ClsInstancesGroup,
    addons::classification::instance_dataset::ClsInstancesDataset,
    vision::image as image_op,
    datasets::category_info,
    datasets::dataset_iter,
    datasets::dataset_result,
    datasets::pipelines::compose::TransformOps,
    datasets::pipelines::compose::Compose,
};

use std::{
    fs::File,
    io::Read,
    path::Path,
    fs,
    collections::HashMap
};



#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClsDataset {
    pub map: HashMap<usize, ClsInstancesGroup>,
    pub image_root:String,
    pub pipeline:Compose,
    pub category_info: Option<category_info::CategoryInfo>,
}

impl ClsDataset {
    pub fn new(
        list_path:&String,
        image_root:&String,
        pipeline:Compose,
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
            pipeline : pipeline,
            category_info:category_info,
        }
    }
    pub fn get_info(&self, idx: usize)->Option<&ClsInstancesGroup>{
        self.map.get(&idx)
    }
    pub fn len(&self)->usize{
        self.map.len()
    }
    pub fn prepare(&self, idx:usize)->Option<dataset_result::DatasetResult>{
        if let Some(ins) = self.get_info(idx){
            let name = ins.group_name();
            let path = Path::new(&self.image_root).join(&name);
            let img = image_op::load(path).unwrap();
            let res = dataset_result::DatasetResult{img:img, instances_group:ins.clone()};
            //pipeline process
            Some(self.pipeline.call(res))
            // img = img.mean_dim(&[0], true, Kind::Float).view((1,1,28,28))/255.;
            // Some()
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
    pub fn loads(json_str: &String) -> ClsDataset {
        serde_json::from_str(json_str).unwrap()
    }
    pub fn dumps(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }
    pub fn load_by_file(json_path: &String) -> ClsDataset{
        let data = fs::read_to_string(json_path).unwrap();
        serde_json::from_str(&data).unwrap()
    }
    pub fn dump_to_file(&self, json_path: &String){
        serde_json::to_writer_pretty(&File::create(json_path).unwrap(), &self);
    }
}

