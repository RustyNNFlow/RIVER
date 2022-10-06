use serde::{Serialize, Deserialize};
use rand::seq::SliceRandom;
use crate::{
    Device,
    addons::detection::instance::DetInstancesGroup,
    addons::detection::instance::DetInstanceData,
    addons::detection::instance_dataset::DetInstancesDataset,
    vision::image as image_op,
    datasets::category_info,
    datasets::det_dataset_iter,
    datasets::det_dataset_result,
    datasets::det_pipelines::compose::Compose,
};

use std::{
    fs::File,
    io::Read,
    path::PathBuf,
    fs,
    collections::HashMap,
};



#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetDataset {
    pub map: HashMap<usize, DetInstancesGroup>,
    pub image_root:String,
    pub pipeline:Compose,
    pub category_info: Option<category_info::CategoryInfo>,
}

impl DetDataset {
    pub fn new(
        list_path:&String,
        image_root:&String,
        pipeline:Compose,
        train:bool,
        shuffle:bool,
        anno_path:Option<String>,
        category_info:Option<String>,
    ) -> DetDataset {

        let mut file = File::open(&list_path).unwrap();
        let mut buff = String::new();
        file.read_to_string(&mut buff).unwrap();
        let mut list:Vec<String> = serde_json::from_str(&buff).unwrap();
        if shuffle{
            let mut rng = rand::thread_rng();
            list.shuffle(&mut rng);
        }


        let mut map: HashMap<usize, DetInstancesGroup> = HashMap::new();
        let mut name_map: HashMap<String, DetInstancesGroup> = HashMap::new();

        let category_info:Option<category_info::CategoryInfo> = match category_info {
            Some(s)=>Some(category_info::CategoryInfo::load_by_file(&s)),
            None=>None,
        };
        if train {
            let anno_path = match anno_path{
                Some(s)=>s,
                None=>String::from(""),
            };

            let mut dataset: DetInstancesDataset = DetInstancesDataset::load_by_file(&anno_path);

            for data_single in dataset.data.iter_mut() {
                if let Some(category_info) = category_info.clone() {
                    data_single.update_category_index(category_info);
                }
                name_map.insert(data_single.group_name(), data_single.clone());
            }
        }
        else {
            for name in list.iter(){
                let vec_data:Vec<DetInstanceData> = Vec::new();
                name_map.insert(
                    name.clone(),
                    DetInstancesGroup::new(
                        vec_data,
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

        DetDataset{
            map:map,
            image_root:image_root.clone(),
            pipeline : pipeline,
            category_info:category_info,
        }
    }
    pub fn get_info(&self, idx: usize)->Option<&DetInstancesGroup>{
        self.map.get(&idx)
    }
    pub fn len(&self)->usize{
        self.map.len()
    }
    pub fn prepare(&self, idx:usize)->Option<det_dataset_result::DetDatasetResult>{
        if let Some(ins) = self.get_info(idx){
            let name = ins.group_name();
            let vec: Vec<&str>  = name.split("/").collect();

            let mut path_buf:PathBuf = PathBuf::new();
            path_buf.push(self.image_root.clone());
            path_buf.extend(&vec);
            let path = path_buf.as_path();

            let img = image_op::load(path).unwrap();
            let res = det_dataset_result::DetDatasetResult{img:img, instances_group:ins.clone()};
            //pipeline process
            Some(self.pipeline.call(res))
            // img = img.mean_dim(&[0], true, Kind::Float).view((1,1,28,28))/255.;
            // Some()
        }
        else{
            None
        }

    }
    pub fn iter(&self, batch_size: usize, device: Device) -> det_dataset_iter::DatasetIter {
        det_dataset_iter::DatasetIter::new(
            self.clone(),
            batch_size,
            device,
        )
    }
    pub fn loads(json_str: &String) -> DetDataset {
        serde_json::from_str(json_str).unwrap()
    }
    pub fn dumps(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }
    pub fn load_by_file(json_path: &String) -> DetDataset{
        let data = fs::read_to_string(json_path).unwrap();
        serde_json::from_str(&data).unwrap()
    }
    pub fn dump_to_file(&self, json_path: &String){
        serde_json::to_writer_pretty(&File::create(json_path).unwrap(), &self);
    }
}

