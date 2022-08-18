use crate::{
    nn,
    nn::ModuleT,
    nn::OptimizerConfig,
    Device,
    Tensor,
    data::Iter2,
    models::classifiers::image,
    tensor::index::IndexOp,
    addons::classification::instance,
    addons::classification::instance_dataset,
    Kind,
};

use serde::{Serialize, Deserialize};
use std::{
    collections::HashMap,
    path::PathBuf,
};


#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag="type")]
pub struct CategoryInfo{
    id2cat: HashMap<i64, String>,
    cat2id: HashMap<String, i64>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag="type")]
pub struct ModuleInferCfg{
    model_config:image::ImageClassifierCfg,
    checkpoint_path:String,
    category_info:CategoryInfo,
    batch_size:i64,
}

impl ModuleInferCfg {
    pub fn loads(json_str: &String) -> ModuleInferCfg {
        serde_json::from_str(json_str).unwrap()
    }
    pub fn dumps(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }
}

#[derive(Debug)]
pub struct ModuleInfer{
    model:image::ImageClassifier,
    device:Device,
}

impl ModuleInfer {
    pub fn new(
        p: &nn::Path,
        cfg: &ModuleInferCfg,
    )->ModuleInfer{
        let mut vs = nn::VarStore::new(Device::cuda_if_available());
        let filename = PathBuf::from(cfg.checkpoint_path.clone());
        // build network
        let net = ModuleInfer{
            model: image::ImageClassifier::new(&vs.root(), &cfg.model_config),
            device: vs.device(),
        };
        // load weights
        vs.load(&filename);
        net
    }
    pub fn pipeline(
        &self,
        infer_images: &Tensor,
        test_labels: &Tensor,
        image_list:&Vec<String>,
        batch_size: i64,
    )
    {
        let mut vec_ins_group:Vec<instance::ClsInstancesGroup>=Vec::new();
        for (idx, (xs, ys)) in Iter2::new(infer_images, test_labels, batch_size).return_smaller_last_batch().enumerate() {
            let (scores, ids) = self.model.simple_test(&xs.to_device(self.device));
            let size = ids.size()[0];
            for index in 0..size{
                let id = ids.int64_value(&[index]);
                let score = scores.double_value(&[index, id]);
                let category = String::from(format!("{}",id));
                let ins=instance::ClsInstancesGroup::new(
                    &category,
                    &image_list[index as usize],
                    1024,
                    1024,
                    score,
                );
                vec_ins_group.push(ins);
            }
        }
        let ins_dataset:instance_dataset::ClsInstancesDataset=instance_dataset::ClsInstancesDataset::new(
            vec_ins_group,
            String::from("dataset")
        );
        println!("{:?}", ins_dataset);
        ins_dataset.dump_to_file(&String::from("tmp.json"));
    }
}