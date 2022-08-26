use crate::{
    nn,
    Device,
    addons::classification::instance,
    addons::classification::instance_dataset,
    models::classifiers::image,
    datasets::cls_dataset,
    datasets::category_info,
    datasets::dataset_iter,
    datasets::pipelines::compose::Compose,
};

use serde::{Serialize, Deserialize};
use std::{
    path::PathBuf,
};




#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag="type")]
pub struct ModuleInferCfg{
    model_config:image::ImageClassifierCfg,
    checkpoint_path:String,
    category_info:category_info::CategoryInfo,
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
    category_info:category_info::CategoryInfo,
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
            category_info:cfg.category_info.clone(),
        };
        // load weights
        vs.load(&filename);
        net
    }
    pub fn pipeline(
        &self,
        image_root:&String,
        image_list:&String,
        pipeline:Compose,
        save_dir:&String,
        batch_size: usize,
    )
    {
        let dataset:cls_dataset::ClsDataset = cls_dataset::ClsDataset::new(
            image_list,
            &image_root,
            pipeline,
            false,
            false,
            None,
            None,
        );

        let iter = dataset_iter::DatasetIter::new(
            dataset,
            batch_size,
        );

        let mut vec_ins_group:Vec<instance::ClsInstancesGroup>=Vec::new();


        for (idx, (bimages, blabels, ins_groups)) in iter.enumerate(){
            let (scores, ids) = self.model.simple_test(&bimages.to_device(self.device));
            let size = ids.size()[0];
            for index in 0..size {
                let id = ids.int64_value(&[index]);
                let id_str = String::from(format!("{}", id));
                let score = scores.double_value(&[index, id]);
                let category = self.category_info.id2cat.get(&id_str).unwrap();
                let ins_group = &ins_groups[index as usize];
                let ins = instance::ClsInstancesGroup::new(
                    &category,
                    &ins_group.group_name(),
                    ins_group.image_height(),
                    ins_group.image_width(),
                    score,
                );
                vec_ins_group.push(ins);
            }
        }
        let ins_dataset = instance_dataset::ClsInstancesDataset::new(
            vec_ins_group,
            String::from("")
        );
        ins_dataset.dump_to_file(save_dir);
        println!("save in {:?}", save_dir);
    }
}