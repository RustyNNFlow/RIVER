extern crate serde;
extern crate anyhow;

use anyhow::Result;
use serde::{Serialize, Deserialize};
use serde_json::Value;

use river::{
    nn,
    Device,
};

use std::{
    fs::File,
    io::Read,
};

#[test]
fn test_instance_group_classification(){
    use river::modules::infer::infer;
    // let mut file = File::open("/Users/zhoujinghui/Documents/数据/mnist_jpg/test_image_list.json").unwrap();
    // let mut buff = String::new();
    // file.read_to_string(&mut buff).unwrap();
    // let image_list:Vec<String> = serde_json::from_str(&buff).unwrap();

    let image_root: String = String::from("tests/assets/classification/dataset");
    let image_list: String = String::from("tests/assets/classification/dataset/train_image_list.json");
    let save_dir: String= std::env::temp_dir().join(format!("out.json")).display().to_string();

    let s = "{\"model_config\":{\"backbone\":{\"c1\":2,\"c2\":2,\"c3\":2,\"c4\":2,\"in_channels\":1},\"neck\":{},\"head\":{\"in_channels\":512,\"num_classes\":10}},\"checkpoint_path\":\"/Users/zhoujinghui/Downloads/final.ot\",\"category_info\":{\"id2cat\":{\"0\":\"0\",\"1\":\"1\",\"2\":\"2\",\"3\":\"3\",\"4\":\"4\",\"5\":\"5\",\"6\":\"6\",\"7\":\"7\",\"8\":\"8\",\"9\":\"9\"},\"cat2id\":{\"0\":0,\"1\":1,\"2\":2,\"3\":3,\"4\":4,\"5\":5,\"6\":6,\"7\":7,\"8\":8,\"9\":9}},\"batch_size\":32}";
    let cfg:infer::ModuleInferCfg=infer::ModuleInferCfg::loads(&String::from(s));
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let infer_obj = infer::ModuleInfer::new(&vs.root(), &cfg);
    infer_obj.pipeline(
        &image_root,
        &image_list,
        &save_dir,
        1
    );

}
