extern crate serde;
extern crate anyhow;

use anyhow::Result;
use serde::{Serialize, Deserialize};
use river::{
    nn,
    Device,
};

#[test]
fn test_instance_group_classification(){
    use river::modules::infer::infer;
    let s = "{\"model_config\":{\"backbone\":{\"c1\":2,\"c2\":2,\"c3\":2,\"c4\":2,\"in_channels\":1},\"neck\":{},\"head\":{\"in_channels\":512,\"num_classes\":10}},\"checkpoint_path\":\"/var/folders/pn/0jj6w_qx60752ys80gr89vtw0000gn/T/final.ot\",\"category_info\":{\"id2cat\":{},\"cat2id\":{}},\"batch_size\":32}";
    let cfg:infer::ModuleInferCfg=infer::ModuleInferCfg::loads(&String::from(s));
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let infer_obj = infer::ModuleInfer::new(&vs.root(), &cfg);
    let m = river::datasets::mnist::load_dir("/Users/zhoujinghui/CLionProjects/RIVER/data").unwrap();
    let mut image_list:Vec<String> = Vec::new();
    for i in 1..=1024{
        image_list.push(format!("img_{}.jpg", i));
    }
    infer_obj.pipeline(&m.test_images, &m.test_labels, &image_list, 1024);
}
