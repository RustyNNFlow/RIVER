extern crate serde;
extern crate anyhow;

use anyhow::Result;
use serde::{Serialize, Deserialize};
use serde_json::Value;

use river::{
    nn,
    Device,
    modules::infer::infer,
    datasets::pipelines::compose::Compose,
};

use std::{
    fs::File,
    io::Read,
};

#[test]
fn test_instance_group_classification(){
    let s = String::from("{\"transforms\":[{\"type\":\"ResizeTorch\",\"target_height\":28,\"target_width\":28},{\"type\":\"ToFloat\"},{\"type\":\"AddBatchDim\"}]}");
    let pipeline = Compose::loads(&s);

    let image_root: String = String::from("tests/assets/classification/dataset");
    let image_list: String = String::from("tests/assets/classification/dataset/train_image_list.json");
    let save_dir: String= std::env::temp_dir().join(format!("out.json")).display().to_string();

    let s = "{\"model_config\":{\"backbone\":{\"depth\":34,\"counts\":[2,2,2,2],\"in_channel\":3,\"stem_channel\":64,\"base_channel\":64,\"out_indices\":[3],\"num_stages\":4},\"neck\":{},\"head\":{\"in_channels\":512,\"num_classes\":10}},\"checkpoint_path\":\"final.ot\",\"category_info\":{\"id2cat\":{\"0\":\"0\",\"1\":\"1\",\"2\":\"2\",\"3\":\"3\",\"4\":\"4\",\"5\":\"5\",\"6\":\"6\",\"7\":\"7\",\"8\":\"8\",\"9\":\"9\"},\"cat2id\":{\"0\":0,\"1\":1,\"2\":2,\"3\":3,\"4\":4,\"5\":5,\"6\":6,\"7\":7,\"8\":8,\"9\":9}},\"batch_size\":32}";
    let cfg:infer::ModuleInferCfg=infer::ModuleInferCfg::loads(&String::from(s));
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let infer_obj = infer::ModuleInfer::new(&vs.root(), &cfg);
    infer_obj.pipeline(
        &image_root,
        &image_list,
        pipeline,
        &save_dir,
        1
    );

}
