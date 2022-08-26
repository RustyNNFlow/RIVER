use std::io::Write;
use river::{
    data,
    IndexOp,
    Tensor,
    datasets::dataset_result::DatasetResult,
    datasets::pipelines::transforms,
    datasets::pipelines::compose::TransformOps,
    datasets::pipelines::compose::Compose,
};
use serde::{Serialize, Deserialize};
#[test]
fn test_transforms_add_batch_dim(){
    let img_root = String::from("tests/assets/classification/dataset");
    let img_name = String::from("training/9/21842.jpg");
    let d_r = DatasetResult::new(
        &img_root,
        &img_name,
    );
    let op = transforms::AddBatchDim::new();
    let res = op.call(d_r);
}

#[test]
fn test_transforms_to_float(){
    let img_root = String::from("tests/assets/classification/dataset");
    let img_name = String::from("training/9/21842.jpg");
    let d_r = DatasetResult::new(
        &img_root,
        &img_name,
    );
    let op = transforms::ToFloat::new();
    let _res = op.call(d_r);
}

#[test]
fn test_transforms_resize_torch(){
    let img_root = String::from("tests/assets/classification/dataset");
    let img_name = String::from("training/9/21842.jpg");
    let d_r = DatasetResult::new(
        &img_root,
        &img_name,
    );
    let op = transforms::ResizeTorch::new(
        14,
        14,
    );
    let _res = op.call(d_r);
}

#[test]
fn test_compose_new(){
    let img_root = String::from("tests/assets/classification/dataset");
    let img_name = String::from("training/9/21842.jpg");
    let d_r = DatasetResult::new(
        &img_root,
        &img_name,
    );
    let mut ts:Vec<Box<dyn TransformOps>>=Vec::new();
    ts.push(Box::new(transforms::ResizeTorch::new(14, 14,)));
    let compose = Compose::new(ts);
    let _res = compose.call(d_r);
}

#[test]
fn test_compose_loads_dumps(){
    let s = String::from("{\"transforms\":[{\"type\":\"ResizeTorch\",\"target_height\":14,\"target_width\":14}]}");
    let compose = Compose::loads(&s);
    let o = compose.dumps();
}

#[test]
fn test_compose_load_dump(){
    let s = String::from("tests/assets/classification/transforms0.json");
    let compose = Compose::load_by_file(&s);
    let filename =
        std::env::temp_dir().join(format!("transforms0.json"));
    let o = compose.dump_to_file(&filename.into_os_string().into_string().unwrap());
}
