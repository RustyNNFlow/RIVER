use std::io::Write;
use river::{
    data,
    IndexOp,
    Tensor,
    datasets::dataset_result::DatasetResult,
    datasets::pipelines::transforms,
};

#[test]
fn test_transforms_result(){
    let img_root = String::from("tests/assets/classification/dataset");
    let img_name = String::from("training/9/21842.jpg");
    let mut d_r = DatasetResult::new(
        &img_root,
        &img_name,
    );
    let op = transforms::ResizeTorch::new(
        14,
        14,
    );
    let res = op.call(d_r);
    println!("{:?}", res);
}
