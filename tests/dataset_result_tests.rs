use std::io::Write;
use river::{
    data,
    IndexOp,
    Tensor,
    datasets::dataset_result::DatasetResult,
};

#[test]
fn test_dataset_result(){
    let img_root = String::from("tests/assets/classification/dataset");
    let img_name = String::from("training/9/21842.jpg");
    let d_r = DatasetResult::new(
        &img_root,
        &img_name,
    );
}
