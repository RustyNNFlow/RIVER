use river::{
    Device,
    nn,
    Tensor,
    kind,
    nn::ModuleT,
};
#[test]
fn test_models_multiclass_nms(){

    use river::core::post_processing::bbox_nms::multiclass_nms;
    let multi_bboxes = Tensor::of_slice(&[1, 2, 100, 101, 101, 102, 200, 201, 2, 3, 101, 102]).reshape(&[3,4]);
    let multi_scores = Tensor::of_slice(&[0.05, 0.05, 0.9, 0.1, 0.05, 0.85, 0.1, 0.2, 0.7]).reshape(&[3,3]);
    let (_bboxes, _labels) = multiclass_nms(
        &multi_bboxes,
        &multi_scores,
        0.1,
        0.1,
    );
}
