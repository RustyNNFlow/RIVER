use river::{
    Device,
    nn,
    Tensor,
    kind,
    nn::ModuleT,
};
#[test]
fn test_models_detectors_single_stage_forward(){
    const W: i64 = 128;
    const H: i64 = 128;
    const C: i64 = 3;
    const B: i64 = 1;
    use river::models::detectors::single_stage;
    let s = String::from("{\"backbone\":{\"depth\":34,\"counts\":[2,2,2,2],\"in_channel\":3,\"stem_channel\":64,\"base_channel\":64,\"out_indices\":[1,2,3],\"num_stages\":4},\"neck\":{\"in_channels\":[128,256,512],\"out_channel\":128,\"num_outs\":3,\"start_level\":0,\"end_level\":-1,\"stack\":1,\"add_extra_convs\":false,\"extra_convs_on_inputs\":true,\"relu_before_extra_convs\":false,\"no_norm_on_lateral\":false},\"bbox_head\":{\"in_channels\":128,\"num_classes\":81,\"feat_channels\":256,\"stacked_convs\":2,\"strides\":[8,16,32],\"regress_ranges\":[[-1,64],[64,128],[128,100000000]]}}");
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let cfg:single_stage::SingleStageDetectorCfg = single_stage::SingleStageDetectorCfg::loads(&s);
    let net = single_stage::SingleStageDetector::new(&vs.root(), &cfg);
    let t = Tensor::zeros(&[B,C,H,W], kind::FLOAT_CPU).to_device(vs.device());
    let _o = net.forward_t(&t, true);
}

#[test]
fn test_models_detectors_single_stage_forward_train(){
    const W: i64 = 128;
    const H: i64 = 128;
    const C: i64 = 3;
    const B: i64 = 2;
    const N: i64 = 2;
    use river::models::detectors::single_stage;
    let s = String::from("{\"backbone\":{\"depth\":34,\"counts\":[2,2,2,2],\"in_channel\":3,\"stem_channel\":64,\"base_channel\":64,\"out_indices\":[1,2,3],\"num_stages\":4},\"neck\":{\"in_channels\":[128,256,512],\"out_channel\":128,\"num_outs\":3,\"start_level\":0,\"end_level\":-1,\"stack\":1,\"add_extra_convs\":false,\"extra_convs_on_inputs\":true,\"relu_before_extra_convs\":false,\"no_norm_on_lateral\":false},\"bbox_head\":{\"in_channels\":128,\"num_classes\":81,\"feat_channels\":256,\"stacked_convs\":2,\"strides\":[8,16,32],\"regress_ranges\":[[-1,64],[64,128],[128,100000000]]}}");
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let cfg:single_stage::SingleStageDetectorCfg = single_stage::SingleStageDetectorCfg::loads(&s);
    let net = single_stage::SingleStageDetector::new(&vs.root(), &cfg);
    let t = Tensor::zeros(&[B,C,H,W], kind::FLOAT_CPU).to_device(vs.device());


    let mut vec_gt_bboxes:Vec<Tensor> = Vec::new();
    let mut vec_gt_labels:Vec<Tensor> = Vec::new();
    for i in 0..B {
        vec_gt_bboxes.push(Tensor::ones(&[1+i,4], kind::FLOAT_CPU).to_device(vs.device()));

        vec_gt_labels.push(Tensor::ones(&[1+i,1], kind::INT64_CPU).to_device(vs.device()));
    }

    let _o = net.forward_train(&t, &vec_gt_bboxes, &vec_gt_labels, true);
}