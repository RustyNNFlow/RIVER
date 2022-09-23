use river::{
    Device,
    nn,
    Tensor,
    kind,
    nn::ModuleT,
};
#[test]
fn test_models_bbox_head_single(){
    const W: i64 = 32;
    const H: i64 = 32;
    const C: i64 = 512;
    const B: i64 = 1;

    use river::models::bbox_heads::fcos_head;
    let s = String::from("{\"in_channels\":512,\"num_classes\":81,\"feat_channels\":256,\"stacked_convs\":2,\"stride\":8}");
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let cfg:fcos_head::FCOSHeadSingleCfg = fcos_head::FCOSHeadSingleCfg::loads(&s);
    let n = fcos_head::FCOSHeadSingle::new(&vs.root(), &cfg);
    let t = Tensor::zeros(&[B,C,H,W], kind::FLOAT_CPU).to_device(vs.device());
    let _o = n.forward_t(&t, true);
}

#[test]
fn test_models_bbox_heads(){
    const W: [i64; 3] = [32, 64, 128];
    const H: [i64; 3] = [32, 64, 128];
    const C: i64 = 512;
    const B: i64 = 1;

    use river::models::bbox_heads::fcos_head;
    let s = String::from("{\"in_channels\":512,\"num_classes\":81,\"feat_channels\":256,\"stacked_convs\":2,\"strides\":[8,16,32],\"regress_ranges\":[[-1,64],[64,128],[128,100000000]]}");
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let cfg:fcos_head::FCOSHeadCfg = fcos_head::FCOSHeadCfg::loads(&s);
    let n = fcos_head::FCOSHead::new(&vs.root(), &cfg);
    let mut ts: Vec<Tensor> = Vec::new();
    for i in 0..3{
        ts.push(Tensor::zeros(&[B,C,H[i],W[i]], kind::FLOAT_CPU).to_device(vs.device()));
    }
    let _os = n.forward_t(&ts, true);
}

#[test]
fn test_models_bbox_head_get_points_single(){
    const W: i64 = 512;
    const H: i64 = 128;
    const S: i64 = 32;

    use river::models::bbox_heads::fcos_head;
    let s = String::from("{\"in_channels\":512,\"num_classes\":81,\"feat_channels\":256,\"stacked_convs\":2,\"strides\":[8,16,32],\"regress_ranges\":[[-1,64],[64,128],[128,100000000]]}");
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let cfg:fcos_head::FCOSHeadCfg = fcos_head::FCOSHeadCfg::loads(&s);
    let n = fcos_head::FCOSHead::new(&vs.root(), &cfg);

    let _o = n.get_points_single(H, W, S);
}
#[test]
fn test_models_bbox_head_get_points(){
    const W: [i64; 3] = [32, 64, 128];
    const H: [i64; 3] = [64, 128, 256];
    const S: [i64; 3] = [32, 16, 8];

    use river::models::bbox_heads::fcos_head;
    let s = String::from("{\"in_channels\":512,\"num_classes\":81,\"feat_channels\":256,\"stacked_convs\":2,\"strides\":[8,16,32],\"regress_ranges\":[[-1,64],[64,128],[128,100000000]]}");

    let vs = nn::VarStore::new(Device::cuda_if_available());
    let cfg:fcos_head::FCOSHeadCfg = fcos_head::FCOSHeadCfg::loads(&s);
    let n = fcos_head::FCOSHead::new(&vs.root(), &cfg);

    let _o = n.get_points(H.to_vec(), W.to_vec(), S.to_vec());
}

#[test]
fn test_models_bbox_fcos_target_single(){
    const W: i64 = 16;
    const H: i64 = 16;
    const S: i64 = 32;

    use river::models::bbox_heads::fcos_head;
    let s = String::from("{\"in_channels\":512,\"num_classes\":81,\"feat_channels\":256,\"stacked_convs\":2,\"strides\":[8,16,32],\"regress_ranges\":[[-1,64],[64,128],[128,100000000]]}");

    let vs = nn::VarStore::new(Device::cuda_if_available());
    let cfg:fcos_head::FCOSHeadCfg = fcos_head::FCOSHeadCfg::loads(&s);
    let n = fcos_head::FCOSHead::new(&vs.root(), &cfg);

    let points = n.get_points_single(H, W, S);

    let vs = nn::VarStore::new(Device::cuda_if_available());
    let gt_bboxes = Tensor::arange_start_step(0, 160, 20, kind::FLOAT_CPU).reshape(&[2,4]);
    let gt_labels = Tensor::ones(&[2,1], kind::INT64_CPU);

    let mut regress_range = Tensor::arange_start_step(-1, 65, 65, kind::INT64_CPU).expand_as(&points);
    let _o=n.fcos_target_single(
        points,
        gt_bboxes,
        gt_labels,
        regress_range
    );


}

#[test]
fn test_models_bbox_fcos_target(){
    const W: [i64; 3] = [32, 64, 128];
    const H: [i64; 3] = [64, 128, 256];
    const S: [i64; 3] = [32, 16, 8];
    let rr = [[-1, 64],[64, 128], [128, 256]];

    use river::models::bbox_heads::fcos_head;
    let s = String::from("{\"in_channels\":512,\"num_classes\":81,\"feat_channels\":256,\"stacked_convs\":2,\"strides\":[8,16,32],\"regress_ranges\":[[-1,64],[64,128],[128,100000000]]}");

    let vs = nn::VarStore::new(Device::cuda_if_available());
    let cfg:fcos_head::FCOSHeadCfg = fcos_head::FCOSHeadCfg::loads(&s);
    let n = fcos_head::FCOSHead::new(&vs.root(), &cfg);

    let mut vec_points:Vec<Tensor> = Vec::new();
    let mut vec_regress_range:Vec<Tensor> = Vec::new();
    for i in 0..3{
        vec_points.push(n.get_points_single(H[i], W[i], S[i]));
        vec_regress_range.push(
            Tensor::of_slice(&rr[i])
        );
    }

    let gt_bboxes = Tensor::arange_start_step(0, 160, 20, kind::FLOAT_CPU).reshape(&[2,4]);
    let gt_labels = Tensor::ones(&[2,1], kind::INT64_CPU);
    let _o = n.fcos_target(
        vec_points,
        gt_bboxes,
        gt_labels,
        vec_regress_range,
    );
}
