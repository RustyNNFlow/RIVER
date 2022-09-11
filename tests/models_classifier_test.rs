use river::{
    Device,
    nn,
    Tensor,
    kind,
    nn::ModuleT,
};
#[test]
fn test_models_classifier_image(){
    const W: i64 = 32;
    const H: i64 = 32;
    const C: i64 = 3;
    const B: i64 = 1;
    use river::models::classifiers::image;
    let s = String::from("{\"backbone\":{\"depth\":34,\"counts\":[2,2,2,2],\"in_channel\":3,\"stem_channel\":64,\"base_channel\":64,\"out_indices\":[3],\"num_stages\":4},\"neck\":{},\"head\":{\"in_channels\":512,\"num_classes\":2}}");
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let cfg:image::ImageClassifierCfg = image::ImageClassifierCfg::loads(&s);
    let net = image::ImageClassifier::new(&vs.root(), &cfg);
    let t = Tensor::zeros(&[B,C,H,W], kind::FLOAT_CPU).to_device(vs.device());
    let _o = net.forward_t(&t, true);
}