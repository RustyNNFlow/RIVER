use river::{
    Device,
    nn,
    Tensor,
    kind,
    nn::ModuleT,
};
#[test]
fn test_models_necks_gap(){
    const W: i64 = 32;
    const H: i64 = 32;
    const C: i64 = 3;
    const B: i64 = 1;
    use river::models::necks::gap;
    let cfg = gap::GlobalAveragePoolingCfg{};
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let net = gap::GlobalAveragePooling::new(&vs.root(), &cfg);
    let t = Tensor::zeros(&[B,C,H,W], kind::FLOAT_CPU).to_device(vs.device());
    let o_ = net.forward_t(&t, true);
}