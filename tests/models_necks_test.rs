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

#[test]
fn test_models_necks_bifpn_module(){
    const W: [i64; 3] = [128, 64, 32];
    const H: [i64; 3] = [128, 64, 32];
    const C: i64 = 512;
    const B: i64 = 1;

    use river::models::necks::bifpn;
    let s = String::from("{\"channels\":512,\"levels\":3}");
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let cfg:bifpn::BiFPNModuleCfg = bifpn::BiFPNModuleCfg::loads(&s);
    let n = bifpn::BiFPNModule::new(&vs.root(), &cfg);
    let mut ts: Vec<Tensor> = Vec::new();
    for i in 0..3{
        ts.push(Tensor::zeros(&[B,C,H[i],W[i]], kind::FLOAT_CPU).to_device(vs.device()));
    }
    let os_ = n.forward(ts, true);
}

#[test]
fn test_models_necks_bifpn(){
    const W: [i64; 3] = [128, 64, 32];
    const H: [i64; 3] = [128, 64, 32];
    const C: [i64;3] =  [128, 256, 512];
    const B: i64 = 1;

    use river::models::necks::bifpn;
    let s = String::from("{\"in_channels\":[128, 256, 512],\"out_channel\":128,\"num_outs\":3,\"start_level\":0,\"end_level\":-1,\"stack\":1,\"add_extra_convs\":false,\"extra_convs_on_inputs\":true,\"relu_before_extra_convs\":false,\"no_norm_on_lateral\":false}");
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let cfg:bifpn::BiFPNCfg = bifpn::BiFPNCfg::loads(&s);

    let mut ts: Vec<Tensor> = Vec::new();
    for i in 0..3{
        ts.push(Tensor::zeros(&[B,C[i],H[i],W[i]], kind::FLOAT_CPU).to_device(vs.device()));
    }
    let n = bifpn::BiFPN::new(&vs.root(), &cfg);
    let o_ = n.forward(ts, true);

}