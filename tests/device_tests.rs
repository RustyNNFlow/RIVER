use river::{Device, Tensor};

#[test]
fn tensor_device() {
    let t = Tensor::of_slice(&[3, 1, 4]);
    assert_eq!(t.device(), Device::Cpu)
}
