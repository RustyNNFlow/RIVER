// CNN model. This should rearch 99.1% accuracy.

use anyhow::Result;
use river::{nn, nn::ModuleT, nn::OptimizerConfig, Device, Tensor};
use river::models::classifiers::image;

pub fn run() -> Result<()> {
    let m = river::datasets::mnist::load_dir("data")?;
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let s = String::from("{\"backbone\":{\"c1\":2,\"c2\":2,\"c3\":2,\"c4\":2,\"in_channels\":1},\"neck\":{},\"head\":{\"in_channels\":512,\"num_classes\":10}}");
    let cfg:image::ImageClassifierCfg = image::ImageClassifierCfg::loads(&s);
    let net = image::ImageClassifier::new(&vs.root(), &cfg);
    let mut opt = nn::Adam::default().build(&vs, 1e-4)?;
    for epoch in 1..100 {
        let iter_len = m.train_iter(256).total_size();
        for (idx, (bimages, blabels)) in m.train_iter(256).shuffle().to_device(vs.device()).enumerate() {
            let loss = net.forward_t(&bimages, true).cross_entropy_for_logits(&blabels);
            println!("epoch:{} iter:{}/{} loss: {:?}",epoch, idx, iter_len, loss);
            opt.backward_step(&loss);
            // if idx >=100{
            //     break
            // }
        }
        let test_accuracy =
            net.batch_accuracy_for_logits(&m.test_images, &m.test_labels, vs.device(), 1024);
        println!("epoch: {:4} test acc: {:5.2}%", epoch, 100. * test_accuracy,);
    }
    Ok(())
}

fn main() -> Result<()> {
    run()
}