// CNN model. This should rearch 99.1% accuracy.

use anyhow::Result;
use std::path::PathBuf;
use river::{
    nn,
    nn::ModuleT,
    nn::OptimizerConfig,
    Device,
    Tensor,
    models::classifiers::image,
    datasets::cls_dataset,
    datasets::dataset_iter,
    datasets::pipelines::compose::Compose,
    modules::infer::infer,
};

pub fn train() -> Result<()> {
    let s = String::from("{\"transforms\":[{\"type\":\"ResizeTorch\",\"target_height\":28,\"target_width\":28},{\"type\":\"ToFloat\"},{\"type\":\"AddBatchDim\"}]}");
    let pipeline = Compose::loads(&s);

    let dataset:cls_dataset::ClsDataset = cls_dataset::ClsDataset::new(
        &String::from("/home/N3_3090U5/数据/项目数据集/mnist_jpg/train_image_list.json"),
        &String::from("/home/N3_3090U5/数据/项目数据集/mnist_jpg"),
        Compose::loads(&String::from("{\"transforms\":[{\"type\":\"ResizeTorch\",\"target_height\":28,\"target_width\":28},{\"type\":\"ToFloat\"},{\"type\":\"AddBatchDim\"}]}")),
        true,
        true,
        Some(String::from("/home/N3_3090U5/数据/项目数据集/mnist_jpg/train_anno.json")),
        Some(String::from("/home/N3_3090U5/数据/项目数据集/mnist_jpg/category_info.json")),
    );
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let s = String::from("{\"backbone\":{\"depth\":34,\"counts\":[2,2,2,2],\"in_channel\":3,\"stem_channel\":64,\"base_channel\":64,\"out_indices\":[3],\"num_stages\":4},\"neck\":{},\"head\":{\"in_channels\":512,\"num_classes\":10}}");
    let cfg:image::ImageClassifierCfg = image::ImageClassifierCfg::loads(&s);
    let net = image::ImageClassifier::new(&vs.root(), &cfg);
    let mut opt = nn::Adam::default().build(&vs, 1e-4)?;

    for epoch in 1..2 {
        for (idx, (bimages, blabels, ins_groups)) in dataset.iter(512).enumerate(){
            let bimages = bimages.to_device(device);
            let blabels = blabels.to_device(device);
            let loss = net.forward_train(&bimages, &blabels);
            println!("epoch:{} iter:{} loss: {:?}",epoch, idx, loss);
            opt.backward_step(&loss);
            // if idx >=100{
            //     break
            // }
        }
    }
    let filename = PathBuf::from(String::from("/home/N3_3090U5/数据/项目数据集/mnist_jpg/final.ot"));
    vs.save(&filename).unwrap();
    println!("save checkpoint: {:?}",filename);
    Ok(())
}

pub fn test() -> Result<()> {
    let s = String::from("{\"transforms\":[{\"type\":\"ResizeTorch\",\"target_height\":28,\"target_width\":28},{\"type\":\"ToFloat\"},{\"type\":\"AddBatchDim\"}]}");
    let pipeline = Compose::loads(&s);

    let image_list = String::from("/home/N3_3090U5/数据/项目数据集/mnist_jpg/test_image_list.json");
    let image_root = String::from("/home/N3_3090U5/数据/项目数据集/mnist_jpg");
    let save_dir =
        std::env::temp_dir()
            .join(format!("tmp.json"))
            .into_os_string().into_string().unwrap();

    let mut vs = nn::VarStore::new(Device::cuda_if_available());
    // let s = String::from("{\"model_config\":{\"backbone\":{\"c1\":2,\"c2\":2,\"c3\":2,\"c4\":2,\"in_channels\":3},\"neck\":{},\"head\":{\"in_channels\":512,\"num_classes\":10}},\"checkpoint_path\":\"E:\\\\mnist_jpg\\\\final.ot\",\"category_info\":{\"id2cat\":{\"0\":\"0\",\"1\":\"1\",\"2\":\"2\",\"3\":\"3\",\"4\":\"4\",\"5\":\"5\",\"6\":\"6\",\"7\":\"7\",\"8\":\"8\",\"9\":\"9\"},\"cat2id\":{\"0\":0,\"1\":1,\"2\":2,\"3\":3,\"4\":4,\"5\":5,\"6\":6,\"7\":7,\"8\":8,\"9\":9}},\"batch_size\":32}");
    let s = String::from("{\"model_config\":{\"backbone\":{\"depth\":34,\"counts\":[2,2,2,2],\"in_channel\":3,\"stem_channel\":64,\"base_channel\":64,\"out_indices\":[3],\"num_stages\":4},\"neck\":{},\"head\":{\"in_channels\":512,\"num_classes\":10}},\"checkpoint_path\":\"/home/N3_3090U5/数据/项目数据集/mnist_jpg/final.ot\",\"category_info\":{\"id2cat\":{\"0\":\"0\",\"1\":\"1\",\"2\":\"2\",\"3\":\"3\",\"4\":\"4\",\"5\":\"5\",\"6\":\"6\",\"7\":\"7\",\"8\":\"8\",\"9\":\"9\"},\"cat2id\":{\"0\":0,\"1\":1,\"2\":2,\"3\":3,\"4\":4,\"5\":5,\"6\":6,\"7\":7,\"8\":8,\"9\":9}},\"batch_size\":32}");

    let cfg:infer::ModuleInferCfg=infer::ModuleInferCfg::loads(&String::from(s));

    let infer_obj = infer::ModuleInfer::new(&vs.root(), &cfg);
    infer_obj.pipeline(
        &image_root,
        &image_list,
        pipeline,
        &save_dir,
        1024
    );
    Ok(())
}

fn main() -> Result<()> {
    train();
    test()
}