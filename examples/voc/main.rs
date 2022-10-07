// CNN model. This should rearch 99.1% accuracy.

use anyhow::Result;
use std::path::PathBuf;
use river::{
    nn,
    nn::ModuleT,
    nn::OptimizerConfig,
    Device,
    Tensor,
    models::detectors::single_stage,
    datasets::det_dataset,
    datasets::det_dataset_iter,
    datasets::det_pipelines::compose::Compose,

};

pub fn train() -> Result<()> {
    let dataset:det_dataset::DetDataset = det_dataset::DetDataset::new(
        &String::from("/home/N3_3090U5/数据/项目数据集/VOC/train_image_list.json"),
        &String::from("/home/N3_3090U5/数据/项目数据集/VOC"),
        Compose::loads(&String::from("{\"transforms\":[{\"type\":\"ResizeTorch\",\"target_height\":512,\"target_width\":512},{\"type\":\"ToFloat\"},{\"type\":\"AddBatchDim\"}]}")),
        true,
        true,
        Some(String::from("/home/N3_3090U5/数据/项目数据集/VOC/train_anno.json")),
        Some(String::from("/home/N3_3090U5/数据/项目数据集/VOC/category_info.json")),
    );
    let device = Device::cuda_if_available();

    let vs = nn::VarStore::new(device);

    let s = String::from("{\"backbone\":{\"depth\":34,\"counts\":[2,2,2,2],\"in_channel\":3,\"stem_channel\":64,\"base_channel\":64,\"out_indices\":[1,2,3],\"num_stages\":4},\"neck\":{\"in_channels\":[128,256,512],\"out_channel\":128,\"num_outs\":3,\"start_level\":0,\"end_level\":-1,\"stack\":1,\"add_extra_convs\":false,\"extra_convs_on_inputs\":true,\"relu_before_extra_convs\":false,\"no_norm_on_lateral\":false},\"bbox_head\":{\"in_channels\":128,\"num_classes\":81,\"feat_channels\":256,\"stacked_convs\":2,\"strides\":[8,16,32],\"regress_ranges\":[[-1,64],[64,128],[128,100000000]]}}");
    let cfg:single_stage::SingleStageDetectorCfg = single_stage::SingleStageDetectorCfg::loads(&s);
    let net = single_stage::SingleStageDetector::new(&vs.root(), &cfg);
    let mut opt = nn::Adam::default().build(&vs, 2e-3)?;

    for epoch in 1..200 {
        for (idx, (bimages, vec_gt_bboxes, vec_gt_labels, ins_groups)) in dataset.iter(64, device).enumerate(){
            let loss = net.forward_train(&bimages, &vec_gt_labels, &vec_gt_bboxes, true);
            println!("epoch:{} iter:{} loss: {:?}",epoch, idx, loss);
            opt.backward_step(&loss);
            // if idx >=100{
            //     break
            // }
        }
    }
    let filename = PathBuf::from(String::from("/home/N3_3090U5/数据/项目数据集/VOC/final.ot"));
    vs.save(&filename).unwrap();
    println!("save checkpoint: {:?}",filename);
    Ok(())
}

fn main() -> Result<()> {
    train()
}