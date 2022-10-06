extern crate serde;
extern crate anyhow;
use river::Device;
use river::datasets::{
    det_dataset,
    det_dataset_iter,
    det_pipelines::compose::Compose,
};
#[test]
fn test_detection_dataset_train(){
    let s = String::from("{\"transforms\":[{\"type\":\"ResizeTorch\",\"target_height\":14,\"target_width\":14}]}");
    let compose = Compose::loads(&s);

    let obj:det_dataset::DetDataset = det_dataset::DetDataset::new(
        &String::from("tests/assets/detection/dataset/train_image_list.json"),
        &String::from("tests/assets/detection/dataset"),
        compose,
        true,
        true,
        Some(String::from("tests/assets/detection/dataset/train_anno.json")),
        Some(String::from("tests/assets/detection/dataset/category_info.json")),
    );
    let o = obj.prepare(0);
    let _s = obj.len();
    if let Some(res) = o{
        let _n = res.group_name();
    }

}

#[test]
fn test_detection_dataset_test(){
    let s = String::from("{\"transforms\":[{\"type\":\"ResizeTorch\",\"target_height\":14,\"target_width\":14}]}");
    let compose = Compose::loads(&s);

    let obj:det_dataset::DetDataset = det_dataset::DetDataset::new(
        &String::from("tests/assets/detection/dataset/train_image_list.json"),
        &String::from("tests/assets/detection/dataset"),
        compose,
        false,
        false,
        None,
        None,
    );
    let o = obj.prepare(0);
    let _s = obj.len();
    if let Some(res) = o{
        let _n = res.group_name();
    }
}

#[test]
fn test_detection_dataset_train_iter(){
    let s = String::from("{\"transforms\":[{\"type\":\"ResizeTorch\",\"target_height\":14,\"target_width\":14}]}");
    let compose = Compose::loads(&s);

    let obj:det_dataset::DetDataset = det_dataset::DetDataset::new(
        &String::from("tests/assets/detection/dataset/train_image_list.json"),
        &String::from("tests/assets/detection/dataset"),
        compose,
        true,
        true,
        Some(String::from("tests/assets/detection/dataset/train_anno.json")),
        Some(String::from("tests/assets/detection/dataset/category_info.json")),
    );

    for (idx, (bimages, gt_labels, gt_bboxes, ins_groups)) in obj.iter(1, Device::cuda_if_available()).enumerate(){
        println!("{} {:?} {:?} {:?} {:?}", idx, bimages, gt_labels, gt_bboxes, ins_groups);
    }

}

#[test]
fn test_detection_dataset_test_iter(){
    let s = String::from("{\"transforms\":[{\"type\":\"ResizeTorch\",\"target_height\":14,\"target_width\":14}]}");
    let compose = Compose::loads(&s);

    let obj:det_dataset::DetDataset = det_dataset::DetDataset::new(
        &String::from("tests/assets/detection/dataset/train_image_list.json"),
        &String::from("tests/assets/detection/dataset"),
        compose,
        false,
        false,
        None,
        None,
    );
    for (idx, (bimages, gt_labels, gt_bboxes, ins_groups)) in obj.iter(1, Device::cuda_if_available()).enumerate(){
        println!("{} {:?} {:?} {:?} {:?}", idx, bimages, gt_labels, gt_bboxes, ins_groups);
    }
}

#[test]
fn test_detection_dataset_load_dump(){
    let s = String::from("tests/assets/detection/dataset/det_dataset_0.json");
    let obj:det_dataset::DetDataset = det_dataset::DetDataset::load_by_file(&s);
    obj.dump_to_file(
        &std::env::temp_dir()
        .join(format!("det_dataset_0.json"))
        .into_os_string()
        .into_string()
        .unwrap()
    );
}

// #[test]
// fn test_detection_dataset_voc_train_iter(){
//     let s = String::from("{\"transforms\":[{\"type\":\"ResizeTorch\",\"target_height\":14,\"target_width\":14}]}");
//     let compose = Compose::loads(&s);
//
//     let obj:det_dataset::DetDataset = det_dataset::DetDataset::new(
//         &String::from("/Users/zhoujinghui/Documents/数据/项目数据集/VOCdevkit/VOC/train_image_list.json"),
//         &String::from("/Users/zhoujinghui/Documents/数据/项目数据集/VOCdevkit/VOC"),
//         compose,
//         true,
//         true,
//         Some(String::from("/Users/zhoujinghui/Documents/数据/项目数据集/VOCdevkit/VOC/train_anno.json")),
//         Some(String::from("/Users/zhoujinghui/Documents/数据/项目数据集/VOCdevkit/VOC/category_info.json")),
//     );
//
//     for (idx, (bimages, gt_labels, gt_bboxes, ins_groups)) in obj.iter(2).enumerate(){
//         println!("{} {:?} {:?} {:?} {:?}", idx, bimages, gt_labels, gt_bboxes, ins_groups);
//     }
//
// }