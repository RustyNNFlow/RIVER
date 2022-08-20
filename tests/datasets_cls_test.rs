extern crate serde;
extern crate anyhow;

#[test]
fn test_classification_dataset(){
    use river::datasets::cls_dataset;
    let obj:cls_dataset::ClsDataset = cls_dataset::ClsDataset::new(
        &String::from("tests/assets/classification/dataset/train_image_list.json"),
        &String::from("tests/assets/classification/dataset"),
        true,
        Some(String::from("tests/assets/classification/dataset/train_anno.json")),
    );
    let o = obj.prepare(0);
    println!("{:?}", o);

    let obj:cls_dataset::ClsDataset = cls_dataset::ClsDataset::new(
        &String::from("tests/assets/classification/dataset/train_image_list.json"),
        &String::from("tests/assets/classification/dataset"),
        false,
        None,
    );
    let o = obj.prepare(0);
    println!("{:?}", o);

}
