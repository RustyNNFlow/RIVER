use serde::{Serialize, Deserialize};
use crate::{
    datasets::dataset_result,
    vision::image as image_op,
    Kind,
};
use super::compose::TransformOps;



#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ResizeTorch{
    target_height: i64,
    target_width: i64,
}
impl ResizeTorch {
    pub fn new(
        target_height: i64,
        target_width: i64,
    ) -> ResizeTorch {
        ResizeTorch {
            target_height: target_height,
            target_width: target_width,
        }
    }
}
#[typetag::serde]
impl TransformOps for ResizeTorch{
    fn call(
        &self,
        res:dataset_result::DatasetResult,
    )->dataset_result::DatasetResult{
        let mut res_new = res;
        let img_new = image_op::resize(
            &res_new.img,
            self.target_width,
            self.target_height,
        ).unwrap();
        res_new.update_img(img_new);
        res_new.update_image_width(self.target_width);
        res_new.update_image_height(self.target_height);
        res_new
    }
    fn repr(&self)->String{
        String::from(format!("{:?}", self))
    }
}


#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ToFloat{
}
impl ToFloat {
    pub fn new() -> ToFloat {
        ToFloat{}
    }
}
#[typetag::serde]
impl TransformOps for ToFloat{
    fn call(
        &self,
        res:dataset_result::DatasetResult,
    )->dataset_result::DatasetResult{
        let mut res_new = res;

        res_new.update_img(res_new.img.to_kind(Kind::Float));
        res_new
    }
    fn repr(&self)->String{
        String::from(format!("{:?}", self))
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AddBatchDim{
}
impl AddBatchDim {
    pub fn new() -> AddBatchDim {
        AddBatchDim{}
    }
}
#[typetag::serde]
impl TransformOps for AddBatchDim{
    fn call(
        &self,
        res:dataset_result::DatasetResult,
    )->dataset_result::DatasetResult{
        let mut res_new = res;
        let mut size:Vec<i64> = res_new.img.size();
        size.insert(0, 1);
        res_new.update_img(res_new.img.view(size.as_slice()));
        res_new
    }
    fn repr(&self)->String{
        String::from(format!("{:?}", self))
    }
}
