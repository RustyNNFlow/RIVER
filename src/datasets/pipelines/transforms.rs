use serde::{Serialize, Deserialize};
use crate::{
    datasets::dataset_result,
    vision::image as image_op,
};



#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ResizeTorch{
    target_height: i64,
    target_width: i64,
}
impl ResizeTorch{
    pub fn new(
        target_height: i64,
        target_width: i64,
    )->ResizeTorch{
        ResizeTorch{
            target_height: target_height,
            target_width: target_width,
        }
    }
    pub fn call(
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
}