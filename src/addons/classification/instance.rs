use serde::{Serialize, Deserialize};
use crate::datasets::category_info;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct  ClsInstanceData{
    category:String,
    score:Option<f64>,
    pub category_index:Option<i64>,
}
impl ClsInstanceData{
    pub fn update_category_index(&mut self, index:i64){
        self.category_index = Some(index);
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ClsGroupUsage{
    image_name:String,
    image_height:i64,
    image_width:i64,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ClsInstancesGroup{
    pub data:Vec<ClsInstanceData>,
    usage:ClsGroupUsage,
}

impl ClsInstancesGroup{
    pub fn loads(json_str: &String)->ClsInstancesGroup{
        serde_json::from_str(json_str).unwrap()
    }
    pub fn dumps(&self)->String{
        serde_json::to_string(&self).unwrap()
    }
    pub fn group_name(&self)->String{
        self.usage.image_name.clone()
    }
    pub fn instance_num(&self)->usize{
        self.data.len()
    }
    pub fn image_height(&self)->i64{
        self.usage.image_height
    }
    pub fn image_width(&self)->i64{
        self.usage.image_width
    }
    pub fn update_category_index(&mut self, category_info: category_info::CategoryInfo){
        for d in self.data.iter_mut(){
            let idx=category_info.cat2id.get(&d.category).unwrap();
            d.update_category_index(*idx);
        }

    }
    pub fn new(
        category:&String,
        image_name:&String,
        image_height:i64,
        image_width:i64,
        score:f64,
    )->ClsInstancesGroup
    {
        let mut vec_data: Vec<ClsInstanceData>= Vec::new();
        vec_data.push(
            ClsInstanceData
            {
                category:category.clone(),
                score:Some(score),
                category_index:None,
            }
        );
        ClsInstancesGroup{
            data:vec_data,
            usage:ClsGroupUsage{
                image_name:image_name.clone(),
                image_height:image_height,
                image_width:image_width,
            },
        }
    }
}