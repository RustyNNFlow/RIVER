use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag="type")]
pub enum ClsInstanceData{
    Train {category:String},
    Infer {category:String, score:f32},
}
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag="type")]
pub struct ClsGroupUsage{
    image_name:String,
    image_height:i32,
    image_width:i32,
}
#[derive(Serialize, Deserialize, Debug)]
#[serde(tag="type")]
pub struct ClsInstancesGroup{
    data:Vec<ClsInstanceData>,
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
    pub fn image_height(&self)->i32{
        self.usage.image_height
    }
    pub fn image_width(&self)->i32{
        self.usage.image_width
    }
}