//! Dataset iterators.
use crate::{
    Device,
    TchError,
    Tensor,
    addons::detection::instance,
};
use super::det_dataset::DetDataset;

#[derive(Debug)]
pub struct DatasetIter {
    dataset:DetDataset,
    batch_index:usize,
    batch_size: usize,
    total_size: usize,
    device: Device,
    return_smaller_last_batch: bool,
}

impl DatasetIter {

    pub fn f_new(dataset: DetDataset, batch_size: usize) -> Result<DatasetIter, TchError> {
        let total_size = dataset.len();

        Ok(DatasetIter {
            dataset:dataset,
            batch_index:0,
            batch_size:batch_size,
            total_size:total_size,
            device: Device::Cpu,
            return_smaller_last_batch: false,
        })
    }

    pub fn new(dataset: DetDataset, batch_size: usize) -> DatasetIter {
        DatasetIter::f_new(dataset, batch_size).unwrap()
    }

    /// Shuffles the dataset.
    ///
    /// The iterator would still run over the whole dataset but the order in
    /// which elements are grouped in mini-batches is randomized.
    pub fn shuffle(&mut self) -> &mut DatasetIter {
        self
    }

    /// Transfers the mini-batches to a specified device.
    #[allow(clippy::wrong_self_convention)]
    pub fn to_device(&mut self, device: Device) -> &mut DatasetIter {
        self.device = device;
        self
    }

    /// When set, returns the last batch even if smaller than the batch size.
    pub fn return_smaller_last_batch(&mut self) -> &mut DatasetIter {
        self.return_smaller_last_batch = true;
        self
    }

    pub fn total_size(&self)->usize{
        self.total_size
    }
}

impl Iterator for DatasetIter {
    type Item = (Tensor, Vec<Tensor>, Vec<Tensor>, Vec<instance::DetInstancesGroup>);

    fn next(&mut self) -> Option<Self::Item> {
        let start = self.batch_index * self.batch_size;
        let size = std::cmp::min(self.batch_size, self.total_size - start);
        if size <= 0 || (!self.return_smaller_last_batch && size < self.batch_size) {
            None
        } else {
            self.batch_index += 1;
            let mut xs: Vec<Tensor> = Vec::new();
            let mut gt_labels: Vec<Tensor> = Vec::new();
            let mut gt_bboxes: Vec<Tensor> = Vec::new();
            let mut ins_groups: Vec<instance::DetInstancesGroup> = Vec::new();
            for i in start..start+size{
                let o = self.dataset.prepare(i);
                if let Some(res) = o {
                    xs.push(res.x());
                    gt_labels.push(res.gt_labels().to_device(self.device));
                    gt_bboxes.push(res.gt_bboxes().to_device(self.device));
                    ins_groups.push(res.instances_group);
                }
            }
            Some((
                Tensor::concat(&xs, 0).to_device(self.device),
                gt_labels,
                gt_bboxes,
                ins_groups,
            ))
        }
    }
}
