import os
from pathlib import Path
import pickle

from typing import Optional
import numpy as np
import pandas as pd
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader, RandomSampler, random_split

from data.datasets import *
from data.datasets import AbstractDataset, AbstractDataset_Test
import test
from utils.data_related import find_healthy_subjects, find_indices_of_images, process_cmr_images

from pytorch_lightning.callbacks import Callback

from data.samplers import EventsBalancedBatchSampler, RandomDistributedSampler, EventsBalancedSampler, AnchorBalancedBatchSampler, AnchorBalancedSampler

class CMRDataModule(pl.LightningDataModule):
    def __init__(self, load_dir: str, processed_dir: str,  
                 all_feature_tabular_dir: str, biomarker_tabular_dir: str, dataloader_file_folder: str, 
                 cmr_path_pickle_name: str,
                 biomarker_table_pickle_name: str,
                 processed_table_pickle_name: str,
                 table_condition_dict: Optional[dict] = None,
                 processed_file_name: str = "processed_seg_allax.npz",
                 dataset_cls: Dataset = AbstractDataset, load_seg: bool = False,
                 num_train: int = 1000, num_val: int = 100, num_test: int = 100, 
                 train_num_per_epoch: int = 1000, 
                 all_value_names: list[str] = ["Age"],
                 target_value_name: str = "Age",
                 sax_slice_num: int = 8, 
                 batch_size: int = 16,
                 num_workers: int = 8,
                 image_size: list[int] = [128, 128],
                 augment: bool = True,
                 sorting_group: int = 5,
                 replace_processed: bool = False,
                 **kwargs):
        super().__init__()
        
        self.load_dir = load_dir
        self.processed_dir = processed_dir
        self.all_feature_tabular_dir = all_feature_tabular_dir
        self.biomarker_tabular_dir = biomarker_tabular_dir
        self.dataloader_file_folder = dataloader_file_folder
        self.processed_file_name = processed_file_name
        self.cmr_path_pickle_name = cmr_path_pickle_name
        self.biomarker_table_pickle_name = biomarker_table_pickle_name
        self.processed_table_pickle_name = processed_table_pickle_name
        self.table_condition_dict = table_condition_dict

        self.dataset_cls = dataset_cls
        self.load_seg = load_seg
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.train_num_per_epoch = train_num_per_epoch
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.all_value_names = all_value_names
        self.target_value_name = target_value_name
        self.image_size = image_size
        self.sax_slice_num = sax_slice_num
        self.augment = augment
        self.sorting_group = sorting_group
        self.replace_processed = replace_processed
        
        self.train_dset = None
        self.val_dset = None
        self.test_dset = None
        self.num_cases = num_train + num_val + num_test
        Path(processed_dir).mkdir(parents=True, exist_ok=True)
        Path(dataloader_file_folder).mkdir(parents=True, exist_ok=True)
        
    def setup(self, stage):
        if self.train_dset is not None and self.val_dset is not None:
            # trainer.fit seems to call datamodule.setup(), we don't wanna do it twice because our
            # model was already given the val_dset and we don't want it to change  
            return         
        cmr_path_pickle_dir = Path(self.dataloader_file_folder) / self.cmr_path_pickle_name
        biomarker_table_pickle_dir = Path(self.dataloader_file_folder) / self.biomarker_table_pickle_name
        processed_table_pickle_dir = Path(self.dataloader_file_folder) / self.processed_table_pickle_name
        
        # Load the value table containing all relevant features and the paths of processed images/segmentation maps
        if os.path.exists(biomarker_table_pickle_dir) and os.path.exists(cmr_path_pickle_dir) \
            and not self.replace_processed:
            target_table = pd.read_pickle(biomarker_table_pickle_dir)
            with open(cmr_path_pickle_dir, 'rb') as handle:
                paths = pickle.load(handle)
            imgs_n = len(paths["train"]) + len(paths["val"])+len(paths["test"])
            print(f"Loaded {imgs_n} images from {cmr_path_pickle_dir}.")
        else:
            if os.path.exists(processed_table_pickle_dir) and not self.replace_processed:
                processed_table = pd.read_pickle(processed_table_pickle_dir)
            else:
                # Find the subjects with enough sax slice number and image size
                subjects_ids = find_indices_of_images(self.load_dir, sax_bbox_size=self.image_size, 
                                                        lax_bbox_size=self.image_size)
                processed_ids_ = process_cmr_images(load_dir=self.load_dir, 
                                                     prep_dir=self.processed_dir, 
                                                     file_name=self.processed_file_name,
                                                     sax_bbox_size=self.image_size,
                                                     lax_bbox_size=self.image_size, 
                                                     replace_processed=self.replace_processed,
                                                     id_list=subjects_ids) 
                processed_ids_ = sorted(processed_ids_) # Subjects with CMR images
                # processed_ids_ = [int(i) for i in processed_ids_ if i.isnumeric()]
                biomarker_table = pd.read_csv(self.biomarker_tabular_dir) # Load biomarker tabular data
                col_names = biomarker_table.columns
                self.all_value_names = list(set(self.all_value_names).intersection(set(col_names))) # Keep the onces in table
                processed_idx = biomarker_table["eid_87802"].isin(processed_ids_) # Cases in the image list
                processed_table_ = biomarker_table.loc[processed_idx, ["eid_87802"] + self.all_value_names]
                processed_table_ = processed_table_.dropna() # Remove nan
                processed_table = self.apply_table_conditions(processed_table_, **self.table_condition_dict, 
                                                            condition_table_dir=self.all_feature_tabular_dir)
                processed_table.to_pickle(processed_table_pickle_dir)
            
            # Select the required target values
            column_keys = ["eid_87802"] + [self.target_value_name]
            target_table = processed_table[column_keys]
            target_table.to_pickle(biomarker_table_pickle_dir) # Save the selected indices with biomarkers
            
            # Split training, validation, and test subject paths
            subject_paths = []
            for parent in target_table["eid_87802"]:
                subject_path = Path(self.processed_dir) / str(parent) / Path(self.processed_file_name)
                subject_paths.append(subject_path)
                
            split = (self.num_train, self.num_val, self.num_test)
            topk_paths = subject_paths[:sum(split)]
            # topk_paths = subject_paths[7000:sum(split)+7000] # For visualization
            train_idxs, val_idxs, test_idxs = [list(s) for s in random_split(list(range(len(topk_paths))), split)]
            train_paths = [topk_paths[i] for i in train_idxs]
            val_paths = [topk_paths[i] for i in val_idxs]
            test_paths = [topk_paths[i] for i in test_idxs]
            paths = {"train": train_paths, "val": val_paths, "test": test_paths}
            with open(cmr_path_pickle_dir, 'wb') as handle:
                pickle.dump(paths, handle, protocol=pickle.HIGHEST_PROTOCOL)
            assert len(paths["train"]) == self.num_train
            assert len(paths["val"]) == self.num_val
            assert len(paths["test"]) == self.num_test
        
        self.train_dset = eval(f'{self.dataset_cls}')(paths["train"], target_table, self.target_value_name,
                                                      load_seg=self.load_seg, augs=self.augment, sax_slice_num=self.sax_slice_num)
        self.val_dset = eval(f'{self.dataset_cls}_Test')(paths["val"], target_table, self.target_value_name,
                                                         load_seg=self.load_seg, augs=self.augment, sax_slice_num=self.sax_slice_num)
        self.test_dset = eval(f'{self.dataset_cls}_Test')(paths["test"], target_table, self.target_value_name,
                                                          load_seg=self.load_seg, augs=self.augment, sax_slice_num=self.sax_slice_num)
        self._train_dataloader = DataLoader(self.train_dset, 
                                            batch_size=self.batch_size,
                                            sampler=RandomSampler(self.train_dset, num_samples=self.train_num_per_epoch),
                                            num_workers=self.num_workers, 
                                            pin_memory=True,
                                            persistent_workers=self.num_workers > 0)
        self._val_dataloader = DataLoader(self.val_dset, batch_size=self.batch_size, num_workers=0)
        self._test_dataloader = DataLoader(self.test_dset, batch_size=self.batch_size, num_workers=0)

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader
    
    @staticmethod
    def apply_table_conditions(table: pd.DataFrame, 
                               healthy_cases: bool=True,
                               sorting_with_age: bool=True,
                               condition_table_dir: Optional[str] = None,):
        """
        Generate a table containing the subject indices with its target biomarker values.
        The subjects in the table satifies the following conditions:
            - considered healthy,
            - have all required target cardiac biomarkers.
        """
        # Load healthy subjects indices ftom all feature tabular data
        if healthy_cases:
            assert condition_table_dir is not None, "All feature table is bot provided."
            print("Selecting healthy subjects takes a couple of minutes")
            healthy_idx_ = find_healthy_subjects(condition_table_dir)
            idx_ = table["eid_87802"]
            idx = list(set(idx_) & set(healthy_idx_))
            table = table.loc[table["eid_87802"].isin(idx)]

        # Sort the table based on 5 age groups evenly
        if sorting_with_age:
            age_groups = np.arange(table["Age"].min()-5, table["Age"].max()+5, 5)
            table["Age_group"] = pd.cut(table["Age"], bins=age_groups, labels=age_groups[:-1])
            for age_group in age_groups[:-1]:
                num = len(table.loc[table["Age_group"] == age_group])
                table.loc[table["Age_group"] == age_group, "Age_group_idx"] = range(num)
            table = table.sort_values(by=["Age_group_idx", "Age_group"])

        return table



class SetEpochCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        datamodule = trainer.datamodule
        if hasattr(datamodule, '_train_dataloader') and hasattr(datamodule._train_dataloader, 'sampler'):
            if isinstance(datamodule._train_dataloader.sampler, RandomDistributedSampler):
                datamodule._train_dataloader.sampler.set_epoch(trainer.current_epoch)



class WBDataModule(pl.LightningDataModule):
    def __init__(self,
                 load_dir: str,
                 labels_folder: str,
                 dataset_cls: Dataset = AbstractDataset, load_seg: bool = False,
                 num_train: int = 1000, num_val: int = 100, num_test: int = 100,
                 train_num_per_epoch: int = 1000,
                 all_value_names: list[str] = ["age"],
                 target_value_name: str = "age",
                 batch_size: int = 16,
                 num_workers: int = 8,
                 image_size: list[int] = [224, 168, 363],
                 augment: bool = True,
                 sorting_group: int = 5,
                 replace_processed: bool = False,
                 body_mask_dir: str = None,
                 multi_gpu: bool = False,
                 labels_file: str = "labels.csv",
                 augmentations: list = ["random_flip"],
                 both_contrast: bool = True,
                 return_body_mask: bool = True,
                 **kwargs):
        super().__init__()

        self.load_dir = load_dir
        self.labels_folder = labels_folder

        self.dataset_cls = dataset_cls
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.train_num_per_epoch = train_num_per_epoch
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.all_value_names = all_value_names
        self.target_value_name = target_value_name
        self.image_size = image_size
        self.augment = augment
        self.sorting_group = sorting_group
        self.replace_processed = replace_processed

        self.train_dset = None
        self.val_dset = None
        self.test_dset = None
        self.num_cases = num_train + num_val + num_test
        self.body_mask_dir = body_mask_dir
        self.multi_gpu = multi_gpu
        self.labels_file = labels_file
        self.augmentations = augmentations
        print("Using augmentations: ", self.augmentations)
        self.both_contrast = both_contrast
        self.return_body_mask = return_body_mask

    def setup(self, stage):

        labels = pd.read_csv(os.path.join(self.labels_folder, self.labels_file))
        labels_train = labels[labels["split"] == "train"]
        labels_val = labels[labels["split"] == "val"]
        labels_test = labels[labels["split"] == "test"]

        # reset index
        labels_train = labels_train.reset_index(drop=True)
        labels_val = labels_val.reset_index(drop=True)
        labels_test = labels_test.reset_index(drop=True)

        anchor_class = 1

        train_balanced = AnchorBalancedSampler(labels_train["event"].values, anchor_class=anchor_class, batch_size=self.batch_size)
        #train_balanced = EventsBalancedBatchSampler(labels_train["event"].values)
        #train_balanced = RandomSampler(self.train_dset, num_samples=self.train_num_per_epoch)
        val_balanced = AnchorBalancedSampler(labels_val["event"].values, anchor_class=anchor_class, batch_size=self.batch_size)
        #val_balanced = RandomSampler(self.val_dset, num_samples=self.train_num_per_epoch)
        #print("len val_balanced: ", len(val_balanced))
        #val_balanced = EventsBalancedBatchSampler(labels_val["event"].values)
        test_balanced = AnchorBalancedSampler(labels_test["event"].values, anchor_class=anchor_class, batch_size=self.batch_size)
        #test_balanced = EventsBalancedBatchSampler(labels_test["event"].values)
        


        self.train_dset = eval(f'{self.dataset_cls}')(
            labels=labels_train,
            target_value_name=self.target_value_name,
            load_dir=self.load_dir,
            augs=self.augment,
            img_size=self.image_size,
            body_mask_dir=self.body_mask_dir,
            augmentations=self.augmentations,
            both_contrast=self.both_contrast,
            return_body_mask=self.return_body_mask
        )
        self.val_dset = eval(f'{self.dataset_cls}_Test')(
            labels=labels_val,
            target_value_name=self.target_value_name,
            load_dir=self.load_dir,
            augs=self.augment,
            img_size=self.image_size,
            body_mask_dir=self.body_mask_dir,
            augmentations=self.augmentations,
            both_contrast=self.both_contrast,
            return_body_mask=self.return_body_mask
        )
        self.test_dset = eval(f'{self.dataset_cls}_Test')(
            labels=labels_test,
            target_value_name=self.target_value_name,
            load_dir=self.load_dir,
            augs=self.augment,
            img_size=self.image_size,
            body_mask_dir=self.body_mask_dir,
            augmentations=self.augmentations,
            both_contrast=self.both_contrast,
            return_body_mask=self.return_body_mask
        )

        #print("Train dataset size: ", len(self.train_dset))
        #print("Train per epoch: ", self.train_num_per_epoch)
        if self.multi_gpu:
            print("Using multi-gpu training")
            print("RandomDistributedSampler is initialized with train_num_per_epoch: ", self.train_num_per_epoch)
            per_gpu_batch_size = self.batch_size
            self._train_dataloader = DataLoader(self.train_dset,
                                                batch_size=per_gpu_batch_size,
                                                #sampler=RandomSampler(self.train_dset,
                                                #                 num_samples=self.train_num_per_epoch),
                                                sampler=RandomDistributedSampler(self.train_dset,
                                                                      num_samples=self.train_num_per_epoch),
                                                num_workers=self.num_workers,
                                                pin_memory=True,
                                                persistent_workers=self.num_workers > 0)

        else:
            if self.target_value_name == "survival":
                self._train_dataloader = DataLoader(self.train_dset,
                                                    batch_size=self.batch_size,
                                                    sampler=train_balanced,
                                                    num_workers=self.num_workers,
                                                    #pin_memory=False,
                                                    #persistent_workers=False,
                                                    pin_memory=True,
                                                    persistent_workers=self.num_workers > 0
                                                    #prefetch_factor=1,
                                                    )
            else:
                self._train_dataloader = DataLoader(self.train_dset,
                                            batch_size=self.batch_size,
                                            sampler=RandomSampler(self.train_dset,
                                                                  num_samples=self.train_num_per_epoch),
                                            num_workers=self.num_workers,
                                            pin_memory=True,
                                            persistent_workers=self.num_workers > 0)
        if self.target_value_name == "survival":
            self._val_dataloader = DataLoader(self.val_dset,
                                            batch_size=self.batch_size,
                                            sampler=val_balanced,
                                            num_workers=self.num_workers,
                                            #pin_memory=False,
                                            #persistent_workers=False,
                                            pin_memory=True,
                                            persistent_workers=self.num_workers > 0,
                                            drop_last=False
                                            )
        else:    
            self._val_dataloader = DataLoader(self.val_dset, batch_size=self.batch_size, num_workers=0)
        self._test_dataloader = DataLoader(self.test_dset, batch_size=self.batch_size, num_workers=0, drop_last=False)

        if "time_to_event" in labels.columns:
            self.durations = np.array(labels[labels["split"] == "train"]["time_to_event"].values.tolist())
        else:
            self.durations = None

        if "event" in labels.columns:
            #unique event labels 
            self.num_events = labels["event"].unique()
            self.num_events = len(self.num_events)
            print("num events: ", self.num_events)
            self.events = np.array(labels[labels["split"] == "train"]["event"].values.tolist())
        else:   
            self.num_events = None

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader