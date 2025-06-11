from email.mime import image
from math import e
from pathlib import Path
import torch
import pandas as pd
import pickle
import numpy as np
from torchvision import transforms as v2
from torch.utils.data import Dataset
from typing import Tuple, Optional

from utils import image_normalization
from data.transforms import *
import nibabel as nib
import torchio as tio
import gc


__all__ = ["Cardiac2DplusT", "Cardiac2DplusT_Test", "Cardiac3DSAX", "Cardiac3DSAX_Test", 
           "Cardiac3DLAX", "Cardiac3DLAX_Test", "Cardiac3DplusTSAX", "Cardiac3DplusTSAX_Test", 
           "Cardiac3DplusTLAX", "Cardiac3DplusTLAX_Test", "Cardiac3DplusTAllAX", "Cardiac3DplusTAllAX_Test",
           "WB3DWatFat_Test", "WB3DWatFat"]

EID_COL = "eid"

class AbstractDataset(Dataset):
    def __init__(self,
                 load_dir: str,
                 labels: pd.DataFrame,
                 target_value_name: list[int],
                 augs: bool = True,
                 num_classes: int = 4,
                 **kwargs):
        self.load_dir = load_dir
        self.labels = labels
        self.target_value_name = target_value_name
        self.augs = augs
        self.num_classes = num_classes
        self.z_seg_relative = kwargs.get("z_seg_relative", 4)
        self.augmentation = self._augment
        #self.view = self.get_view()
        #self.view = self.get_view()
        
    @property
    def _augment(self) -> bool:

        return self.augs
    
    def get_view(self) -> int:
        raise NotImplementedError

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        raise NotImplementedError("__getitem__ is not implemented for AbstractDataset")


class AbstractDataset_Test(AbstractDataset):
    
    @property
    def _augment(self) -> bool:
        return False

class WB3DWatFat(AbstractDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_size = kwargs.get("img_size", (224, 168, 363))
        self.body_mask_dir = kwargs.get("body_mask_dir", None)
        self.augmentations = kwargs.get("augmentations", None)
        self.both_contrast = kwargs.get("both_contrast", True)
        self.return_body_mask = kwargs.get("return_body_mask", True)
        self.slice_num = 2
        self.crop_or_pad = tio.CropOrPad(self.img_size)


    def _load_nifti_image(self, img_path):
        return nib.load(img_path).get_fdata(dtype=np.float32)  #added dtype

    def __getitem__(self, idx):
        batch_dict = {}
        img_name_wat = os.path.join(str(self.labels.loc[idx, EID_COL]), "wat.nii.gz")
        img_path_wat = os.path.join(self.load_dir, img_name_wat)
        image_nifti_wat = self._load_nifti_image(img_path_wat)
        image_nifti_wat = np.expand_dims(image_nifti_wat, axis=0)  # add channel dimension
        if self.both_contrast:
            img_name_fat = os.path.join(str(self.labels.loc[idx, EID_COL]), "fat.nii.gz")
            img_path_fat = os.path.join(self.load_dir, img_name_fat)
            image_nifti_fat = self._load_nifti_image(img_path_fat)
            image_nifti_fat = np.expand_dims(image_nifti_fat, axis=0)  # add channel dimension
            image_nifti_fat = self._min_max_normalize(image_nifti_fat)

        # Min-Max Normalization
        image_nifti_wat = self._min_max_normalize(image_nifti_wat)
        


        image_nifti_wat = self.crop_or_pad(image_nifti_wat)
        if self.both_contrast:
            image_nifti_fat = self.crop_or_pad(image_nifti_fat)
            image = np.concatenate((image_nifti_wat, image_nifti_fat), axis=0)
            del image_nifti_wat, image_nifti_fat
        else:
            image = image_nifti_wat
            del image_nifti_wat
        
        gc.collect()

        if self.augmentation:
            image = self._apply_augmentation(image)
        
        image = torch.from_numpy(image).float()  # [2, 224, 168, 363] torchio expects (C, W, H, D)
        # permute to [C, D, H, W]
        image = image.permute(0, 3, 2, 1) # [2, 363, 224, 168]

        batch_dict["image"] = image

        if self.return_body_mask:

            body_mask_path = os.path.join(self.body_mask_dir, str(self.labels.loc[idx, EID_COL]), "body_mask.nii.gz")
            body_mask = self._load_nifti_image(body_mask_path)
            # add 1 channel dimension
            body_mask = np.expand_dims(body_mask, axis=0)
            # crop or pad body mask
            body_mask = self.crop_or_pad(body_mask)

            # print the number of 1 in the body mask:

            # duplicate body mask for 2 channels
            if self.both_contrast:
                body_mask = body_mask.repeat(2, axis=0)
            body_mask = torch.from_numpy(body_mask).float()
            body_mask = body_mask.permute(0, 3, 2, 1)   # [2, 360, 168, 224]
            batch_dict["body_mask"] = body_mask

        if self.target_value_name:
            if self.target_value_name == "survival":
                t = self.labels.loc[idx, "time_to_event"]
                e = self.labels.loc[idx, "event"]
                target_value = (t, e)  
            else: 
                target_value = self.labels.loc[idx, self.target_value_name]
            #target_value = self._load_values(self._get_subject_id(idx))
        batch_dict["target_value"] = target_value

        batch_dict["sub_idx"] = idx

        batch_dict["eid"] = str(self.labels.loc[idx, EID_COL])

        return batch_dict

    def _load_values(self, subject_idx: int):
        """
        Just copied. has to be revised
        """
        target_value = self.labels[self.labels[EID_COL] == subject_idx][self.target_value_name]
        target_value = np.array(target_value.iloc[0].tolist(), dtype=np.float32)
        target_value = torch.from_numpy(target_value).reshape(1)
        return target_value

    def _get_subject_id(self, index):
        return int(self.labels[index].parent.name)

    def _apply_augmentation(self, im):
        im = self._apply_mask_boxes_augmentation(im)
        augmnentations_compose = self._get_augmentations_compose()
        transforms = tio.transforms.Compose(augmnentations_compose)

        return transforms(im)

    def  _get_augmentations_compose(self):
        augmentations_compose = []
        random_value = np.random.rand()
        for augm in self.augmentations:
            if augm == "random_flip":
                augmentations_compose.append(tio.transforms.RandomFlip(axes=0, p=0.5))
                augmentations_compose.append(tio.transforms.RandomFlip(axes=1, p=0.5))
                augmentations_compose.append(tio.transforms.RandomFlip(axes=2, p=0.5))
            elif augm == "random_blur":
                augmentations_compose.append(tio.transforms.RandomBlur(p=0.5, std=np.min([random_value, 0.5])))
            elif augm == "random_noise":
                augmentations_compose.append(tio.transforms.RandomNoise(p=0.5, std=np.min([random_value, 0.5])))
        return augmentations_compose
    
    def _apply_mask_boxes_augmentation(self, im):
        if "mask_boxes" in self.augmentations:
            nr_boxes = np.random.randint(0, 10)
            # apply _mask_boxes to the water and fat images
            image_nifti_wat = im[0, :, :, :]
            image_nifti_wat = self._mask_boxes(image_nifti_wat, range_box_size=(20, 60), nr_boxes=nr_boxes)
            if self.both_contrast:
                image_nifti_fat = im[1, :, :, :]
                image_nifti_fat = self._mask_boxes(image_nifti_fat, range_box_size=(20, 60), nr_boxes=nr_boxes)
            if self.both_contrast:
                im = np.stack((image_nifti_wat, image_nifti_fat), axis=0)
            else:
                im = image_nifti_wat
        return im

    
    def _min_max_normalize(self, image):
        """Applies min-max normalization to an image."""
        # Flatten the image to find min and max values
        min_val = np.min(image)
        max_val = np.max(image)

        # Handle edge case where min and max are the same (e.g., constant image)
        if max_val == min_val:
            return np.zeros_like(image)  # If all values are the same, return an array of zeros

        # Apply min-max normalization to scale to [0, 1]
        return (image - min_val) / (max_val - min_val)

    def get_view(self) -> int:
        if self.both_contrast:
            return 2
        else:
            return 1
        
    
    def _mask_boxes(self, image, range_box_size:Tuple, nr_boxes:int):
        for _ in range(nr_boxes):
            z = np.random.randint(0, image.shape[0])
            y = np.random.randint(0, image.shape[1])
            x = np.random.randint(0, image.shape[2])
            z_size = np.random.randint(range_box_size[0], range_box_size[1])
            y_size = np.random.randint(range_box_size[0], range_box_size[1])
            x_size = np.random.randint(range_box_size[0], range_box_size[1])
            image[z:z+z_size, y:y+y_size, x:x+x_size] = 0

        return image


class WB3DWatFat_Test(WB3DWatFat):

    @property
    def _augment(self) -> bool:
        return False


class Cardiac2DplusT(AbstractDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.slice_num = 1
        
    def __getitem__(self, index) -> Tuple[np.ndarray, int]:
        """Load image files in the form of 2D+t.
        
        images: [T, H, W] where S is the number of slices, T is the number of time frames, and X and Y are the spatial dimensions.
        age: int, which is the age of the subject in years.
        """
        subject_id = self.get_subject_id(index)
        im_data, seg_data = self.load_im_seg_arr(index, z_2D_random=True, 
                                                 z_seg_relative=self.z_seg_relative, z_num=self.slice_num)
        im_data = np.transpose(im_data, (2, 3, 0, 1))
        im_data = image_normalization(im_data)
        im_data = torch.from_numpy(im_data)
        if self.augmentation:
            im_data = self.apply_augmentations(im_data)
            
        if self.load_seg:
            seg_data = np.transpose(seg_data, (2, 3, 0, 1))
            seg_data = torch.from_numpy(seg_data)
            return im_data, seg_data, index
        else:
            target_value = self.load_values(subject_id)
            target_value = torch.from_numpy(target_value).reshape(1)
            return im_data, target_value, index

    def get_subject_id(self, index):
        return int(self.subject_paths[index].parent.name)
    
    def get_view(self) -> int:
        return 0 # For short-axis view #TODO: should include long-axis view
    
    def load_im_seg_arr(self, img_idx: int, 
                        z_2D_random: Optional[bool]=False,
                        z_seg_relative: Optional[int]=None,
                        z_num: Optional[int]=None,) -> Tuple[np.ndarray, np.ndarray]:
        """"Load short axis image and segmentation files in the form of 2D+t. 
        
        The slice is picked relative to the start of the LV. The output is in the form of [T, X, Y]. T is the number of time frames, X and Y are the spatial dimensions.
        :param img_idx: Index of image in dataset image list.
        :param z_seg_relative: Picked SAX slice relative to the start of the LV. If None, all slices are returned.
        :param z_num: Number of slices to return. If None, all slices are returned.
        """
        npy_path = self.subject_paths[img_idx]
        assert os.path.exists(npy_path), f"File not found: {npy_path}"
        if npy_path.name[-4:] == ".npy":
            process_npy = np.load(npy_path, allow_pickle=True).item()
        elif npy_path.name[-4:] == ".npz":
            process_npy = np.load(npy_path)
            
        sax_im_data = process_npy["sax"].astype(np.float32) # [H, W, S, T]
        lax_im_data = process_npy["lax"].astype(np.float32) # [H, W, S, T]
        seg_sax_data = process_npy["seg_sax"].astype(np.int32) # [H, W, S, T]
        seg_lax_data = process_npy["seg_lax"].astype(np.int32) # [H, W, 3, T], for 2ch, 3ch, and 4ch
        
        # Change segmentation labels of long-axis slices
        if self.slice_num != 3:
            seg_lax_data[seg_lax_data == 1] = 4
            seg_lax_data[seg_lax_data == 2] = 5
        # Pick a random 2D+t slice among the stack of multi-view.
        if z_2D_random:
            im_data = np.concatenate([lax_im_data, sax_im_data], axis=2)
            seg_data = np.concatenate([seg_lax_data, seg_sax_data], axis=2)
            
            z = np.random.randint(im_data.shape[2])
            slice_im_data = np.moveaxis(im_data[..., z : z + 1, :], 1, 0)
            slice_seg_data = np.moveaxis(seg_data[..., z : z + 1, :], 1, 0)
            return slice_im_data, slice_seg_data
        # Remove slice dimension and keep only 2D+time
        if z_seg_relative is not None:
            z_seg_start = (seg_sax_data[..., 0] == 1).any((0, 1)).argmax()
            z = z_seg_start + z_seg_relative
            seg_sax_data = seg_sax_data[:, :, z] # [H, W, T]
            sax_im_data = sax_im_data[:, :, z]
            assert len(sax_im_data.shape) == 3, f"Img path: {npy_path}"
        # Select only the from 3rd to 3+z_num slices with segmentation map
        if z_num is not None:
            z_seg_start = (seg_sax_data[..., 0] == 1).any((0, 1)).argmax() + 2
            z_max = min(z_seg_start + z_num, seg_sax_data.shape[-2])
            z_min = z_max - z_num
            seg_sax_data = seg_sax_data[..., z_min : z_max, :] # [H, W, z_num, T]
            sax_im_data = sax_im_data[..., z_min : z_max, :]
            assert sax_im_data.shape[-2] == z_num, f"Img path: {npy_path}, shape: {sax_im_data.shape}"
        # Mirror image on x=y so the RV is pointing left (standard short axis view)
        sax_im_data = np.moveaxis(sax_im_data, 1, 0)
        lax_im_data = np.moveaxis(lax_im_data, 1, 0)
        seg_sax_data = np.moveaxis(seg_sax_data, 1, 0)
        seg_lax_data = np.moveaxis(seg_lax_data, 1, 0)
        return sax_im_data, seg_sax_data, lax_im_data, seg_lax_data
    
    def load_values(self, subject_idx: int):
        """Load values from csv file."""
        target_value = self.target_table[self.target_table["eid_87802"] == subject_idx][self.target_value_name]
        target_value = np.array(target_value.iloc[0].tolist(), dtype=np.float32)
        return target_value
    
    def apply_augmentations(self, im):
        """Transform image using torchvision transforms.
        input: [..., T, H, W]
        output: [..., T, H, W]"""
        transforms = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=180),
        ])
        im = transforms(im)
        contrast = v2.RandomAutocontrast(p=0.5)
        im = contrast(im.unsqueeze(-3))
        im = im.squeeze(-3) # Remove channel dimension
        return im

class Cardiac2DplusT_Test(Cardiac2DplusT):
        
    @property
    def _augment(self) -> bool:
        return False


class Cardiac3DplusTSAX(Cardiac2DplusT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.slice_num = self.sax_slice_num

    def __getitem__(self, index):
        """Load short axis image files in the form of 3D+t.
        
        images: [S, T, H, W] where S is the number of slices, T is the number of time frames, and X and Y are the spatial dimensions.
        target_value: int, which is the target value of the subject.
        """
        subject_id = self.get_subject_id(index)
        im_data, seg_data, *_ = self.load_im_seg_arr(index, z_num=self.slice_num)
        im_data = np.transpose(im_data, (2, 3, 0, 1)) # Move the slice and time dimension to the front
        assert len(im_data.shape) == 4 and im_data.shape[0] == self.slice_num
        im_data = image_normalization(im_data)
        im_data = torch.from_numpy(im_data)
        if self.augmentation:
            im_data = self.apply_augmentations(im_data)
        if self.load_seg:
            seg_data = np.transpose(seg_data, (2, 3, 0, 1))
            seg_data = torch.from_numpy(seg_data)
            return im_data, seg_data, index
        else:
            target_value = self.load_values(subject_id)
            target_value = torch.from_numpy(target_value).reshape(1)
            return im_data, target_value, index

    def get_view(self) -> int:
        return 0 # For short-axis view
    

class Cardiac3DplusTSAX_Test(Cardiac3DplusTSAX):
    
    @property
    def _augment(self) -> bool:
        return False
    

class Cardiac3DSAX(Cardiac3DplusTSAX):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.slice_num = self.sax_slice_num
        self.frame_idx = kwargs.get("frame_idx", 25)

    def __getitem__(self, index):
        """Load short axis image files in the form of 3D+t.
        
        images: [S, H, W] where S is the number of slices, and X and Y are the spatial dimensions.
        target_value: int, which is the target value of the subject.
        """
        subject_id = self.get_subject_id(index)
        im_data, seg_data, *_ = self.load_im_seg_arr(index, z_num=self.slice_num)
        im_data = np.transpose(im_data, (2, 3, 0, 1)) # Move the slice and time dimension to the front
        assert len(im_data.shape) == 4 and im_data.shape[0] == self.slice_num
        im_data = im_data[:, self.frame_idx, ...]
        im_data = image_normalization(im_data)
        im_data = torch.from_numpy(im_data)

        if self.augmentation:
            im_data = self.apply_augmentations(im_data)
        if self.load_seg:
            seg_data = np.transpose(seg_data, (2, 3, 0, 1))
            seg_data = seg_data[:, self.frame_idx, ...]
            seg_data = torch.from_numpy(seg_data)
            return im_data, seg_data, index
        else:
            target_value = self.load_values(subject_id)
            target_value = torch.from_numpy(target_value).reshape(1)
            return im_data, target_value, index
    
    def get_view(self) -> int:
        return 0 # For short-axis view

class Cardiac3DSAX_Test(Cardiac3DSAX):
    
    @property
    def _augment(self) -> bool:
        return False
    
    
class Cardiac3DplusTLAX(Cardiac2DplusT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.slice_num = 3
        self.num_classes = 3

    def __getitem__(self, index):
        """Load long axis image files in the form of 3D+t.
        
        images: [S, T, H, W] where S is the number of slices, T is the number of time frames, and X and Y are the spatial dimensions.
        target_value: int, which is the target value of the subject.
        """
        # Load long axis images
        subject_id = self.get_subject_id(index)
        _, _, im_data, seg_data = self.load_im_seg_arr(index)
        im_data = np.transpose(im_data, (2, 3, 0, 1)) # Move the slice and time dimension to the front
        assert len(im_data.shape) == 4 and im_data.shape[0] == self.slice_num
        im_data = image_normalization(im_data)
        im_data = torch.from_numpy(im_data)
        if self.augmentation:
            im_data = self.apply_augmentations(im_data)
        if self.load_seg:
            seg_data = np.transpose(seg_data, (2, 3, 0, 1))
            seg_data = torch.from_numpy(seg_data)
            return im_data, seg_data, index
        else:
            target_value = self.load_values(subject_id)
            target_value = torch.from_numpy(target_value).reshape(1)
            return im_data, target_value, index
    
    def get_view(self) -> int:
        return 1 # For long-axis view

class Cardiac3DplusTLAX_Test(Cardiac3DplusTLAX):
    
    @property
    def _augment(self) -> bool:
        return False
    

class Cardiac3DLAX(Cardiac3DplusTLAX):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.slice_num = 3
        self.num_classes = 3
        self.frame_idx = kwargs.get("frame_idx", 25)

    def __getitem__(self, index):
        """Load short axis image files in the form of 3D+t.
        
        images: [S, H, W] where S is the number of slices, and X and Y are the spatial dimensions.
        target_value: int, which is the target value of the subject.
        """
        # Load long axis images
        subject_id = self.get_subject_id(index)
        _, _, im_data, seg_data = self.load_im_seg_arr(index)
        im_data = np.transpose(im_data, (2, 3, 0, 1)) # Move the slice and time dimension to the front
        assert len(im_data.shape) == 4 and im_data.shape[0] == self.slice_num
        im_data = im_data[:, self.frame_idx, ...]
        im_data = image_normalization(im_data)
        im_data = torch.from_numpy(im_data)
        if self.augmentation:
            im_data = self.apply_augmentations(im_data)
        if self.load_seg:
            seg_data = np.transpose(seg_data, (2, 3, 0, 1))
            seg_data = seg_data[:, self.frame_idx, ...]
            seg_data = torch.from_numpy(seg_data)
            return im_data, seg_data, index
        else:
            target_value = self.load_values(subject_id)
            target_value = torch.from_numpy(target_value).reshape(1)
            return im_data, target_value, index
    
    def get_view(self) -> int:
        return 1 # For long-axis view


class Cardiac3DLAX_Test(Cardiac3DLAX):
    
    @property
    def _augment(self) -> bool:
        return False    
    

class Cardiac3DplusTAllAX(Cardiac2DplusT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = 6
        self.slice_num = self.sax_slice_num + 3

    def __getitem__(self, index):
        """Load image files in the form of 3D+t.
        
        images: [S, T, H, W] where S is the number of slices, T is the number of time frames, and X and Y are the spatial dimensions.
        target_value: int, which is the target value of the subject.
        """
        # Load short axis and long axis images
        subject_id = self.get_subject_id(index)
        sax_slice_num = self.slice_num-3
        sax_im_data, seg_sax_data, lax_im_data, seg_lax_data = self.load_im_seg_arr(index, z_num=sax_slice_num)
        
        im_data = np.concatenate([lax_im_data, sax_im_data], axis=2)
        im_data = np.transpose(im_data, (2, 3, 0, 1)) # Move the slice and time dimension to the front
        assert len(im_data.shape) == 4 and im_data.shape[0] == self.slice_num
        im_data = image_normalization(im_data)
        im_data = torch.from_numpy(im_data)
        if self.augmentation:
            im_data = self.apply_augmentations(im_data)
        if self.load_seg:
            # Relabel the long-axis segmentation
            seg_lax_data[..., 0, :][seg_lax_data[..., 0, :] == 1] = 4
            seg_lax_data[..., 2, :][seg_lax_data[..., 2, :] == 1] = 4
            seg_lax_data[..., 2, :][seg_lax_data[..., 2, :] == 2] = 5
            seg_data = np.concatenate([seg_lax_data, seg_sax_data], axis=2)
            seg_data = np.transpose(seg_data, (2, 3, 0, 1))
            seg_data = torch.from_numpy(seg_data)
            return im_data, seg_data, index
        else:
            target_value = self.load_values(subject_id)
            target_value = torch.from_numpy(target_value).reshape(1)
            return im_data, target_value, index
    
    def get_view(self) -> int:
        return 2 # For both long and short-axis views
    
    
class Cardiac3DplusTAllAX_Test(Cardiac3DplusTAllAX):
    
    @property
    def _augment(self) -> bool:
        return False



