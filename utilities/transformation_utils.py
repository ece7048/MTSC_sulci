#Author: Michail Mamalakis
#Version: 0.1
#Licence:
from __future__ import division, print_function
from collections.abc import Callable, Hashable, Mapping, Sequence
import os
import torch
import sys
from copy import deepcopy
from typing import Tuple, Dict, Union, Optional
from torch.utils.data import Dataset, DataLoader
#from torchdata.datapipes.iter import IterableWrapper
from monai.data.meta_obj import get_track_meta
from torchvision.transforms import InterpolationMode
from monai.transforms.inverse import InvertibleTransform, TraceableTransform
from monai.data.meta_tensor import MetaTensor
from monai.utils import ImageMetaKey as Key
import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage.transform import resize
from monai.transforms.transform import LazyTransform,Transform
from monai.transforms import (
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureTyped,
    FgBgToIndicesd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    RandFlip,
    RandRotate,
    RandZoom,
    RandAffineGrid,
    RandGaussianNoise,
    RandShiftIntensity,
    NormalizeIntensity,
    MapTransform,
    Crop,
)
from monai.utils import (
    LazyAttr,
    Method,
    PytorchPadMode,
    TraceKeys,
    TransformBackends,
    convert_data_type,
    convert_to_tensor,
    deprecated_arg_default,
    ensure_tuple,
    ensure_tuple_rep,
    fall_back_tuple,
    look_up_option,
    pytorch_after,
)
from torch import Tensor
from functools import partial
from monai import transforms
from monai.transforms import (
    AsDiscrete,
    Activations,
    ScaleIntensityRangePercentiles,
)

from monai.config import print_config, KeysCollection
from monai import data

from monai.config.type_definitions import NdarrayOrTensor
from monai.utils.enums import TransformBackends
from monai.transforms.traits import MultiSampleTrait



class ConvertToMultiChannelsulcalClasses(Transform):

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]
    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:

        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)

        result = [(img == 2), (img == 1), (img == 0)]

        return torch.stack(result, dim=0) if isinstance(img, torch.Tensor) else np.stack(result, axis=0)

class ConvertToMultiChannelsulcalClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the background
    label 2 is the skeleton
    label 3 is the Region of interest
    

    """
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.converter = ConvertToMultiChannelsulcalClasses()

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d

class SpatialCrop(Crop):
    """
    Crop at the given center of image with specified ROI size.
    If mension of the expected ROI size is larger than the input image size, will not crop that dimension.
    So the cropped result may be smaller than the expected ROI, and the cropped results of several images may
    not have exactly the same shape.

    This trorm is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        roi_size: the spatial size of the crop region e.g. [224,224,128]
            if mension of ROI size is larger than image size, will not crop that dimension of the image.
            If its nts have non-positive values, the corresponding size of input image will be used.
            for example: ispatial size of input data is [40, 40, 40] and `roi_size=[32, 64, -1]`,
            the spatial size of output data will be [32, 40, 40].
        lazy: a flag to indicate whether this transform should execute lazily or not. Defaults to False.
    """

    def __init__(self, roi_size: Union[Sequence[int], int], lazy: bool = False) -> None:
        super().__init__(lazy=lazy)
        self.roi_size = roi_size

    def compute_slices(self, spatial_size: Sequence[int], roi_center: Union[Sequence[int], int]) -> tuple[slice]:  # type: ignore[override]
        roi_size = fall_back_tuple(self.roi_size, spatial_size)
        if len(roi_center)==1:
            roi_center_up = [roi_center for i in spatial_size]
        else:
            roi_center_up=roi_center
        return super().compute_slices(roi_center=roi_center_up, roi_size=roi_size)


    def __call__(self, img: torch.Tensor,roi_center: Union[Sequence[int], int], lazy: Optional[bool]= None) -> torch.Tensor:  # type: ignore[override]
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.

        """
        lazy_ = self.lazy if lazy is None else lazy
        return super().__call__(
            img=img,
            slices=self.compute_slices(img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:],roi_center),
            lazy=lazy_,
        )


class SpatialCropSamples(TraceableTransform, LazyTransform, MultiSampleTrait):
    """
    Crop image with random size or specific size ROI to generate a list of N samples.
    It can crop at a random position as center or at the image center. And allows to set
    the minimum size to limit the randomly generated ROI.
    It will return a list of cropped images.

    Note: even `random_size=False`, if a dimension of the expected ROI size is larger than the input image size,
    will not crop that dimension. So the cropped result may be smaller than the expected ROI, and the cropped
    results of several images may not have exactly the same shape.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        roi_size: if `random_size` is True, it specifies the minimum crop region.
            if `random_size` is False, it specifies the expected ROI size to crop. e.g. [224, 224, 128]
            if a dimension of ROI size is larger than image size, will not crop that dimension of the image.
            If its components have non-positive values, the corresponding size of input image will be used.
            for example: if the spatial size of input data is [40, 40, 40] and `roi_size=[32, 64, -1]`,
            the spatial size of output data will be [32, 40, 40].
        num_samples: number of samples (crop regions) to take in the returned list.
        max_roi_size: if `random_size` is True and `roi_size` specifies the min crop region size, `max_roi_size`
            can specify the max crop region size. if None, defaults to the input image size.
            if its components have non-positive values, the corresponding size of input image will be used.
        random_center: crop at random position as center or the image center.
        random_size: crop with random size or specific size ROI.
            The actual size is sampled from `randint(roi_size, img_size)`.
        lazy: a flag to indicate whether this transform should execute lazily or not. Defaults to False.

    Raises:
        ValueError: When ``num_samples`` is nonpositive.

    """

    backend = SpatialCrop.backend

    def __init__(
        self,
        roi_size: Union[Sequence[int], int],
        image_size: Union[Sequence[int], int] = None,
        lazy: bool = False,
    ) -> None:
        LazyTransform.__init__(self, lazy)
        if isinstance(roi_size, list):
            roi_s=roi_size
        else:
            roi_s=[roi_size,roi_size,roi_size]
        self.roi=roi_size

        if len(image_size)==1:
            image_s=[image_size,image_size,image_size]
        else:
            image_s=image_size
        self.image_size=image_size
        self.num_samples=1
        self.cropper = SpatialCrop(roi_s, lazy)
        if self.roi!=1:
            sx=int(image_s[0]/roi_s[0])
            sy=int(image_s[1]/roi_s[1])
            sz=int(image_s[2]/roi_s[2])
            difx=int((image_s[0]-sx*(roi_s[0]))/2)
            dify=int((image_s[1]-sy*(roi_s[1]))/2)
            difz=int((image_s[2]-sz*(roi_s[2]))/2)
            if difx>0:
                sx=sx+1 
                difx=int((abs(image_s[0]-sx*(roi_s[0])))/2)
            if dify>0:
                sy=sy+1
            dify=int((abs(image_s[1]-sy*(roi_s[1])))/2)
            if difz>0:
                sz=sz+1
                difz=int((abs(image_s[2]-sz*(roi_s[2])))/2)
            self.num_samples = int(sx*sy*sz)

            self.cropper = SpatialCrop(roi_s, lazy)
            img_center=[]
            start_point_center=[int(roi_s[0]/2)+1, int(roi_s[1]/2)+1, int(roi_s[2]/2)+1]
            j,k,l=0,0,0
            for i in range(self.num_samples):
                if i==0:
                    c1=start_point_center[0]
                    c2=start_point_center[1]
                    c3=start_point_center[2]
                elif i<sx:  # (int(image_s[0]/roi_s[0])):
                    c1=start_point_center[0]+(i*(start_point_center[0]-1)-difx) 
                    c2=start_point_center[1]
                    c3=start_point_center[2]
                elif i<(sx*sy):  # (int(image_s[0]/roi_s[0])*int(image_s[1]/roi_s[1])):
                    c1=start_point_center[0]+(j*(start_point_center[0]-1)-difx)
                    c2=start_point_center[1]+(k*(start_point_center[1]-1)-dify)
                    c3=start_point_center[2]
                    if k<=sy:   # (int(image_s[1]/roi_s[1])):
                        k=k+1
                    else:
                        j=j+1
                        k=1
                else:
                    if i==(sx*sy): # (int(image_s[0]/roi_s[0])*int(image_s[1]/roi_s[1])):
                        j,k=0,0
                    c1=start_point_center[0]+(j*(start_point_center[0]-1)-difx)
                    c2=start_point_center[1]+(k*(start_point_center[1]-1)-dify)
                    c3=start_point_center[2]+(l*(start_point_center[2]-1)-difz)
                    if l<=sz: # (int(image_s[2]/roi_s[2])):
                        l=l+1
                    elif k<=sy: # (int(image_s[1]/roi_s[1])):
                        l=1
                        k=k+1
                    else:
                        k=1
                        l=1
                        j=j+1
                img_center.append([c1,c2,c3])
            self.img_center=img_center

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, value: bool) -> None:
        self._lazy = value
        self.cropper.lazy = value


    def __call__(self, img: torch.Tensor, lazy:Optional[bool]= None) -> list[torch.Tensor]:
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        cropping doesn't change the channel dim.
        """
        ret = []
        lazy_ = self.lazy if lazy is None else lazy
        check_size=[img.shape[1],img.shape[2],img.shape[3]]
        if (self.image_size!=check_size):
            img_red=np.squeeze(img)
            resize_s=[img.shape,self.image_size[0],self.image_size[1],self.image_size[2]]
            img_upd=resize(img_red, self.image_size, order=1,mode='symmetric', cval=0,clip=True,anti_aliasing=True,anti_aliasing_sigma=None)
            #https://docs.monai.io/en/0.1.0/_modules/monai/transforms/transforms.html
            img_upd=np.expand_dims(img_upd,0)
        else:
            img_upd=img
        if self.roi==1:
            print('no crop just resize')
            ret=img_upd
        else:    
            for i in range(self.num_samples):
                cropped = self.cropper(img_upd,self.img_center[i], lazy=lazy_)
                if get_track_meta():
                    cropped.meta[Key.PATCH_INDEX] = i  # type: ignore
                    self.push_transform(cropped, replace=True, lazy=lazy_)  # track as this class instead of RandSpatialCrop
                ret.append(cropped)
        return ret


class SpatialCropSamplesd( MapTransform, LazyTransform, MultiSampleTrait):
    """
    Dictionary-based version :py:class:`monai.transforms.RandSpatialCropSamples`.
    Crop image with random size or specific size ROI to generate a list of N samples.
    It can crop at a random position as center or at the image center. And allows to set
    the minimum size to limit the randomly generated ROI. Suppose all the expected fields
    specified by `keys` have same shape, and add `patch_index` to the corresponding metadata.
    It will return a list of dictionaries for all the cropped images.

    Note: even `random_size=False`, if a dimension of the expected ROI size is larger than the input image size,
    will not crop that dimension. So the cropped result may be smaller than the expected ROI, and the cropped
    results of several images may not have exactly the same shape.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        roi_size: if `random_size` is True, it specifies the minimum crop region.
            if `random_size` is False, it specifies the expected ROI size to crop. e.g. [224, 224, 128]
            if a dimension of ROI size is larger than image size, will not crop that dimension of the image.
            If its components have non-positive values, the corresponding size of input image will be used.
            for example: if the spatial size of input data is [40, 40, 40] and `roi_size=[32, 64, -1]`,
            the spatial size of output data will be [32, 40, 40].
        num_samples: number of samples (crop regions) to take in the returned list.
        max_roi_size: if `random_size` is True and `roi_size` specifies the min crop region size, `max_roi_size`
            can specify the max crop region size. if None, defaults to the input image size.
            if its components have non-positive values, the corresponding size of input image will be used.
        random_center: crop at random position as center or the image center.
        random_size: crop with random size or specific size ROI.
            The actual size is sampled from `randint(roi_size, img_size)`.
        allow_missing_keys: don't raise exception if key is missing.
        lazy: a flag to indicate whether this transform should execute lazily or not. Defaults to False.

    Raises:
        ValueError: When ``num_samples`` is nonpositive.

    """

    backend = SpatialCropSamples.backend

    def __init__(
        self,
        keys: KeysCollection,
        roi_size: Union[Sequence[int], int],
        image_size: Union[Sequence[int], int] = None,
        allow_missing_keys: bool = False,
        lazy: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        LazyTransform.__init__(self, lazy)
        self.cropper = SpatialCropSamples(
            roi_size, image_size, lazy=lazy
        )

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, value: bool) -> None:
        self._lazy = value
        self.cropper.lazy = value

    def __call__(
        self, data: Mapping[Hashable, torch.Tensor], lazy: Optional[bool] = None
    ) -> list[dict[Hashable, torch.Tensor]]:
        ret: list[dict[Hashable, torch.Tensor]] = [dict(data) for _ in range(self.cropper.num_samples)]
        # deep copy all the unmodified data
        for i in range(self.cropper.num_samples):
            for key in set(data.keys()).difference(set(self.keys)):
                ret[i][key] = deepcopy(data[key])

        lazy_ = self.lazy if lazy is None else lazy
        for key in self.key_iterator(dict(data)):
            #self.cropper.set_random_state(seed=self.sub_seed)
            for i, im in enumerate(self.cropper(data[key], lazy=lazy_)):
                ret[i][key] = im
        return ret
