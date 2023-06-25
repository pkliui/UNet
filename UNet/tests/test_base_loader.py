import os
import unittest
from ddt import ddt
import shutil, tempfile


from UNet.data_handling.base import BaseDataLoader
from UNet.data_handling.unetdataset import UNetDataset


@ddt
class TestBaseDataLoader(unittest.TestCase):

    def setUp(self):
        # create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # remove temporary directory after the test
        shutil.rmtree(self.test_dir)

    def test_arguments(self):
        """
        test input arguments are existing and are either None or equal to expected default values
        """

        self.bdl = BaseDataLoader(dataset=["somedummydata", "somedummydata"],
                 batch_size=1,
                 validation_split=0,
                 shuffle_for_split=True,
                 random_seed_split=0)
        vals = [["somedummydata", "somedummydata"], 1, 0]
        print(self.bdl.__dict__)
        for idx, var in enumerate(["dataset", "batch_size", "validation_split"]):
            self.assertIn(var, self.bdl.__dict__)
            self.assertEqual(self.bdl.__dict__[var], vals[idx])

    def test_read_data(self):
        """
        test reading some non-existing data
        """
        # instantiate class
        self.bdl = BaseDataLoader(dataset=["somedummydata", "somedummydata", "somedummydata", "somedummydata"],
                                  batch_size=1,
                                  validation_split=0.25,
                                  test_split=0.25,
                                  shuffle_for_split=True,
                                  random_seed_split=0)
        self.assertEqual(self.bdl.dataset, ["somedummydata", "somedummydata", "somedummydata", "somedummydata"])
        self.assertEqual(self.bdl.batch_size, 1)
        self.assertEqual(self.bdl.validation_split, 0.25)
        self.assertEqual(self.bdl.test_split, 0.25)

    def test_read_batch(self):
        """
        Test reading batches of existing data
        following this file structure for PH2 data
            root_folder
            ├── folder_with_images
            │    ├── sampleID1
            │    │    ├── sampleID1_image
            │    │    │    └── image1.bmp
            │    │    └── sampleID1_mask
            │    │         └── mask1.bmp
            │    ├── sampleID2
            │    │    ├── sampleID2_image
            │    │    │    └── image2.bmp
            │    │    └── sampleID2_mask
            │    │         └── mask2.bmp
            │    ├──...
            ├── ...
        """
        num_images = 10
        image_size_linear = 572
        mask_size_linear = 388
        batch_size = 2
        validation_split = 0.2
        test_split = 0.2

        # Create dummy image and mask folders and files
        sample_data = []
        for i in range(1, num_images + 1):
            sampleID = f"sampleID{i}"
            image_file = f"image{i}.bmp"
            mask_file = f"mask{i}.bmp"
            sample_data.append({
                "sampleID": sampleID,
                "image_file": image_file,
                "mask_file": mask_file
            })

            sample_folder = os.path.join(self.test_dir, sampleID)
            os.makedirs(sample_folder)

            image_folder = os.path.join(sample_folder, f"{sampleID}_image")
            os.makedirs(image_folder)
            image_file_path = os.path.join(image_folder, image_file)
            with open(image_file_path, "w") as f:
                f.write("Dummy image data")

            mask_folder = os.path.join(sample_folder, f"{sampleID}_mask")
            os.makedirs(mask_folder)
            mask_file_path = os.path.join(mask_folder, mask_file)
            with open(mask_file_path, "w") as f:
                f.write("Dummy mask data")

        # Create the UNetDataset with the temporary image and mask files
        images_list = [
            os.path.join(self.test_dir, sample["sampleID"], f"{sample['sampleID']}_image", sample["image_file"])
            for sample in sample_data
        ]
        masks_list = [
            os.path.join(self.test_dir, sample["sampleID"], f"{sample['sampleID']}_mask", sample["mask_file"])
            for sample in sample_data
        ]


        # make unetdataset
        unet_data = UNetDataset(images_list=images_list,
                                masks_list=masks_list,
                                resize_required=True,
                                required_image_size=(image_size_linear, image_size_linear),
                                required_mask_size=(mask_size_linear, mask_size_linear))

        #
        # set args for dataloader

        # create the corresponding dataloader for training and validation
        self.bdl = BaseDataLoader(dataset=unet_data,
                                     batch_size=batch_size,
                                     validation_split=validation_split,
                                        test_split=test_split,
                                  shuffle_for_split=True,
                                  random_seed_split=0)
        # check the batch size
        self.assertEqual(self.bdl.batch_size, batch_size)
        # check the validation split
        self.assertEqual(self.bdl.validation_split, validation_split)
        # check the number of batches for validation
        self.assertEqual(len(self.bdl.val_loader), 1)
        self.assertEqual(len(self.bdl.test_loader), 1)
        # check the number of batches for training
        self.assertEqual(len(self.bdl.train_loader), 3)



