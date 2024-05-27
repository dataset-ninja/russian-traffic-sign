import os, csv, glob
import shutil
from collections import defaultdict
import supervisely as sly
from supervisely.io.fs import (
    file_exists,
    get_file_name,
    get_file_name_with_ext,
    get_file_size,
)
from tqdm import tqdm

import src.settings as s
from dataset_tools.convert import unpack_if_archive


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    # Possible structure for bbox case. Feel free to modify as you needs.

    dataset_path = "/home/alex/DATASETS/IMAGES/russian traffic sign/rtsd-public/detection"
    batch_size = 30


    def create_ann(image_path):
        labels = []

        image_np = sly.imaging.image.read(image_path)[:, :, 0]
        img_height = image_np.shape[0]
        img_wight = image_np.shape[1]

        file_name = get_file_name_with_ext(image_path)

        data = name_to_data.get(file_name)
        if data is not None:
            data = list(map(list, set(map(tuple, data))))  # del duplicate data
            for curr_data in data:
                obj_class = meta.get_obj_class(curr_data[-2])
                tag_value = curr_data[-1]
                tag = sly.Tag(sign_type, value=tag_value)

                left = int(curr_data[0])
                top = int(curr_data[1])
                right = left + int(curr_data[2])
                bottom = top + int(curr_data[3])
                rect = sly.Rectangle(left=left, top=top, right=right, bottom=bottom)
                label = sly.Label(rect, obj_class, tags=[tag])
                labels.append(label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels)


    sign_type = sly.TagMeta(
        "type",
        sly.TagValueType.ONEOF_STRING,
        possible_values=["blue border", "blue rect", "danger", "main road", "mandatory", "prohibitory"],
    )

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(tag_metas=[sign_type])


    # where are the same images in rtsd-d*-frames
    all_train_images_pathes = glob.glob(dataset_path + "/*/train/*.jpg")
    train_images_names = []
    train_images_pathes = []
    for im_path in all_train_images_pathes:
        im_name = get_file_name(im_path)
        if im_name not in train_images_names:
            train_images_names.append(im_name)
            train_images_pathes.append(im_path)

    all_test_images_pathes = glob.glob(dataset_path + "/*/test/*.jpg")
    test_images_names = []
    test_images_pathes = []
    for im_path in all_test_images_pathes:
        im_name = get_file_name(im_path)
        if im_name not in test_images_names:
            test_images_names.append(im_name)
            test_images_pathes.append(im_path)


    all_anns_pathes = glob.glob(dataset_path + "/*/*/*.csv")

    name_to_data = defaultdict(list)

    classes_names = set()

    for ann_path in all_anns_pathes:
        tag_name = ann_path.split("/")[-2].replace("_", " ")

        with open(ann_path, "r") as file:
            csvreader = csv.reader(file)
            for idx, row in enumerate(csvreader):
                if idx == 0:
                    continue
                curr_data = row[1:]
                class_name = curr_data[-1]
                classes_names.add(class_name)
                curr_data.append(tag_name)
                name_to_data[row[0]].append(curr_data)

    obj_classes = []
    for name in classes_names:
        obj_class = sly.ObjClass(name, sly.Rectangle)
        obj_classes.append(obj_class)
    meta = meta.add_obj_classes(obj_classes)
    api.project.update_meta(project.id, meta.to_json())

    ds_name_to_data = {"train": train_images_pathes, "test": test_images_pathes}

    for ds_name, images_pathes in ds_name_to_data.items():
        progress = sly.Progress("Create dataset {}".format(ds_name), len(images_pathes))

        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

        for img_pathes_batch in sly.batched(images_pathes, batch_size=batch_size):
            img_names_batch = [get_file_name_with_ext(im_path) for im_path in img_pathes_batch]

            img_infos = api.image.upload_paths(dataset.id, img_names_batch, img_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]

            anns = [create_ann(image_path) for image_path in img_pathes_batch]
            api.annotation.upload_anns(img_ids, anns)

            progress.iters_done_report(len(img_names_batch))

    return project
