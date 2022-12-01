import numpy as np
import pandas as pd
import json
from pandas import json_normalize

def main():
    # train.json
    labeled_data_path = "/opt/ml/dataset/fold_0_remove_train.json"
    with open(labeled_data_path) as f:
        labeled_data = json.load(f)

    df_images = json_normalize(labeled_data['images'])
    df_annotations = json_normalize(labeled_data['annotations'])

    # 마지막 요소의 값들 가져오기
    width, height, _, license, flickr_url, coco_url, date_captured, image_id_1 = df_images.tail(1).values[0]
    image_id_2, category_id, area, bbox, iscrowd, anno_id = df_annotations.tail(1).values[0]

    # Unlabeled data (submission)
    submission_csv = '/opt/ml/baseline/UniverseNet/ensemble/submission_12.csv' 
    data = pd.read_csv(submission_csv, keep_default_na=False)
    data = data.values.tolist()

    unlabeled = dict() # json 변환을 위한 dictionary
    unlabeled['images'] = []
    unlabeled['annotations'] = []

    # 변환해가며 확인하기
    confidence_threshold = 0.7

    for predict, image in data:
        # pass if not predicted data
        if predict == None: 
            continue
        predict = predict.strip() # 띄어쓰기만 있는 경우가 있을 수 있음
        if predict == '': # 예측하지 못한 데이터는 pass
            continue

        count = 0 # annotation 개수 체크
        split_predict = predict.split(' ')
        anns_length = len(split_predict) // 6 # annotation 개수
        image_save = False # 이미지 저장 여부
        tmp_image, tmp_annotation = dict(), dict()
        for i in range(anns_length):
            class_ = int(split_predict[i*6])
            confidence = float(split_predict[(i*6)+1])
            Left = float(split_predict[(i*6)+2])
            Top = float(split_predict[(i*6)+3])
            Right = float(split_predict[(i*6)+4])
            Bottom = float(split_predict[(i*6)+5])
            Width = Right - Left
            Height = Bottom - Top
            Area = round(Width * Height, 2)
            # confidence Threshold 걸은 경우
            if confidence_threshold != None and confidence < confidence_threshold: 
                continue
            
            # Image 추가
            if image_save == False: # 추가된 이미지인지 확인
                image_id_2 += 1
                tmp_image['width'] = width # 마지막 데이터 그대로 이용
                tmp_image['height'] = height # 마지막 데이터 그대로 이용
                tmp_image['file_name'] = image
                tmp_image['license'] = license # 마지막 데이터 그대로 이용
                tmp_image['flickr_url'] = flickr_url # 마지막 데이터 그대로 이용
                tmp_image['coco_url'] = coco_url # 마지막 데이터 그대로 이용
                tmp_image['date_captured'] = date_captured # 마지막 데이터 그대로 이용
                tmp_image['id'] = image_id_2
                image_save = True

            # Annotation 추가
            anno_id += 1
            count += 1
            tmp_annotation['image_id'] = image_id_2
            tmp_annotation['category_id'] = class_
            tmp_annotation['area'] = Area
            tmp_annotation['bbox'] = [round(Left, 1), round(Top, 1), round(Width, 1), round(Height, 1)]
            tmp_annotation['iscrowd'] = iscrowd # 마지막 데이터 그대로 이용
            tmp_annotation['id'] = anno_id

        if count > 0: # 주석이 그려진게 있다면
            unlabeled['images'].append(tmp_image)
            unlabeled['annotations'].append(tmp_annotation)

    labeled_data['images'] += unlabeled['images']
    labeled_data['annotations'] += unlabeled['annotations']

    with open("../../dataset/train_new_2.json", "w") as new_file:
        json.dump(unlabeled, new_file)


if __name__ == '__main__':
    main()