import os
class path:
    def __init__(self):
        self.dataset_path = 'data/vqa_datasets/vqa/'
        self.feature_path = 'data/vqa_datasets/coco_extract/'
        self.init_path()

    def init_path(self):
        self.img_feat_path = {
            'train': self.feature_path + 'train2014/',
            'val': self.feature_path + 'val2014/',
            'test': self.feature_path + 'test2015/',}

        self.question_path = {
            'train': self.dataset_path + 'v2_OpenEnded_mscoco_train2014_questions.json',
            'val': self.dataset_path + 'v2_OpenEnded_mscoco_val2014_questions.json',
            'test': self.dataset_path + 'v2_OpenEnded_mscoco_test2015_questions.json',
            'vg': self.dataset_path + 'VG_questions.json',}

        self.answer_path = {
            'train': self.dataset_path + 'v2_mscoco_train2014_annotations.json',
            'val': self.dataset_path + 'v2_mscoco_val2014_annotations.json',
            'vg': self.dataset_path + 'VG_annotations.json', }

        self.result_path = './vqaeval_result'
 

    def check_path(self):
        print('Checking dataset ...')
        for mode in self.img_feat_path:
            if not os.path.exists(self.img_feat_path[mode]):
                print(self.img_feat_path[mode] + 'NOT EXIST')
                exit(-1)

        for mode in self.question_path:
            if not os.path.exists(self.question_path[mode]):
                print(self.question_path[mode] + 'NOT EXIST')
                exit(-1)

        for mode in self.answer_path:
            if not os.path.exists(self.answer_path[mode]):
                print(self.answer_path[mode] + 'NOT EXIST')
                exit(-1)

