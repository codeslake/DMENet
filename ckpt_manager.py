import os 
from shutil import *

import tensorlayer as tl
import numpy as np

class CKPT_Manager:
    def __init__(self, root_dir, model_name, max_files_to_keep = 10):
        self.root_dir = root_dir
        self.model_name = model_name
        self.max_files = max_files_to_keep

        self.ckpt_list = os.path.join(self.root_dir, 'checkpoints')

    def load_ckpt(self, sess):
        # read file
        with open(self.ckpt_list, 'r') as file:
            lines = file.read().splitlines()
            file.close()

        # get ckpt path
        file_path = lines[0].split(' ')[0]

        tl.files.load_and_assign_npz_dict(name = file_path, sess = sess)

    def save_ckpt(self, sess, save_vars, epoch, score):
        file_name = self.model_name + '_' + str(epoch) + '.npz'
        save_path = os.path.join(self.root_dir, file_name)

        tl.files.save_npz_dict(save_vars, name = save_path, sess = sess)

        with open(self.ckpt_list, 'a') as file:
            file.write(save_path + ' ' + str(score) + os.linesep)
            file.close()

        self._update_files()

    def _update_files(self):
        # read file
        with open(self.ckpt_list, 'r') as file:
            lines = file.read().splitlines()
            file.close()

        # sort by score
        lines = self._sort(lines)

        # delete ckpt
        while len(lines) > self.max_files:
            line_to_remove = lines[len(lines) - 1]
            os.remove(line_to_remove.split(' ')[0])
            del(lines[len(lines) - 1])

        # update ckpt list
        with open(self.ckpt_list, 'w') as file:
            for line in lines:
                file.write(line + os.linesep)
            file.close()

    def _sort(self, lines):
        scores = [float(score.split(' ')[1]) for score in lines]
        lines = [line for _, line in sorted(zip(scores, lines))]

        return lines

