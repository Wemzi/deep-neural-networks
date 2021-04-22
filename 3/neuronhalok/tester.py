
#
# Automatic tests for ANN BSc part1, 2021 spring, HW2
#
# Authors: Balint Kovacs, Viktor Varga
#

import os
import urllib
import numpy as np

import tensorflow as tf
import tensorflow.keras.models
import tensorflow.keras.optimizers
import tensorflow.keras.layers
import tensorflow.keras.callbacks
import tensorflow.keras.activations

N_TASKS = 6
TASK_LIST = ["M4WZZO, A/1, B/2, C/1, D/1, F/1, H/3", "JBUU9D, A/2, B/2, C/2, D/3, F/2, H/1", "PXEH9T, A/3, B/2, C/2, D/3, F/1, H/3",
             "YMALQG, A/2, B/3, C/1, D/3, F/1, H/3", "BXUMGO, A/5, B/3, C/1, D/2, F/2, H/3", "DQK6TE, A/3, B/2, C/2, D/2, F/1, H/1",
             "NFHJBP, A/4, B/1, C/2, D/3, F/1, H/3", "CWERBU, A/4, B/3, C/2, D/2, F/1, H/1", "IHY4K7, A/6, B/3, C/1, D/2, F/1, H/3",
             "O1SHMO, A/2, B/3, C/2, D/3, F/2, H/3", "M4DSSN, A/5, B/2, C/2, D/3, F/1, H/3", "IIAG43, A/4, B/2, C/2, D/3, F/1, H/1",
             "G4Q311, A/4, B/2, C/1, D/2, F/1, H/3", "IW80QB, A/5, B/2, C/2, D/2, F/1, H/2", "GT9BQ9, A/2, B/2, C/2, D/1, F/1, H/3",
             "LGLFAT, A/3, B/2, C/1, D/1, F/2, H/3", "YRSBWC, A/6, B/3, C/2, D/2, F/2, H/3", "YO0D0V, A/3, B/1, C/2, D/1, F/2, H/3",
             "GHDW67, A/6, B/2, C/2, D/2, F/1, H/1", "CTHI78, A/1, B/2, C/1, D/2, F/2, H/2", "JWDNZY, A/1, B/2, C/2, D/3, F/1, H/2",
             "GOGMF5, A/1, B/1, C/2, D/3, F/1, H/1", "ONP3PA, A/1, B/3, C/1, D/2, F/2, H/2", "EMK76R, A/4, B/3, C/2, D/2, F/1, H/1",
             "X9UUS1, A/4, B/1, C/2, D/2, F/1, H/1", "C6GGX3, A/6, B/1, C/1, D/3, F/1, H/2", "VZGDQF, A/6, B/3, C/1, D/2, F/1, H/2",
             "J1UBK4, A/5, B/1, C/2, D/2, F/2, H/2", "S545O8, A/6, B/1, C/1, D/1, F/2, H/1", "FO5CT0, A/4, B/3, C/2, D/1, F/2, H/1",
             "OE85D7, A/5, B/1, C/1, D/2, F/2, H/3", "BB8ITG, A/5, B/3, C/1, D/2, F/1, H/3", "WTQ64H, A/3, B/3, C/1, D/3, F/1, H/3",
             "RL4ESW, A/1, B/1, C/2, D/2, F/1, H/3", "CQ4C98, A/6, B/3, C/2, D/1, F/1, H/2", "EXB2DG, A/2, B/1, C/2, D/3, F/2, H/1",
             "BNECNK, A/4, B/2, C/2, D/3, F/1, H/1", "GWJIAO, A/2, B/1, C/1, D/1, F/2, H/1", "BQ2RKO, A/2, B/2, C/2, D/2, F/1, H/2",
             "G3I6O6, A/5, B/2, C/2, D/1, F/2, H/2", "IE3RQX, A/6, B/2, C/2, D/2, F/1, H/1", "SN20U1, A/2, B/3, C/2, D/1, F/1, H/2",
             "ICERSR, A/1, B/3, C/2, D/1, F/1, H/2", "GZ3ECG, A/2, B/3, C/2, D/3, F/1, H/1", "RDAI40, A/3, B/1, C/1, D/2, F/1, H/3",
             "FWFRIF, A/6, B/3, C/2, D/1, F/2, H/1", "B6GU5T, A/1, B/3, C/2, D/1, F/1, H/1", "RM70C6, A/3, B/3, C/1, D/3, F/1, H/3",
             "ACILJP, A/3, B/1, C/2, D/3, F/2, H/2", "C2YCDP, A/4, B/3, C/1, D/2, F/1, H/3", "HF3LBG, A/2, B/3, C/2, D/3, F/1, H/2",
             "JNJWRG, A/4, B/1, C/2, D/2, F/2, H/3", "MJL56Q, A/2, B/1, C/1, D/2, F/1, H/2", "IDU27K, A/4, B/3, C/1, D/1, F/2, H/3",
             "DEXFD3, A/6, B/2, C/1, D/3, F/2, H/1", "AMU435, A/3, B/1, C/2, D/1, F/1, H/1", "B6TTW7, A/5, B/3, C/2, D/1, F/2, H/2",
             "TSZDUX, A/3, B/3, C/2, D/1, F/1, H/3", "SBZBXR, A/2, B/1, C/1, D/2, F/1, H/2", "ID8G0I, A/1, B/2, C/2, D/3, F/1, H/1",
             "SIUCCP, A/1, B/1, C/2, D/2, F/1, H/3", "KQNFM2, A/2, B/2, C/1, D/3, F/2, H/2", "IAKJFF, A/6, B/3, C/1, D/1, F/1, H/2",
             "PVHWCI, A/3, B/2, C/2, D/2, F/1, H/3", "J04HYX, A/5, B/1, C/1, D/3, F/2, H/2", "HBGN4T, A/1, B/2, C/2, D/3, F/2, H/3",
             "IXEQ43, A/5, B/3, C/2, D/2, F/2, H/1", "Z39J23, A/1, B/3, C/2, D/2, F/1, H/3", "AL29V7, A/4, B/3, C/2, D/2, F/1, H/3",
             "RR0RMG, A/3, B/3, C/1, D/3, F/1, H/3", "Y5G10Z, A/4, B/1, C/1, D/2, F/1, H/2", "TN00RU, A/5, B/3, C/2, D/1, F/1, H/1",
             "LPPN7N, A/3, B/1, C/2, D/3, F/2, H/1", "ASE3TM, A/6, B/3, C/1, D/3, F/1, H/1", "D1OJAN, A/1, B/2, C/2, D/1, F/1, H/2",
             "MJJIYQ, A/2, B/2, C/2, D/1, F/1, H/3", "C14ZVE, A/1, B/3, C/1, D/2, F/2, H/2", "F64BNR, A/3, B/3, C/1, D/1, F/2, H/1",
             "SMSZ00, A/2, B/2, C/1, D/2, F/1, H/3", "FILSPA, A/6, B/1, C/2, D/2, F/2, H/1", "ENPCTE, A/4, B/3, C/2, D/1, F/1, H/3",
             "LNK7TK, A/5, B/3, C/1, D/1, F/1, H/1", "H6YS81, A/3, B/2, C/2, D/3, F/2, H/3", "R4JGSN, A/5, B/2, C/2, D/1, F/2, H/3",
             "H0IZYC, A/2, B/3, C/2, D/1, F/1, H/2", "X6D0A6, A/1, B/1, C/1, D/3, F/1, H/3", "GUDGLL, A/2, B/1, C/2, D/1, F/1, H/3",
             "K9P7QZ, A/2, B/3, C/2, D/1, F/2, H/1", "HZIBYS, A/2, B/1, C/1, D/1, F/2, H/1", "FQ7VYO, A/6, B/3, C/2, D/2, F/1, H/3",
             "L3OFLJ, A/1, B/3, C/2, D/3, F/1, H/1", "C3WJNE, A/3, B/3, C/1, D/2, F/1, H/3", "AJ6M9X, A/2, B/3, C/2, D/2, F/2, H/2",
             "JF8XOC, A/2, B/1, C/1, D/3, F/1, H/2", "JFIRBJ, A/1, B/3, C/1, D/1, F/2, H/3", "T492YN, A/6, B/2, C/1, D/2, F/1, H/1",
             "MK67P7, A/5, B/2, C/2, D/3, F/1, H/2", "VP3IGZ, A/4, B/2, C/2, D/1, F/1, H/3", "WD0661, A/4, B/1, C/1, D/2, F/2, H/1",
             "D4GH8M, A/4, B/1, C/2, D/3, F/2, H/2", "TGA1DC, A/1, B/1, C/2, D/2, F/1, H/3", "SELV7B, A/2, B/3, C/2, D/1, F/2, H/3",
             "EK7R2G, A/6, B/2, C/1, D/3, F/2, H/1", "Q8NYZW, A/3, B/1, C/1, D/1, F/2, H/3", "FAZMVB, A/3, B/3, C/1, D/2, F/2, H/2",
             "CF5Q1Y, A/2, B/2, C/2, D/2, F/1, H/2", "PCD3OW, A/1, B/3, C/1, D/1, F/2, H/3", "EJJB7G, A/6, B/2, C/1, D/1, F/2, H/3",
             "IT8A0D, A/2, B/3, C/2, D/3, F/1, H/2", "AELE02, A/5, B/1, C/1, D/3, F/2, H/1", "XCPAFD, A/5, B/3, C/2, D/1, F/1, H/1",
             "OIHSY3, A/3, B/1, C/1, D/1, F/2, H/2", "H8HVZX, A/3, B/2, C/2, D/2, F/2, H/3", "ARW5KJ, A/3, B/1, C/1, D/1, F/1, H/1",
             "AZRC41, A/3, B/2, C/2, D/2, F/1, H/3", "B5J2XQ, A/5, B/2, C/1, D/2, F/2, H/3", "CQKKH4, A/4, B/3, C/2, D/2, F/1, H/3",
             "BFTME9, A/1, B/2, C/1, D/3, F/2, H/1"]

TASK_DATA = {'A_urls': {1: 'https://nipg12.inf.elte.hu/~vavsaai@nipg.lab/annbsc21_p1_hw2/qsar_fish_toxicity.csv',
                             2: 'https://nipg12.inf.elte.hu/~vavsaai@nipg.lab/annbsc21_p1_hw2/student-mat_prep.csv',
                             3: 'https://nipg12.inf.elte.hu/~vavsaai@nipg.lab/annbsc21_p1_hw2/student-por_prep.csv',
                             4: 'https://nipg12.inf.elte.hu/~vavsaai@nipg.lab/annbsc21_p1_hw2/student-mat_nograde_prep.csv',
                             5: 'https://nipg12.inf.elte.hu/~vavsaai@nipg.lab/annbsc21_p1_hw2/student-por_nograde_prep.csv',
                             6: 'https://nipg12.inf.elte.hu/~vavsaai@nipg.lab/annbsc21_p1_hw2/Concrete_Data.csv'},
             'A_shapes': {1:(908, 6), 2:(395, 32), 3:(649, 32), 4:(395, 30), 5:(649, 30), 6:(1030, 8)},
             'A_names': {1:"fish-toxicity", 2:"SP-math", 3:"SP-portugal", 4:"SP-math-nograde",\
                         5:"SP-portugal-nograde", 6:"CCS"},
             'B': {1: 0.7, 2: 0.6, 3: 0.5},
             'C': {1: 'meanstd', 2:'minmax'},
             'D': {1: 'd50relu_dp03', 2: 'd20relu_dp02_d20relu_dp02_d10relu_dp02', 3: 'd50tanh_d30relu'},
             'F': {1: 3, 2: 4},
             'H': {1: 'd50relu_dp03', 2: 'd20relu_dp02_d20relu_dp02_d10relu_dp02', 3: 'd50tanh_d30relu'}}


class Tester:

    '''
    Member fields:

        TESTS_DICT: dict{test_name - str: test function - Callable}

        neptun_code: str; letter characters converted to capital
        student_tasks: dict{task_name - str: task_id - int}
        dataset_content: None OR str; dataset loaded into a string

        # STORED DATA FROM PREVIOUS TESTS

        features: ndarray; Set in __test_dataset_shape()
        labels: ndarray; Set in __test_dataset_shape()

        # RESULT OF LASTEST PREVIOUS TEST RUNS
        test_results: dict{test_name - str: success - bool}

    '''

    def __init__(self, neptun_code, debug_mode=False):
        '''
        Load assigned tasks for each student (by neptun_code) or specify exact task IDs while passing debug_mode=True.
        Paramters:
            neptun_code: str 
                         OR dict(n_tasks){str: int} of int; if 'debug_mode' is True; the exact tasks given
            debug_mode: bool
        '''
        
        self.TESTS_DICT = {'dataset_shape': self.__test_dataset_shape,
              'dataset_split': self.__test_dataset_split,
              'dataset_rescale': self.__test_dataset_rescale,
              'reg_model_architecture': self.__test_reg_model_architecture,
              'reg_model_learning': self.__test_reg_model_learning,
              'cl_dataset': self.__test_cl_dataset,
              'cl_onehot': self.__test_cl_onehot,
              'cl_model_architecture': self.__test_cl_model_architecture,
              'cl_model_learning': self.__test_cl_model_learning}

        self.dataset_content = None
        self.features = None
        self.labels = None
        self.test_results = {k: False for k in self.TESTS_DICT.keys()}

        if debug_mode:
            assert type(neptun_code) == dict
            assert len(neptun_code) == N_TASKS
            self.student_tasks = neptun_code
        else:
            self.neptun_code = neptun_code.strip().upper()
            self.__get_student_tasks()


    def get_dataset_content(self):
        '''
        Sets self.dataset_content.
        Returns:
            dataset_content: str
        '''
        if self.dataset_content is None:
            url = TASK_DATA['A_urls'][self.student_tasks['A']]
            ftpstream = urllib.request.urlopen(url)
            self.dataset_content = ftpstream.read().decode('utf-8')
        return self.dataset_content

    def test(self, test_name, *args):
        '''
        Parameters:
            test_name: str
            *args: varargs; the arguments for the selected test
        '''
        if test_name not in self.TESTS_DICT:
            assert False, "Tester error: Invalid test name: " + str(test_name)

        self.test_results[test_name] = False
        test_func = self.TESTS_DICT[test_name]
        test_func(*args)
        self.test_results[test_name] = True    # only executed if no assert happened during test

    def print_all_tests_successful(self):
        if all(list(self.test_results.values())):
            print("Tester: All tests were successful.")

    # PRIVATE

    def __get_student_tasks(self):
        '''
        Sets self.student_tasks.
        '''
        for entry in TASK_LIST:
            words = [word.strip() for word in entry.split(',')]
            assert len(words) == N_TASKS+1, "Tester error: Invalid task entry length: " + str(words)
            neptun_code = words[0]
            assert len(neptun_code) == 6, "Tester error: Invalid neptun code format: " + str(neptun_code)
            if neptun_code.upper() != self.neptun_code:
                continue
            # neptun code found
            assert all([len(word) == 3 for word in words[1:]]), "Tester error: Invalid task format: " + str(words)
            self.student_tasks = {word[0]: int(word[-1]) for word in words[1:]}
            return
        # neptun code not found
        assert False, "Tester: Neptun code was not found in the student list: " + str(self.neptun_code)

    # TESTS

    def __test_dataset_shape(self, *args):
        '''
        Expected parameters:
            features: ndarray(n_samples, n_features) of float32
            labels: ndarray(n_samples,) of float32
        '''
        assert len(args) == 2, "Tester error: __test_dataset_shape() expects 2 parameters: features, labels. "
        features, labels = args

        # test features, labels array type & shape
        assert type(features) == np.ndarray and type(labels) == np.ndarray, "Tester: features and labels both must be numpy arrays."
        assert features.dtype == np.float32 and labels.dtype == np.float32, "Tester: features and labels both must have float32 dtype."
        expected_shape = TASK_DATA['A_shapes'][self.student_tasks['A']]
        assert features.shape == expected_shape, "Tester: features array shape should be " + str(expected_shape)
        assert labels.shape == features.shape[:1], "Tester: labels array must have (n_samples,) shape."

        # storing data for later use
        self.features = np.copy(features)
        self.labels = np.copy(labels)

        print("Tester: Dataset preparation OK")

    def __test_dataset_split(self, *args):
        '''
        Expected parameters:
            x_unnorm_train, x_unnorm_val, x_unnorm_test: ndarray(n_split_samples, n_features) of float32
            y_train, y_val, y_test: ndarray(n_split_samples,) of float32
        '''
        assert len(args) == 6, "Tester error: __test_dataset_split() expects 6 parameters: " +\
                                    "x_unnorm_train, x_unnorm_val, x_unnorm_test, y_train, y_val, y_test. "
        assert self.features is not None and self.labels is not None, "Tester error: Run tester for task 'A' first."

        # test split array types, shapes
        assert all([type(a) == np.ndarray for a in args]), "Tester: all split arrays must be numpy arrays."
        x_unnorm_train, x_unnorm_val, x_unnorm_test, y_train, y_val, y_test = args
        expected_shape = TASK_DATA['A_shapes'][self.student_tasks['A']]
        assert all(a.shape[1] == expected_shape[1] for a in [x_unnorm_train, x_unnorm_val, x_unnorm_test]), "Tester: split feature array " +\
                                                    "shape should be (?," + str(expected_shape[1]) + ")."
        assert all([ax.shape[0] == ay.shape[0] for ax, ay in zip([x_unnorm_train, x_unnorm_val, x_unnorm_test], [y_train, y_val, y_test])]), \
                                                    "Tester: shape of x and y splits must match along axis#0. "
        x_concat = np.concatenate([x_unnorm_train, x_unnorm_val, x_unnorm_test], axis=0)
        y_concat = np.concatenate([y_train, y_val, y_test], axis=0)
        assert x_concat.shape == expected_shape, "Tester: concatenated x splits do not match original features array shape."
        assert y_concat.shape == expected_shape[:1], "Tester: concatenated y splits do not match original labels array shape."
        split_train_ratio = TASK_DATA['B'][self.student_tasks['B']]

        n_train_samples = int(expected_shape[0]*split_train_ratio)
        n_valtest_samples = int(expected_shape[0]*(1.-split_train_ratio)*0.5)
        assert abs(x_unnorm_train.shape[0] - n_train_samples) <= 2, "Tester: Train split ratio must be " + str(split_train_ratio)
        assert abs(x_unnorm_val.shape[0] - n_valtest_samples) <= 2, "Tester: Vaidation split ratio must be " + str(n_valtest_samples)
        assert abs(x_unnorm_test.shape[0] - n_valtest_samples) <= 2, "Tester: Test split ratio must be " + str(n_valtest_samples)

        # test if shuffled
        assert np.any(x_concat != self.features), "Tester: features and labels array must be shuffled before split. "

        # test if shuffle was consistent
        u_pairs_before = np.unique(np.concatenate([self.features, self.labels[:,None]], axis=-1), axis=0)
        u_pairs_after = np.unique(np.concatenate([x_concat, y_concat[:,None]], axis=-1), axis=0)
        assert np.array_equal(u_pairs_before, u_pairs_after), "Tester: the features and labels array were shuffled with different permutations."

        print("Tester: Dataset split OK")

    def __test_dataset_rescale(self, *args):
        '''
        Expected parameters:
            x_train, x_val, x_test: ndarray(n_split_samples, n_features) of float32
        '''
        assert len(args) == 3, "Tester error: __test_dataset_rescale() expects 3 parameters: " +\
                                    "x_train, x_val, x_test. "
        assert self.features is not None and self.labels is not None, "Tester error: Run tester for task 'A' first."
        
        rescale_method_name = TASK_DATA['C'][self.student_tasks['C']]
        x_train, x_val, x_test = args

        # test shape of splits
        assert all([type(a) == np.ndarray for a in args]), "Tester: all split arrays must be numpy arrays."
        expected_shape = TASK_DATA['A_shapes'][self.student_tasks['A']]
        assert all(a.shape[1] == expected_shape[1] for a in [x_train, x_val, x_test]), "Tester: split feature array " +\
                                                    "shape should be (?," + str(expected_shape[1]) + ")."
        x_concat = np.concatenate([x_train, x_val, x_test], axis=0)
        assert x_concat.shape == expected_shape, "Tester: concatenated x splits do not match original features array shape."

        if rescale_method_name == 'meanstd':
            # test mean, std of training set
            assert np.all(np.fabs(np.mean(x_train, axis=0)) < 0.0001), "Tester: Mean-std normalized x_train mean should be around zero."
            assert np.all(np.fabs(np.std(x_train, axis=0) - 1.) < 0.01), "Tester: Mean-std normalized x_train std should be around one."

            # test mean, std of validation & test set: should not be very close to 0 mean, 1 std as we used tr set statistics for normalization
            assert np.any(np.fabs(np.mean(x_val, axis=0)) > 0.01), "Tester: x_val should be normalized with mean-std computed from x_train."
            assert np.any(np.fabs(np.mean(x_val, axis=0) - 1.) > 0.01), "Tester: x_val should be normalized with mean-std computed from x_train."
            assert np.any(np.fabs(np.mean(x_test, axis=0)) > 0.01), "Tester: x_test should be normalized with mean-std computed from x_train."
            assert np.any(np.fabs(np.mean(x_test, axis=0) - 1.) > 0.01), "Tester: x_test should be normalized with mean-std computed from x_train."

        elif rescale_method_name == 'minmax':
            # test mean, std of training set
            assert np.all(np.fabs(np.amax(x_train, axis=0) - 1.) < 0.01), "Tester: Min-max normalized x_train max should be around one."
            assert np.all(np.fabs(np.amin(x_train, axis=0)) < 0.01), "Tester: Min-max normalized x_train min should be around zero."

            # test min-max of validation & test set: should not be very close to 0 min, 1 max as we used tr set statistics for normalization
            # DISABLED
            #assert np.any(np.fabs(np.amax(x_val, axis=0) - 1.) > 0.01), "Tester: x_val should be normalized with min-max computed from x_train."
            #assert np.any(np.fabs(np.amin(x_val, axis=0)) > 0.01), "Tester: x_val should be normalized with min-max computed from x_train."
            #assert np.any(np.fabs(np.amax(x_test, axis=0) - 1.) > 0.01), "Tester: x_test should be normalized with min-max computed from x_train."
            #assert np.any(np.fabs(np.amin(x_test, axis=0)) > 0.01), "Tester: x_test should be normalized with min-max computed from x_train."

        else:
            assert False, "Tester error: __test_dataset_rescale() invalid rescale_method_name: " + str(rescale_method_name)

        print("Tester: Feature rescale OK")

    def __test_reg_model_architecture(self, *args):
        '''
        Expected parameters:
            reg_model: keras Model
        '''
        assert len(args) == 1, "Tester error: __test_reg_model_architecture() expects 1 parameter: reg_model "
        
        model_architecture_name = TASK_DATA['D'][self.student_tasks['D']]
        reg_model, = args

        assert isinstance(reg_model, tf.keras.models.Sequential), "Tester: In this assignment you must use the Sequential model (tf.keras version)"
        assert reg_model._is_compiled, "Tester: The model must be compiled."
        assert 'mse' in str(reg_model.loss) or 'mean_squared_error' in str(reg_model.loss), "Tester: Incorrect regression loss."

        if model_architecture_name == 'd50relu_dp03':
            assert len(reg_model.layers) == 3, "Tester: In your task, the network should have exactly 3 layers (dropout included)."
            assert isinstance(reg_model.layers[0], tf.keras.layers.Dense), "Tester: Layer#0 should be a Dense layer."
            assert isinstance(reg_model.layers[1], tf.keras.layers.Dropout), "Tester: Layer#1 should be a Dropout layer."
            assert isinstance(reg_model.layers[2], tf.keras.layers.Dense), "Tester: Layer#2 should be a Dense layer."
            assert reg_model.layers[0].units == 50, "Tester: Layer#0 should have 50 neurons."
            assert reg_model.layers[2].units == 1,  "Tester: Incorrect number of neurons in Layer#2 (should be last layer)."
            assert reg_model.layers[0].activation == tf.keras.activations.relu, "Tester: Layer#0 should have a ReLU activation function."
            assert reg_model.layers[2].activation == tf.keras.activations.linear, "Tester: Incorrect activation function in Layer#2 (should be last layer)."
            assert abs(reg_model.layers[1].rate - .3) < 0.001, "Tester: Dropout layer should have a dropout rate of 0.3."

        elif model_architecture_name == 'd20relu_dp02_d20relu_dp02_d10relu_dp02':
            assert len(reg_model.layers) == 7, "Tester: In your task, the network should have exactly 7 layers (dropout included)."
            assert isinstance(reg_model.layers[0], tf.keras.layers.Dense), "Tester: Layer#0 should be a Dense layer."
            assert isinstance(reg_model.layers[1], tf.keras.layers.Dropout), "Tester: Layer#1 should be a Dropout layer."
            assert isinstance(reg_model.layers[2], tf.keras.layers.Dense), "Tester: Layer#2 should be a Dense layer."
            assert isinstance(reg_model.layers[3], tf.keras.layers.Dropout), "Tester: Layer#3 should be a Dropout layer."
            assert isinstance(reg_model.layers[4], tf.keras.layers.Dense), "Tester: Layer#4 should be a Dense layer."
            assert isinstance(reg_model.layers[5], tf.keras.layers.Dropout), "Tester: Layer#5 should be a Dropout layer."
            assert isinstance(reg_model.layers[6], tf.keras.layers.Dense), "Tester: Layer#6 should be a Dense layer."
            assert reg_model.layers[0].units == 20, "Tester: Layer#0 should have 20 neurons."
            assert reg_model.layers[2].units == 20, "Tester: Layer#2 should have 20 neurons."
            assert reg_model.layers[4].units == 10, "Tester: Layer#4 should have 10 neurons."
            assert reg_model.layers[6].units == 1, "Tester: Incorrect number of neurons in Layer#6 (should be last layer)."
            assert reg_model.layers[0].activation == tf.keras.activations.relu, "Tester: Layer#0 should have a ReLU activation function."
            assert reg_model.layers[2].activation == tf.keras.activations.relu, "Tester: Layer#2 should have a ReLU activation function."
            assert reg_model.layers[4].activation == tf.keras.activations.relu, "Tester: Layer#4 should have a ReLU activation function."
            assert reg_model.layers[6].activation == tf.keras.activations.linear, "Tester: Incorrect activation function in Layer#6 (should be last layer)."
            assert abs(reg_model.layers[1].rate - .2) < 0.001, "Tester: Dropout layer should have a dropout rate of 0.2."
            assert abs(reg_model.layers[3].rate - .2) < 0.001, "Tester: Dropout layer should have a dropout rate of 0.2."
            assert abs(reg_model.layers[5].rate - .2) < 0.001, "Tester: Dropout layer should have a dropout rate of 0.2."

        elif model_architecture_name == 'd50tanh_d30relu':
            assert len(reg_model.layers) == 3, "Tester: In your task, the network should have exactly 3 layers."
            assert isinstance(reg_model.layers[0], tf.keras.layers.Dense), "Tester: Layer#0 should be a Dense layer."
            assert isinstance(reg_model.layers[1], tf.keras.layers.Dense), "Tester: Layer#1 should be a Dense layer."
            assert isinstance(reg_model.layers[2], tf.keras.layers.Dense), "Tester: Layer#2 should be a Dense layer."
            assert reg_model.layers[0].units == 50, "Tester: Layer#0 should have 50 neurons."
            assert reg_model.layers[1].units == 30, "Tester: Layer#1 should have 30 neurons."
            assert reg_model.layers[2].units == 1, "Tester: Incorrect number of neurons in Layer#2 (should be last layer)."
            assert reg_model.layers[0].activation == tf.keras.activations.tanh, "Tester: Layer#0 should have a tanh activation function."
            assert reg_model.layers[1].activation == tf.keras.activations.relu, "Tester: Layer#1 should have a ReLU activation function."
            assert reg_model.layers[2].activation == tf.keras.activations.linear, "Tester: Incorrect activation function in Layer#2 (should be last layer)."

        else:
            assert False, "Tester error: __test_reg_model_architecture() invalid model_architecture_name: " \
                                                                        + str(model_architecture_name)

        print("Tester: Regression model architecture OK")


    def __test_reg_model_learning(self, *args):
        '''
        Expected parameters:
            test_mse, test_mae: float
        '''
        assert len(args) == 2, "Tester error: __test_reg_model_learning() expects 2 parameters: test_mse, test_mae "
        
        '''
        test MSEs (with D/1,2,3.):
            'WQ-white': 0.56, 0.6, 0.58
            'SP-math': 4.62, 6, 2.92, 4.58
            'SP-portugal': 2.9, 5.6, 1.66
            'SP-math-nograde': 14, 18, 16
            'SP-portugal-nograde': 8.6, 6.9, 7.5
            'CCS': 54.8, 44, 35
        test MAEs (with D/1,2,3.):
            'WQ-white': 0.58, 0.59, 0.58
            'SP-math': 1.58, 1.92, 1.12, 1.63
            'SP-portugal': 1.05, 1.26, 0.95
            'SP-math-nograde': 3, 3.5, 3.2
            'SP-portugal-nograde': 2.4, 2, 2.2
            'CCS': 5.5, 4.8, 4.2
        '''

        dataset_name = TASK_DATA['A_names'][self.student_tasks['A']]
        test_mse, test_mae = args
        min_test_mses = {'fish-toxicity': 0.3,
                         'SP-math': 1.5,
                         'SP-portugal': 0.8,
                         'SP-math-nograde': 6.,
                         'SP-portugal-nograde': 3.,
                         'CCS': 15.}
        max_test_mses = {'fish-toxicity': 1.5,
                         'SP-math': 10.,
                         'SP-portugal': 8.,
                         'SP-math-nograde': 25.,
                         'SP-portugal-nograde': 12.,
                         'CCS': 90.}
        min_test_maes = {'fish-toxicity': 0.6,
                         'SP-math': 0.5,
                         'SP-portugal': 0.5,
                         'SP-math-nograde': 1.5,
                         'SP-portugal-nograde': 1.,
                         'CCS': 2.}
        max_test_maes = {'fish-toxicity': 1,
                         'SP-math': 3.,
                         'SP-portugal': 2.,
                         'SP-math-nograde': 6.,
                         'SP-portugal-nograde': 4.,
                         'CCS': 9.}
        assert dataset_name in min_test_mses.keys(), "Tester error: __test_reg_model_learning() invalid dataset_name: "\
                                                                 + str(dataset_name)
        assert test_mse > min_test_mses[dataset_name] and test_mse < max_test_mses[dataset_name], "Tester: A well-trained model "+\
                    "should produce an MSE between " + str(min_test_mses[dataset_name]) + " and " + str(max_test_mses[dataset_name])
        assert test_mae > min_test_maes[dataset_name] and test_mae < max_test_maes[dataset_name], "Tester: A well-trained model "+\
                    "should produce a MAE between " + str(min_test_maes[dataset_name]) + " and " + str(max_test_maes[dataset_name])

        print("Tester: Regression model learning OK")


    def __test_cl_dataset(self, *args):
        '''
        Expected parameters:
            y_cat_train, y_cat_val, y_cat_test: ndarray(n_split_samples) of int
        '''
        assert len(args) == 3, "Tester error: __test_cl_dataset() expects 3 parameters: y_cat_train, y_cat_val, y_cat_test "

        num_classes = TASK_DATA['F'][self.student_tasks['F']]

        # test shapes
        y_cat_train, y_cat_val, y_cat_test = args
        y_cat_concat = np.concatenate([y_cat_train, y_cat_val, y_cat_test], axis=0)
        assert y_cat_concat.shape == self.labels.shape, "Tester: concatenated y_cat splits do not match original label array shape."
        assert len(np.unique(y_cat_concat)) == num_classes, "Tester: there should be {:d} categories".format(num_classes)
        # test contents: label distribution should be _approximately_ equal
        for y_cat_split, y_cat_split_name in zip([y_cat_train, y_cat_val, y_cat_test], ["y_cat_train", "y_cat_val", "y_cat_test"]):
            u_y_cat_split, c_y_cat_split = np.unique(y_cat_split, return_counts=True)
            assert np.array_equal(u_y_cat_split, np.arange(num_classes)), "Tester: not all label categories are present in " + str(y_cat_split_name)
            min_c_y_cat_split = np.amin(c_y_cat_split)
            max_c_y_cat_split = np.amax(c_y_cat_split)
            assert min_c_y_cat_split*4. > max_c_y_cat_split, "Tester: number of samples from each label category"+\
                            " should be _approximately_ equal; this is not the case in " + y_cat_split_name

        print("Tester: Classification dataset creation OK")

    def __test_cl_onehot(self, *args):
        '''
        Expected parameters:
            y_onehot_train, y_onehot_val, y_onehot_test: ndarray(n_split_samples, n_categories) of int
        '''
        assert len(args) == 3, "Tester error: __test_cl_onehot() expects 3 parameters: "+\
                                            "y_onehot_train, y_onehot_val, y_onehot_test "

        num_classes = TASK_DATA['F'][self.student_tasks['F']]

        y_onehot_train, y_onehot_val, y_onehot_test = args
        y_onehot_concat = np.concatenate([y_onehot_train, y_onehot_val, y_onehot_test], axis=0)
        assert y_onehot_concat.shape[:1] == self.labels.shape, "Tester: concatenated y_onehot splits do not match original label array shape."
        assert y_onehot_concat.shape[1:] == (num_classes,), "Tester: y_onehot splits should have a shape: (n_samples, {:d})".format(num_classes)

        assert np.all(np.sum(y_onehot_concat, axis=1) == 1.), "Tester: y_onehot vectors must have a sum of 1"
        assert np.all(np.count_nonzero(y_onehot_concat, axis=1) == 1), "Tester: y_onehot vectors must have a single non-zero item"
        
        for y_onehot_split, y_onehot_split_name in zip([y_onehot_train, y_onehot_val, y_onehot_test],\
                                                       ["y_onehot_train", "y_onehot_val", "y_onehot_test"]):
            y_cat_split = np.argmax(y_onehot_split, axis=1)
            u_y_cat_split, c_y_cat_split = np.unique(y_cat_split, return_counts=True)
            assert np.array_equal(u_y_cat_split, np.arange(num_classes)), "Tester: not all label categories are present in " + str(y_onehot_split_name)
            min_c_y_cat_split = np.amin(c_y_cat_split)
            max_c_y_cat_split = np.amax(c_y_cat_split)
            assert min_c_y_cat_split*4. > max_c_y_cat_split, "Tester: number of samples from each label category"+\
                            " should be _approximately_ equal; this is not the case in " + y_onehot_split_name

        print("Tester: One-hot conversion OK")

    def __test_cl_model_architecture(self, *args):
        '''
        Expected parameters:
            cl_model: keras.models.Sequential() instance
        '''
        assert len(args) == 1, "Tester error: __test_cl_model_architecture() expects 1 parameter: cl_model "


        model_architecture_name = TASK_DATA['H'][self.student_tasks['H']]
        num_classes = TASK_DATA['F'][self.student_tasks['F']]
        cl_model, = args


        assert isinstance(cl_model, tf.keras.models.Sequential), "Tester: In this assignment you must use the Sequential model (tf.keras version)"
        assert cl_model._is_compiled, "Tester: The model must be compiled."
        assert 'categorical_crossentropy' in str(cl_model.loss), "Tester: Incorrect multi-class classification loss."

        if model_architecture_name == 'd50relu_dp03':
            assert len(cl_model.layers) == 3, "Tester: In your task, the network should have exactly 3 layers."
            assert isinstance(cl_model.layers[0], tf.keras.layers.Dense), "Tester: Layer#0 should be a Dense layer."
            assert isinstance(cl_model.layers[1], tf.keras.layers.Dropout), "Tester: Layer#1 should be a Dropout layer."
            assert isinstance(cl_model.layers[2], tf.keras.layers.Dense), "Tester: Layer#2 should be a Dense layer."
            assert cl_model.layers[0].units == 50, "Tester: Layer#0 should have 50 neurons."
            assert cl_model.layers[2].units == num_classes, "Tester: Incorrect number of neurons in Layer#2 (should be last layer)."
            assert cl_model.layers[0].activation == tf.keras.activations.relu, "Tester: Layer#0 should have a ReLU activation function."
            assert cl_model.layers[2].activation == tf.keras.activations.softmax, "Tester: Incorrect activation function in Layer#2 (should be last layer)."
            assert abs(cl_model.layers[1].rate - .3) < 0.001, "Tester: Dropout layer should have a dropout rate of 0.3."

        elif model_architecture_name == 'd20relu_dp02_d20relu_dp02_d10relu_dp02':
            assert len(cl_model.layers) == 7, "Tester: In your task, the network should have exactly 7 layers."
            assert isinstance(cl_model.layers[0], tf.keras.layers.Dense), "Tester: Layer#0 should be a Dense layer."
            assert isinstance(cl_model.layers[1], tf.keras.layers.Dropout), "Tester: Layer#1 should be a Dropout layer."
            assert isinstance(cl_model.layers[2], tf.keras.layers.Dense), "Tester: Layer#2 should be a Dense layer."
            assert isinstance(cl_model.layers[3], tf.keras.layers.Dropout), "Tester: Layer#3 should be a Dropout layer."
            assert isinstance(cl_model.layers[4], tf.keras.layers.Dense), "Tester: Layer#4 should be a Dense layer."
            assert isinstance(cl_model.layers[5], tf.keras.layers.Dropout), "Tester: Layer#5 should be a Dropout layer."
            assert isinstance(cl_model.layers[6], tf.keras.layers.Dense), "Tester: Layer#6 should be a Dense layer."
            assert cl_model.layers[0].units == 20, "Tester: Layer#0 should have 20 neurons."
            assert cl_model.layers[2].units == 20, "Tester: Layer#2 should have 20 neurons."
            assert cl_model.layers[4].units == 10, "Tester: Layer#4 should have 10 neurons."
            assert cl_model.layers[6].units == num_classes, "Tester: Incorrect number of neurons in Layer#6 (should be last layer)."
            assert cl_model.layers[0].activation == tf.keras.activations.relu, "Tester: Layer#0 should have a ReLU activation function."
            assert cl_model.layers[2].activation == tf.keras.activations.relu, "Tester: Layer#2 should have a ReLU activation function."
            assert cl_model.layers[4].activation == tf.keras.activations.relu, "Tester: Layer#4 should have a ReLU activation function."
            assert cl_model.layers[6].activation == tf.keras.activations.softmax, "Tester: Incorrect activation function in Layer#6 (should be last layer)."
            assert abs(cl_model.layers[1].rate - .2) < 0.001, "Tester: Dropout layer should have a dropout rate of 0.2."
            assert abs(cl_model.layers[3].rate - .2) < 0.001, "Tester: Dropout layer should have a dropout rate of 0.2."
            assert abs(cl_model.layers[5].rate - .2) < 0.001, "Tester: Dropout layer should have a dropout rate of 0.2."

        elif model_architecture_name == 'd50tanh_d30relu':
            assert len(cl_model.layers) == 3, "Tester: In your task, the network should have exactly 3 layers."
            assert isinstance(cl_model.layers[0], tf.keras.layers.Dense), "Tester: Layer#0 should be a Dense layer."
            assert isinstance(cl_model.layers[1], tf.keras.layers.Dense), "Tester: Layer#1 should be a Dense layer."
            assert isinstance(cl_model.layers[2], tf.keras.layers.Dense), "Tester: Layer#2 should be a Dense layer."
            assert cl_model.layers[0].units == 50, "Tester: Layer#0 should have 50 neurons."
            assert cl_model.layers[1].units == 30, "Tester: Layer#1 should have 30 neurons."
            assert cl_model.layers[2].units == num_classes, "Tester: Incorrect number of neurons in Layer#2 (should be last layer)."
            assert cl_model.layers[0].activation == tf.keras.activations.tanh, "Tester: Layer#0 should have a tanh activation function."
            assert cl_model.layers[1].activation == tf.keras.activations.relu, "Tester: Layer#1 should have a ReLU activation function."
            assert cl_model.layers[2].activation == tf.keras.activations.softmax, "Tester: Incorrect activation function in Layer#2 (should be last layer)."

        else:
            assert False, "Tester error: __test_cl_model_architecture() invalid model_architecture_name: " \
                                                                        + str(model_architecture_name)

        print("Tester: Classification model architecture OK")


    def __test_cl_model_learning(self, *args):
        '''
        Expected parameters:
            test_ce, test_acc: float
        '''
        assert len(args) == 3, "Tester error: __test_cl_model_learning() expects 3 parameters:" +\
                                             " test_ce, test_acc, test_f1"

        dataset_name = TASK_DATA['A_names'][self.student_tasks['A']]
        num_classes = TASK_DATA['F'][self.student_tasks['F']]

        test_ce, test_acc, test_f1 = args
        min_test_ces = {'fish-toxicity': {3: 0.6, 4: 0.8},
                         'SP-math': {3: 0.3, 4: 0.3},
                         'SP-portugal': {3: 0.3, 4: 0.3},
                         'SP-math-nograde': {3: 0.8, 4: 0.8},
                         'SP-portugal-nograde': {3: 0.6, 4: 0.6},
                         'CCS': {3: 0.2, 4: 0.2}}
        max_test_ces = {'fish-toxicity': {3: 1.1, 4: 1.25},
                         'SP-math': {3: 0.9, 4: 1.25},
                         'SP-portugal': {3: 0.8, 4: 1.2},
                         'SP-math-nograde': {3: 1.4, 4: 1.4},
                         'SP-portugal-nograde': {3: 1.25, 4: 1.3},
                         'CCS': {3: 0.8, 4: 1.15}}
        min_test_accs = {'fish-toxicity': {3: 0.5, 4: 0.4},
                         'SP-math': {3: 0.6, 4: 0.4},
                         'SP-portugal': {3: 0.65, 4: 0.45},
                         'SP-math-nograde': {3: 0.33, 4: 0.25},
                         'SP-portugal-nograde': {3: 0.4, 4: 0.33},
                         'CCS': {3: 0.65, 4: 0.4}}
        max_test_accs = {'fish-toxicity': {3: 0.75, 4: 0.8},
                         'SP-math': {3: 0.9, 4: 0.9},
                         'SP-portugal': {3: 0.9, 4: 0.9},
                         'SP-math-nograde': {3: 0.6, 4: 0.6},
                         'SP-portugal-nograde': {3: 0.75, 4: 0.75},
                         'CCS': {3: 0.95, 4: 0.95}}
        min_test_f1s = {'fish-toxicity': {3: 0.55, 4: 0.4},
                       'SP-math': {3: 0.6, 4: 0.4},
                       'SP-portugal': {3: 0.65, 4: 0.45},
                       'SP-math-nograde': {3: 0.33, 4: 0.25},
                       'SP-portugal-nograde': {3: 0.4, 4: 0.33},
                       'CCS': {3: 0.65, 4: 0.4}}
        max_test_f1s = {'fish-toxicity': {3: 0.75, 4: 0.7},
                       'SP-math': {3: 0.9, 4: 0.9},
                       'SP-portugal': {3: 0.9, 4: 0.9},
                       'SP-math-nograde': {3: 0.6, 4: 0.6},
                       'SP-portugal-nograde': {3: 0.75, 4: 0.75},
                       'CCS': {3: 0.95, 4: 0.95}}

        assert dataset_name in min_test_ces.keys(), "Tester error: __test_cl_model_learning() invalid dataset_name: "\
                                                                 + str(dataset_name)

        min_test_ce, max_test_ce = min_test_ces[dataset_name][num_classes], max_test_ces[dataset_name][num_classes]
        assert test_ce > min_test_ce and test_ce < max_test_ce, "Tester: A well-trained model should produce a CE loss between " \
                                                                + str(min_test_ce) + " and " + str(max_test_ce)

        min_test_acc, max_test_acc = min_test_accs[dataset_name][num_classes], max_test_accs[dataset_name][num_classes]
        assert test_acc > min_test_acc and test_acc < max_test_acc, "Tester: A well-trained model should produce accuracy between " \
                                                                    + str(min_test_acc) + " and " + str(max_test_acc)

        min_test_f1, max_test_f1 = min_test_f1s[dataset_name][num_classes], max_test_f1s[dataset_name][num_classes]
        assert test_f1 > min_test_f1 and test_f1 < max_test_f1, "Tester: A well-trained model should produce an F1-score between " \
                                                                    + str(min_test_f1) + " and " + str(max_test_f1)
        print("Tester: Classification model learning OK")