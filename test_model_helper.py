#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import model_helper
import numpy as np
from config import *

class TestModel(unittest.TestCase):
    def setUp(self):
        self.parameter = {
            'cost_cooperate':   4,
            'cost_support':     2,
            'cost_punish':      2,
            'power_social':     4,
            'punish_size':      8,
        }

    def test_generate_groups(self):
        return_values1 = model_helper.generate_groups()

        # ポイント初期化確認
        expected = np.zeros((NUM_GROUPS, NUM_MEMBERS))
        actual = return_values1[:, :, COL_P]
        self.assertEquals(np.sum(expected == actual), NUM_GROUPS * NUM_MEMBERS)

        expected = np.zeros((NUM_GROUPS, NUM_MEMBERS))
        actual = return_values1[:, :, COL_P]
        self.assertEquals(np.sum(expected == actual), NUM_GROUPS * NUM_MEMBERS)

        # ポイントログ初期化確認
        expected = np.zeros((NUM_GROUPS, NUM_MEMBERS))
        actual = return_values1[:, :, COL_P_LOG]
        self.assertEquals(np.sum(expected == actual), NUM_GROUPS * NUM_MEMBERS)

        # 行動の初期化確認
        expected = np.full((NUM_GROUPS, NUM_MEMBERS), -1)
        actual = return_values1[:, :, COL_AC]
        self.assertEquals(np.sum(expected == actual), NUM_GROUPS * NUM_MEMBERS)

        expected = np.full((NUM_GROUPS, NUM_MEMBERS), -1)
        actual = return_values1[:, :, COL_AS]
        self.assertEquals(np.sum(expected == actual), NUM_GROUPS * NUM_MEMBERS)

        expected = np.full((NUM_GROUPS, NUM_MEMBERS), -1)
        actual = return_values1[:, :, COL_APC]
        self.assertEquals(np.sum(expected == actual), NUM_GROUPS * NUM_MEMBERS)

        expected = np.full((NUM_GROUPS, NUM_MEMBERS), -1)
        actual = return_values1[:, :, COL_APS]
        self.assertEquals(np.sum(expected == actual), NUM_GROUPS * NUM_MEMBERS)

        # 乱数区間確認
        expected_area = [0, 1]
        actual = return_values1[:, :, COL_GC]
        self.assertEquals(np.sum(expected_area[0] <= actual), NUM_GROUPS * NUM_MEMBERS)
        self.assertEquals(np.sum(expected_area[1] >= actual), NUM_GROUPS * NUM_MEMBERS)

        expected_area = [0, 1]
        actual = return_values1[:, :, COL_GS]
        self.assertEquals(np.sum(expected_area[0] <= actual), NUM_GROUPS * NUM_MEMBERS)
        self.assertEquals(np.sum(expected_area[1] >= actual), NUM_GROUPS * NUM_MEMBERS)

        expected_area = [0, 1]
        actual = return_values1[:, :, COL_GPC]
        self.assertEquals(np.sum(expected_area[0] <= actual), NUM_GROUPS * NUM_MEMBERS)
        self.assertEquals(np.sum(expected_area[1] >= actual), NUM_GROUPS * NUM_MEMBERS)

        expected_area = [0, 1]
        actual = return_values1[:, :, COL_GPS]
        self.assertEquals(np.sum(expected_area[0] <= actual), NUM_GROUPS * NUM_MEMBERS)
        self.assertEquals(np.sum(expected_area[1] >= actual), NUM_GROUPS * NUM_MEMBERS)

    def test_set_gene_s_by_pc_count(self):
        step = 100
        num_groups = 4
        num_members = 5
        arg1 = np.zeros((num_groups, num_members, NUM_COLUMN))
        arg2 = np.zeros(num_groups)

        p1arg1 = arg1
        p1arg1[:, :, COL_GC] = np.array(
            [
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1],
            ]
        )
        p1arg2 = arg2
        p1arg2 = np.array(
            [0, 0, 0, 0]
        )
        p1arg3 = step

        expected = p1arg1
        expected[:, :, COL_GS] = np.array(
            [
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1],
                [0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0],
            ]
        )

        actual = model_helper.set_gene_s_by_pc_count(p1arg1, p1arg2, p1arg3)
        self.assertEqual(np.sum(expected == actual), num_groups * num_members * NUM_COLUMN)

        p2arg1 = arg1
        p2arg1[:, :, COL_GC] = np.array(
            [
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1]
            ]
        )
        p2arg2 = arg2
        p2arg2 = np.array(
            [50, 50, 50, 50]
        )
        p2arg3 = step

        expected[:, :, COL_GS] = np.array(
            [
                [0.5, 0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5, 0.5],
            ]
        )

        actual = model_helper.set_gene_s_by_pc_count(p2arg1, p2arg2, p2arg3)
        self.assertEqual(np.sum(expected == actual), num_groups * num_members * NUM_COLUMN)

        p3arg1 = arg1
        p3arg1[:, :, COL_GC] = np.array(
            [
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1],
            ]
        )
        p3arg2 = arg2
        p3arg2 = np.array(
            [100, 100, 100, 100],
        )
        p3arg3 = step

        expected[:, :, COL_GS] = np.array(
            [
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1],
            ]
        )

        actual = model_helper.set_gene_s_by_pc_count(p3arg1, p3arg2, p3arg3)
        self.assertEqual(np.sum(expected == actual), num_groups * num_members * NUM_COLUMN)
    
    def test_do_action(self):
        # 乱数による決定が入るため1と0の場合のみをテストする

        # 0の場合
        arg1 = np.zeros((NUM_GROUPS, NUM_MEMBERS, NUM_COLUMN))
        arg2 = np.zeros((NUM_GROUPS, NUM_COLUMN))
        return_values1, return_values2 = model_helper.do_action(arg1, arg2)
        
        expected = np.zeros((NUM_GROUPS, NUM_MEMBERS))
        
        actual = return_values1[:, :, COL_AC]
        self.assertEquals(np.sum(expected == actual), NUM_GROUPS * NUM_MEMBERS)

        actual = return_values1[:, :, COL_AS]
        self.assertEquals(np.sum(expected == actual), NUM_GROUPS * NUM_MEMBERS)

        expected = np.zeros(NUM_GROUPS)

        actual = return_values2[:, COL_APC]
        self.assertEquals(np.sum(expected == actual), NUM_GROUPS)

        actual = return_values2[:, COL_APS]
        self.assertEquals(np.sum(expected == actual), NUM_GROUPS)

        # 1の場合
        arg1 = np.ones((NUM_GROUPS, NUM_MEMBERS, NUM_COLUMN))
        arg2 = np.ones((NUM_GROUPS, NUM_COLUMN))
        return_values1, return_values2 = model_helper.do_action(arg1, arg2)
        
        expected = np.ones((NUM_GROUPS, NUM_MEMBERS))
        
        actual = return_values1[:, :, COL_AC]
        self.assertEquals(np.sum(expected == actual), NUM_GROUPS * NUM_MEMBERS)

        actual = return_values1[:, :, COL_AS]
        self.assertEquals(np.sum(expected == actual), NUM_GROUPS * NUM_MEMBERS)

        expected = np.ones(NUM_GROUPS)

        actual = return_values2[:, COL_APC]
        self.assertEquals(np.sum(expected == actual), NUM_GROUPS)

        actual = return_values2[:, COL_APS]
        self.assertEquals(np.sum(expected == actual), NUM_GROUPS)

    def test_get_members_gain(self):
        # パターンを描くのが辛いためグループ数と成員数は縮小版でテスト
        num_groups = 4
        num_members = 5
        arg1 = np.zeros((num_groups, num_members, NUM_COLUMN))
        arg2 = np.zeros((num_groups, NUM_COLUMN))

        p1arg1 = arg1
        p1arg1[:, :, COL_AC] = np.array(
            [
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1],
            ]
        )
        p1arg1[:, :, COL_AS] = np.array(
            [
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1],
            ]
        )

        p1arg2 = arg2
        p1arg2[:, COL_APC] = np.array(
            [0, 0, 0, 0]
        )
        p1arg2[:, COL_APS] = np.array(
            [0, 0, 0, 0]
        )

        expected = np.array(
            [
                [6, 6, 6, 6, 6],
                [3.2, 9.2, 9.2, 9.2, 9.2],
                [6.4, 6.4, 12.4, 12.4, 12.4],
                [16, 16, 16, 16, 16],
            ]
        )
        
        actual = model_helper.get_members_gain(p1arg1, p1arg2, self.parameter)
        self.assertEquals(np.sum(expected == actual), num_members*num_groups)

        p2arg1 = arg1
        p2arg1[:, :, COL_AC] = np.array(
            [
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1],
            ]
        )
        p2arg1[:, :, COL_AS] = np.array(
            [
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0],
            ]
        )

        p2arg2 = arg2
        p2arg2[:, COL_APC] = np.array(
            [1, 1, 1, 1]
        )
        p2arg2[:, COL_APS] = np.array(
            [0, 0, 0, 0]
        )

        expected = np.array(
            [
                [4 - self.parameter['punish_size'], 6 - self.parameter['punish_size'], 6 - self.parameter['punish_size'], 6 - self.parameter['punish_size'], 6 - self.parameter['punish_size']],
                [3.2, 7.2 - self.parameter['punish_size'], 9.2 - self.parameter['punish_size'], 9.2 - self.parameter['punish_size'], 9.2 - self.parameter['punish_size']],
                [6.4, 6.4, 10.4 - self.parameter['punish_size'], 10.4 - self.parameter['punish_size'], 10.4 - self.parameter['punish_size']],
                [18, 18, 18, 18, 18],
            ]
        )
        
        actual = model_helper.get_members_gain(p2arg1, p2arg2, self.parameter)
        self.assertEquals(np.sum(expected == actual), num_members*num_groups)

        p3arg1 = arg1
        p3arg1[:, :, COL_AC] = np.array(
            [
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1],
            ]
        )
        p3arg1[:, :, COL_AS] = np.array(
            [
                [1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
            ]
        )

        p3arg2 = arg2
        p3arg2[:, COL_APC] = np.array(
            [0, 0, 0, 0]
        )
        p3arg2[:, COL_APS] = np.array(
            [1, 1, 1, 1]
        )

        expected = np.array(
            [
                [4, 4, 6 - self.parameter['punish_size'], 6 - self.parameter['punish_size'], 6 - self.parameter['punish_size']],
                [3.2, 7.2, 7.2, 7.2, 7.2],
                [8.4 - self.parameter['punish_size'], 8.4 - self.parameter['punish_size'], 12.4 - self.parameter['punish_size'], 12.4 - self.parameter['punish_size'], 12.4 - self.parameter['punish_size']],
                [16, 18 - self.parameter['punish_size'], 18 - self.parameter['punish_size'], 18 - self.parameter['punish_size'], 18 - self.parameter['punish_size']],
            ]
        )
        
        actual = model_helper.get_members_gain(p3arg1, p3arg2, self.parameter)
        self.assertEquals(np.sum(expected == actual), num_members*num_groups)

        p4arg1 = arg1
        p4arg1[:, :, COL_AC] = np.array(
            [
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1],
            ]
        )
        p4arg1[:, :, COL_AS] = np.array(
            [
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
            ]
        )

        p4arg2 = arg2
        p4arg2[:, COL_APC] = np.array(
            [1, 1, 1, 1]
        )
        p4arg2[:, COL_APS] = np.array(
            [1, 1, 1, 1]
        )

        expected = np.array(
            [
                [4 - self.parameter['punish_size'], 4 - self.parameter['punish_size'], 4 - self.parameter['punish_size'], 4 - self.parameter['punish_size'], 4 - self.parameter['punish_size']],
                [5.2 - self.parameter['punish_size'], 9.2 - self.parameter['punish_size'] - self.parameter['punish_size'], 9.2 - self.parameter['punish_size'] - self.parameter['punish_size'], 9.2 - self.parameter['punish_size'] - self.parameter['punish_size'], 9.2 - self.parameter['punish_size'] - self.parameter['punish_size']],
                [6.4, 8.4 - self.parameter['punish_size'], 12.4 - self.parameter['punish_size'] - self.parameter['punish_size'], 12.4 - self.parameter['punish_size'] - self.parameter['punish_size'], 12.4 - self.parameter['punish_size'] - self.parameter['punish_size']],
                [16, 16, 18 - self.parameter['punish_size'], 18 - self.parameter['punish_size'], 18 - self.parameter['punish_size']],
            ]
        )
        
        actual = model_helper.get_members_gain(p4arg1, p4arg2, self.parameter)
        self.assertEquals(np.sum(expected == actual), num_members*num_groups)

    def test_get_leaders_gain(self):
        # パターンを描くのが辛いためグループ数と成員数は縮小版でテスト
        num_groups = 4
        num_members = 5
        arg1 = np.zeros((num_groups, num_members, NUM_COLUMN))
        arg2 = np.zeros((num_groups, NUM_COLUMN))

        p1arg1 = arg1
        p1arg1[:, :, COL_AC] = np.array(
            [
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1],
            ]
        )
        p1arg1[:, :, COL_AS] = np.array(
            [
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1],
            ]
        )

        p1arg2 = arg2
        p1arg2[:, COL_APC] = np.array(
            [0, 0, 0, 0]
        )
        p1arg2[:, COL_APS] = np.array(
            [0, 0, 0, 0]
        )

        expected = np.array(
            [0, 2, 4, 10]
        )
        
        actual = model_helper.get_leaders_gain(p1arg1, p1arg2, self.parameter)
        self.assertEquals(np.sum(expected == actual), num_groups)

        p2arg1 = arg1
        p2arg1[:, :, COL_AC] = np.array(
            [
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1],
            ]
        )
        p2arg1[:, :, COL_AS] = np.array(
            [
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1],
            ]
        )

        p2arg2 = arg2
        p2arg2[:, COL_APC] = np.array(
            [1, 1, 1, 1]
        )
        p2arg2[:, COL_APS] = np.array(
            [0, 0, 0, 0]
        )

        expected = np.array(
            [-10, -6, -2, 10]
        )
        
        actual = model_helper.get_leaders_gain(p2arg1, p2arg2, self.parameter)
        self.assertEquals(np.sum(expected == actual), num_groups)

        p3arg1 = arg1
        p3arg1[:, :, COL_AC] = np.array(
            [
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1],
            ]
        )
        p3arg1[:, :, COL_AS] = np.array(
            [
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1],
            ]
        )

        p3arg2 = arg2
        p3arg2[:, COL_APC] = np.array(
            [0, 0, 0, 0]
        )
        p3arg2[:, COL_APS] = np.array(
            [1, 1, 1, 1]
        )

        expected = np.array(
            [-10, -6, -2, 10]
        )
        
        actual = model_helper.get_leaders_gain(p3arg1, p3arg2, self.parameter)
        self.assertEquals(np.sum(expected == actual), num_groups)

        p4arg1 = arg1
        p4arg1[:, :, COL_AC] = np.array(
            [
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1],
            ]
        )
        p4arg1[:, :, COL_AS] = np.array(
            [
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1],
            ]
        )

        p4arg2 = arg2
        p4arg2[:, COL_APC] = np.array(
            [1, 1, 1, 1]
        )
        p4arg2[:, COL_APS] = np.array(
            [1, 1, 1, 1]
        )

        expected = np.array(
            [-20, -14, -8, 10]
        )
        
        actual = model_helper.get_leaders_gain(p4arg1, p4arg2, self.parameter)
        self.assertEquals(np.sum(expected == actual), num_groups)

    def test_softmax_2dim(self):
        # softmaxで算出される値を1つ1つ考えるのはめんどくさいので
        # - 出力値が足して1
        # - 出力値は正
        # のみのテストを行う

        arg = np.array(
            [
                [10, 15, 20, 25],
                [15, 25, 35, 45],
                [30, 35, 40, 45],
            ]
        )

        # 少数誤差があるためか見かけ上は足して1になるが比較が通らない.そのため範囲でassert
        expected_area = [0.9999, 1.0001]
        actual = model_helper.softmax_2dim(arg)

        self.assertEquals(np.sum(actual > 0), arg.shape[0] * arg.shape[1])
        self.assertEquals(np.sum(np.sum(actual, axis=1) > expected_area[0]), arg.shape[0])
        self.assertEquals(np.sum(np.sum(actual, axis=1) < expected_area[1]), arg.shape[0])

    def test_softmax_1dim(self):
        # softmaxで算出される値を1つ1つ考えるのはめんどくさいので
        # - 出力値が足して1
        # - 出力値は正
        # のみのテストを行う

        arg = np.array(
            [10, 15, 20, 25]
        )

        # 少数誤差があるためか見かけ上は足して1になるが比較が通らない.そのため範囲でassert
        expected_area = [0.9999, 1.0001]
        actual = model_helper.softmax_1dim(arg)

        self.assertEquals(np.sum(actual >= 0), arg.shape[0])
        self.assertTrue(np.sum(actual) > expected_area[0])
        self.assertTrue(np.sum(actual) < expected_area[1])
    
    def test_get_leaders_mask(self):
        # 引数listの動作確認
        arg1 = [0 for i in range(NUM_GROUPS)]

        expected = np.zeros((NUM_GROUPS, NUM_MEMBERS), dtype=bool)
        expected[:, 0] = True
        actual = model_helper.get_leaders_mask(arg1)
        
        self.assertEquals(np.sum(actual == expected), NUM_GROUPS * NUM_MEMBERS)

        # 引数np.arrayの動作確認
        arg1 = np.array([0 for i in range(NUM_GROUPS)])

        expected = np.zeros((NUM_GROUPS, NUM_MEMBERS),dtype=bool)
        expected[:, 0] = True
        actual = model_helper.get_leaders_mask(arg1)
        
        self.assertEquals(np.sum(actual == expected), NUM_GROUPS * NUM_MEMBERS)

        expected = np.zeros

    def test_get_groups_mask(self):
        # 引数listの動作確認
        arg1 = [0 for i in range(NUM_GROUPS)]

        expected = np.ones((NUM_GROUPS, NUM_MEMBERS), dtype=bool)
        expected[:, 0] = False
        actual = model_helper.get_groups_mask(arg1)
        
        self.assertEquals(np.sum(actual == expected), NUM_GROUPS * NUM_MEMBERS)

        # 引数np.arrayの動作確認
        arg1 = np.array([0 for i in range(NUM_GROUPS)])

        expected = np.ones((NUM_GROUPS, NUM_MEMBERS), dtype=bool)
        expected[:, 0] = False
        actual = model_helper.get_groups_mask(arg1)
        
        self.assertEquals(np.sum(actual == expected), NUM_GROUPS * NUM_MEMBERS)
    
if __name__ == "__main__":
    unittest.main()